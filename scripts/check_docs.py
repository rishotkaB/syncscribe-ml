"""Сквозной прогон этапов 2–4: две версии источника + производный документ → что править.

Пример:
    python scripts/check_docs.py data/diff/admin_guide_v1.md data/diff/admin_guide_v2.md data/diff/user_guide.md
    python scripts/check_docs.py ... --all          # без поиска, перебор всех пар (как раньше), для сравнения
    python scripts/check_docs.py ... --json out.json  # ещё и сохранить результат в формате SyncBack
    python scripts/check_docs.py ... --fast          # сначала короткий вызов «да/нет», правка только для «да»
    python scripts/check_docs.py ... --labels data/large/labels.json  # посчитать точность и полноту по разметке

Как устроено:
    1. diff: изменённые разделы источника → пары (БЫЛО, СТАЛО);
    2. поиск: для каждого изменения top-k похожих абзацев производного документа (BGE-M3 + BM25);
    3. LLM: затронут ли каждый кандидат, и черновик правки;
    4. раздел удалён целиком → LLM не зовём, похожие абзацы помечаем «проверьте вручную».
"""

import argparse
import json
import sys
import time
from pathlib import Path

from syncscribe_ml.diff import parse_markdown, to_source_pairs
from syncscribe_ml.llm import PROMPTS_DIR, add_llm_args, analyze, classify, configure, load_prompt
from syncscribe_ml.parse import load_document
from syncscribe_ml.retrieval import FragmentIndex, deletion_flags, is_deletion, pair_candidates
from syncscribe_ml import syncback

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("derived", help="Производный документ, который проверяем")
    add_llm_args(parser)
    parser.add_argument("--prompt", default=PROMPTS_DIR / "impact_v2.md", type=Path)
    parser.add_argument("--top-k", type=int, default=5, help="Сколько кандидатов на одно изменение источника отдаём в LLM")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--all", action="store_true", help="Без поиска: каждое изменение × каждый абзац")
    parser.add_argument("--json", type=Path, default=None, help="Сохранить правки в файл в формате ответа SyncBack")
    parser.add_argument("--fast", action="store_true", help="Сначала короткий вызов «да/нет», полный только для «да»")
    parser.add_argument("--quick-prompt", default=PROMPTS_DIR / "impact_quick_v1.md", type=Path)
    parser.add_argument("--labels", type=Path, default=None, help="Разметка {paragraph, impacted}: посчитать точность и полноту")
    args = parser.parse_args()
    configure(args)
    sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    pairs = to_source_pairs(load_document(args.old), load_document(args.new))
    derived = parse_markdown(load_document(args.derived))
    # Фрагмент = абзац производного документа; заголовки не проверяем.
    heading_paragraphs = {u.paragraph for u in derived.units if u.kind == "heading"}
    fragment_ids = [i for i in range(len(derived.paragraphs)) if i not in heading_paragraphs]
    fragments = [derived.paragraphs[i] for i in fragment_ids]
    index = None if args.all else FragmentIndex(fragments, mode=args.mode)
    system_prompt = load_prompt(args.prompt)
    quick_prompt = load_prompt(args.quick_prompt) if args.fast else None
    print(f"Изменённых разделов: {len(pairs)}, абзацев в производном документе: {len(fragments)}\n")

    hits_by_fragment: dict[int, list] = {}
    flags_by_fragment: dict[int, list] = {}
    calls = quick_calls = 0
    llm_seconds = 0.0
    for pair in pairs:
        where = pair.section[-1] if pair.section else "(без раздела)"
        if args.all:
            candidates = [(j, None) for j in range(len(fragments))]
        else:
            candidates = [(h.index, h) for h in pair_candidates(index, pair, k=args.top_k)]

        if is_deletion(pair):
            hits = [h for _, h in candidates if h is not None]
            for h in deletion_flags(hits):
                flags_by_fragment.setdefault(h.index, []).append(where)
            scores = ", ".join(f"абзац {fragment_ids[h.index]}: dense={h.dense}, bm25={h.bm25:.1f}" for h in hits if h.bm25 is not None)
            print(f"[{where[:30]:<30}] раздел удалён, LLM не зовём. Кандидаты: {scores or '—'}")
            continue

        for j, h in candidates:
            score = f" dense={h.dense:.2f}" if h is not None and h.dense is not None else ""
            try:
                quick = ""
                if args.fast:
                    quick_calls += 1
                    yes, latency = classify(args.model, quick_prompt, pair.source_old, pair.source_new, fragments[j])
                    llm_seconds += latency
                    if not yes:
                        print(f"[{where[:30]:<30}] абзац {fragment_ids[j]:>3}: - (быстро {latency:.1f}s){score}")
                        continue
                    quick = f"быстро {latency:.1f}s + "
                calls += 1
                result, latency, _ = analyze(args.model, system_prompt, pair.source_old, pair.source_new, fragments[j])
                llm_seconds += latency
            except Exception as exc:  # один упавший вызов не должен останавливать прогон
                print(f"[{where[:30]:<30}] абзац {fragment_ids[j]:>3}: ОШИБКА {repr(exc).splitlines()[0]}")
                continue
            mark = "ЗАТРОНУТ" if result.impacted else "-"
            print(f"[{where[:30]:<30}] абзац {fragment_ids[j]:>3}: {mark} ({quick}{latency:.1f}s){score}")
            if result.impacted:
                hits_by_fragment.setdefault(j, []).append((where, result))

    brute = sum(not is_deletion(p) for p in pairs) * len(fragments)
    quick_info = f" + коротких «да/нет»: {quick_calls}" if args.fast else ""
    print(f"\nВызовов LLM: полных {calls}{quick_info} (перебор всех пар дал бы {brute})")
    print(f"Время: всего {time.perf_counter() - started:.0f}s, из них в LLM {llm_seconds:.0f}s")
    print("\n=== Итог: что править в", args.derived)
    for j, fragment in enumerate(fragments):
        if j not in hits_by_fragment and j not in flags_by_fragment:
            continue
        print(f"\nАбзац {fragment_ids[j]}: {fragment}")
        for where, r in hits_by_fragment.get(j, []):
            print(f"  из-за [{where}]\n  → {r.new_text}\n  цитата: {r.rationale_quote}")
        for where in flags_by_fragment.get(j, []):
            print(f"  ⚠ раздел источника «{where}» удалён — проверьте вручную, черновика правки нет")
    touched = set(hits_by_fragment) | set(flags_by_fragment)
    print(f"\nЗатронуто абзацев: {len(touched)} из {len(fragments)}")

    if args.labels:
        # Сверяем по тексту абзаца. Абзацы без разметки (вводные) считаем незатронутыми.
        gold = {l["paragraph"]: l["impacted"] for l in json.loads(args.labels.read_text(encoding="utf-8"))}
        truth = {j for j, f in enumerate(fragments) if gold.get(f)}
        found = set(hits_by_fragment)
        tp, fp, fn = len(found & truth), len(found - truth), len(truth - found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"\nПо разметке: нашли {tp} из {len(truth)}, ложных {fp}, пропущено {fn}. "
              f"Точность {precision:.0%}, полнота {recall:.0%}")
        missed = sorted(truth - found)
        if missed:
            print("Пропущены абзацы:", ", ".join(str(fragment_ids[j]) for j in missed))

    if args.json:
        # section_ref = заголовок раздела ПРОИЗВОДНОГО документа, где стоит абзац (так требует SyncBack).
        section_of = {u.paragraph: (u.section[-1] if u.section else "") for u in derived.units}
        suggestions, flags = [], []
        for j in sorted(touched):
            p = fragment_ids[j]
            for _, r in hits_by_fragment.get(j, []):
                suggestions.append(syncback.suggestion(section_of.get(p, ""), derived.raw[p], r))
            for where in flags_by_fragment.get(j, []):
                flags.append(syncback.deletion_flag(section_of.get(p, ""), derived.raw[p], where))
        result = syncback.response(suggestions, flags)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сохранено: {args.json} (правок: {len(result['suggestions'])}, пометок: {len(flags)})")


if __name__ == "__main__":
    main()
