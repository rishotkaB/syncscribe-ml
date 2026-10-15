"""Сквозной прогон этапов 2–4: две версии источника + производный документ → что править.

Пример:
    python scripts/check_docs.py data/diff/admin_guide_v1.md data/diff/admin_guide_v2.md data/diff/user_guide.md
    python scripts/check_docs.py ... --all          # без поиска, перебор всех пар (как раньше), для сравнения

Как устроено:
    1. diff: изменённые разделы источника → пары (БЫЛО, СТАЛО);
    2. поиск: для каждой пары top-k похожих абзацев производного документа (BGE-M3 + BM25);
    3. LLM: затронут ли каждый кандидат, и черновик правки;
    4. раздел удалён целиком → LLM не зовём, похожие абзацы помечаем «проверьте вручную».
"""

import argparse
import sys
from pathlib import Path

from syncscribe_ml.diff import parse_markdown, to_source_pairs
from syncscribe_ml.llm import analyze, load_prompt
from syncscribe_ml.parse import load_document
from syncscribe_ml.retrieval import FragmentIndex, deletion_flags, is_deletion, pair_query

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("derived", help="Производный документ, который проверяем")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--prompt", default=ROOT / "prompts/impact_v2.md", type=Path)
    parser.add_argument("--top-k", type=int, default=3, help="Сколько кандидатов на одно изменение отдаём в LLM")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--all", action="store_true", help="Без поиска: каждое изменение × каждый абзац")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pairs = to_source_pairs(load_document(args.old), load_document(args.new))
    derived = parse_markdown(load_document(args.derived))
    # Фрагмент = абзац производного документа; заголовки не проверяем.
    heading_paragraphs = {u.paragraph for u in derived.units if u.kind == "heading"}
    fragment_ids = [i for i in range(len(derived.paragraphs)) if i not in heading_paragraphs]
    fragments = [derived.paragraphs[i] for i in fragment_ids]
    index = None if args.all else FragmentIndex(fragments, mode=args.mode)
    system_prompt = load_prompt(args.prompt)
    print(f"Изменённых разделов: {len(pairs)}, абзацев в производном документе: {len(fragments)}\n")

    hits_by_fragment: dict[int, list] = {}
    flags_by_fragment: dict[int, list] = {}
    calls = 0
    for pair in pairs:
        where = pair.section[-1] if pair.section else "(без раздела)"
        if args.all:
            candidates = [(j, None) for j in range(len(fragments))]
        else:
            candidates = [(h.index, h) for h in index.search(pair_query(pair), k=args.top_k)]

        if is_deletion(pair):
            hits = [h for _, h in candidates if h is not None]
            for h in deletion_flags(hits):
                flags_by_fragment.setdefault(h.index, []).append(where)
            scores = ", ".join(f"абзац {fragment_ids[h.index]}: dense={h.dense}, bm25={h.bm25:.1f}" for h in hits if h.bm25 is not None)
            print(f"[{where[:30]:<30}] раздел удалён, LLM не зовём. Кандидаты: {scores or '—'}")
            continue

        for j, h in candidates:
            calls += 1
            try:
                result, latency, _ = analyze(args.model, system_prompt, pair.source_old, pair.source_new, fragments[j])
            except Exception as exc:  # один упавший вызов не должен останавливать прогон
                print(f"[{where[:30]:<30}] абзац {fragment_ids[j]:>2}: ОШИБКА {repr(exc).splitlines()[0]}")
                continue
            mark = "ЗАТРОНУТ" if result.impacted else "-"
            score = f" dense={h.dense:.2f}" if h is not None and h.dense is not None else ""
            print(f"[{where[:30]:<30}] абзац {fragment_ids[j]:>2}: {mark} ({latency:.1f}s){score}")
            if result.impacted:
                hits_by_fragment.setdefault(j, []).append((where, result))

    print(f"\nВызовов LLM: {calls} (перебор всех пар дал бы {sum(not is_deletion(p) for p in pairs) * len(fragments)})")
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


if __name__ == "__main__":
    main()
