"""Режим «источник истины»: документ + один или несколько файлов с верными данными → что править.

Пример:
    python scripts/check_truth.py data/large/truth/derived.md data/large/truth/admin_guide.md data/large/truth/faq.md
    python scripts/check_truth.py ... --fast                                  # сначала короткий вызов «только вердикт»
    python scripts/check_truth.py ... --labels data/large/truth/labels.json   # точность и полнота по разметке
    python scripts/check_truth.py ... --json out.json                         # результат в формате SyncBack

Как устроено (подробнее в src/syncscribe_ml/truth.py):
    1. источники режем на куски (абзацы), индексируем (BGE-M3 + BM25);
    2. для каждого абзаца документа ищем top-k кусков источников;
    3. LLM выбирает вердикт: consistent / not_covered (не трогаем), contradicts (правка),
       sources_conflict (источники спорят: правки нет, только цитаты из каждого);
    4. если нашлись куски из разных файлов, модель отвечает по каждому файлу отдельно, а спор
       («один файл подтверждает, другой противоречит») определяет правило truth.combine.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from syncscribe_ml import syncback
from syncscribe_ml.diff import parse_markdown
from syncscribe_ml.llm import PROMPTS_DIR, add_llm_args, configure, load_prompt
from syncscribe_ml.parse import load_document
from syncscribe_ml.retrieval import FragmentIndex
from syncscribe_ml.truth import NEEDS_ATTENTION, Prompts, by_file, judge, load_chunks, search_text

ROOT = Path(__file__).resolve().parent.parent
MARK = {"consistent": "-", "not_covered": "нет в источнике", "contradicts": "ПРОТИВОРЕЧИТ", "sources_conflict": "ИСТОЧНИКИ СПОРЯТ"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("derived", help="Документ, который проверяем и правим")
    parser.add_argument("sources", nargs="+", help="Файлы-источники истины")
    add_llm_args(parser)
    parser.add_argument("--prompt", default=PROMPTS_DIR / "truth_v2.md", type=Path)
    parser.add_argument("--quick-prompt", default=PROMPTS_DIR / "truth_quick_v2.md", type=Path)
    parser.add_argument("--conflict-prompt", default=PROMPTS_DIR / "conflict_v1.md", type=Path)
    parser.add_argument("--top-k", type=int, default=3, help="Сколько кусков источников показываем модели на один абзац")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--fast", action="store_true", help="Сначала короткий вызов «только вердикт», полный — если надо править")
    parser.add_argument("--labels", type=Path, default=None, help="Разметка {paragraph, expected}: точность и полнота")
    parser.add_argument("--json", type=Path, default=None, help="Сохранить правки в формате ответа SyncBack")
    args = parser.parse_args()
    configure(args)
    sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    chunks = load_chunks({s: load_document(s) for s in args.sources})
    index = FragmentIndex([search_text(c) for c in chunks], mode=args.mode)
    derived = parse_markdown(load_document(args.derived))
    headings = {u.paragraph for u in derived.units if u.kind == "heading"}
    fragment_ids = [i for i in range(len(derived.paragraphs)) if i not in headings]
    prompt = load_prompt(args.prompt)
    quick_prompt = load_prompt(args.quick_prompt) if args.fast else None
    conflict_prompt = load_prompt(args.conflict_prompt)
    print(f"Кусков в источниках: {len(chunks)} ({len(args.sources)} файл.), абзацев в документе: {len(fragment_ids)}\n")

    prompts = Prompts(prompt, quick_prompt or "", conflict_prompt)
    verdicts: dict[int, str] = {}
    results: dict[int, tuple] = {}  # абзац → (вердикт, ответ с правкой или None, [(файл, цитата)])
    calls: Counter = Counter()
    llm_seconds = 0.0
    hallucinated = 0
    # Все эмбеддинги запросов одним пакетом до вызовов LLM: Ollama не перегружает модели туда-обратно.
    hits_of = dict(zip(fragment_ids, index.search_many([derived.paragraphs[p] for p in fragment_ids], k=args.top_k)))
    for p in fragment_ids:
        fragment = derived.paragraphs[p]
        groups = by_file([chunks[h.index] for h in hits_of[p]])
        try:
            j = judge(args.model, prompts, fragment, groups, args.fast)
        except Exception as exc:  # один упавший вызов не должен останавливать прогон
            print(f"абзац {p:>3}: ОШИБКА {repr(exc).splitlines()[0]}")
            continue
        calls.update(j.calls)
        llm_seconds += j.seconds
        hallucinated += j.bad_quotes
        verdicts[p] = j.verdict
        if j.calls["full"]:
            results[p] = (j.verdict, j.result, j.quotes)
        warn = f" ⚠ цитат не из источника: {j.bad_quotes}" if j.bad_quotes else ""
        many = f", файлов: {len(groups)}" if len(groups) > 1 else ""
        print(f"абзац {p:>3}: {MARK[j.verdict]} ({' + '.join(j.log) or 'без вызова'}{many}){warn}")

    print(f"\nВызовов LLM: полных {calls['full']}" + (f" + коротких: {calls['quick']}" if args.fast else "")
          + f" + проверок спора: {calls['conflict']} (отклонено {calls['conflict_rejected']})")
    print(f"Время: всего {time.perf_counter() - started:.0f}s, из них в LLM {llm_seconds:.0f}s")
    print("Вердикты:", ", ".join(f"{k} {v}" for k, v in Counter(verdicts.values()).most_common()))
    if hallucinated:
        print(f"Цитат, которых нет дословно в источнике: {hallucinated}")

    print("\n=== Итог: что проверить в", args.derived)
    for p, (verdict, r, quotes) in results.items():
        if verdict not in NEEDS_ATTENTION:
            continue
        print(f"\nАбзац {p}: {derived.paragraphs[p]}")
        if verdict == "contradicts":
            print(f"  → {r.new_text}")
        else:
            print("  ⚠ источники противоречат друг другу, правку не предлагаем:")
        for file, text in quotes:
            print(f"  {file}: «{text}»")

    if args.labels:
        gold = {l["paragraph"]: l["expected"] for l in json.loads(args.labels.read_text(encoding="utf-8"))}
        expected = {p: gold.get(derived.paragraphs[p], "ok") for p in fragment_ids}  # без разметки = не трогать
        truth = {p for p, e in expected.items() if e != "ok"}
        flagged = {p for p, v in verdicts.items() if v in NEEDS_ATTENTION}
        tp, fp, fn = len(flagged & truth), len(flagged - truth), len(truth - flagged)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"\nПо разметке («надо смотреть» = противоречит или спор): нашли {tp} из {len(truth)}, ложных {fp}, "
              f"пропущено {fn}. Точность {precision:.0%}, полнота {recall:.0%}")
        for kind in NEEDS_ATTENTION:
            want = {p for p, e in expected.items() if e == kind}
            same = sum(verdicts.get(p) == kind for p in want)
            print(f"  {kind}: верный вердикт у {same} из {len(want)}")
        if fn:
            print("Пропущены абзацы:", ", ".join(str(p) for p in sorted(truth - flagged)))
        if fp:
            print("Ложные:", ", ".join(str(p) for p in sorted(flagged - truth)))

    if args.json:
        section_of = {u.paragraph: (u.section[-1] if u.section else "") for u in derived.units}
        suggestions, flags = [], []
        for p, (verdict, r, quotes) in results.items():
            if verdict == "contradicts":
                suggestions.append(syncback.truth_suggestion(section_of.get(p, ""), derived.raw[p], r.analysis, r.new_text, quotes))
            elif verdict == "sources_conflict":
                flags.append(syncback.conflict_flag(section_of.get(p, ""), derived.raw[p], quotes))
        out = syncback.response(suggestions, flags)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сохранено: {args.json} (правок: {len(out['suggestions'])}, пометок: {len(flags)})")


if __name__ == "__main__":
    main()
