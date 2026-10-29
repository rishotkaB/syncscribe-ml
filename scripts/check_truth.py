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
       sources_conflict (источники спорят: правки нет, только цитаты из каждого).
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from syncscribe_ml import syncback
from syncscribe_ml.diff import parse_markdown
from syncscribe_ml.llm import analyze_truth, classify_truth, load_prompt
from syncscribe_ml.parse import load_document
from syncscribe_ml.retrieval import FragmentIndex
from syncscribe_ml.truth import NEEDS_ATTENTION, bad_quotes, load_chunks, quoted_sources, search_text

ROOT = Path(__file__).resolve().parent.parent
MARK = {"consistent": "-", "not_covered": "нет в источнике", "contradicts": "ПРОТИВОРЕЧИТ", "sources_conflict": "ИСТОЧНИКИ СПОРЯТ"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("derived", help="Документ, который проверяем и правим")
    parser.add_argument("sources", nargs="+", help="Файлы-источники истины")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--prompt", default=ROOT / "prompts/truth_v1.md", type=Path)
    parser.add_argument("--quick-prompt", default=ROOT / "prompts/truth_quick_v1.md", type=Path)
    parser.add_argument("--top-k", type=int, default=3, help="Сколько кусков источников показываем модели на один абзац")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--fast", action="store_true", help="Сначала короткий вызов «только вердикт», полный — если надо править")
    parser.add_argument("--labels", type=Path, default=None, help="Разметка {paragraph, expected}: точность и полнота")
    parser.add_argument("--json", type=Path, default=None, help="Сохранить правки в формате ответа SyncBack")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    chunks = load_chunks({s: load_document(s) for s in args.sources})
    index = FragmentIndex([search_text(c) for c in chunks], mode=args.mode)
    derived = parse_markdown(load_document(args.derived))
    headings = {u.paragraph for u in derived.units if u.kind == "heading"}
    fragment_ids = [i for i in range(len(derived.paragraphs)) if i not in headings]
    prompt = load_prompt(args.prompt)
    quick_prompt = load_prompt(args.quick_prompt) if args.fast else None
    print(f"Кусков в источниках: {len(chunks)} ({len(args.sources)} файл.), абзацев в документе: {len(fragment_ids)}\n")

    verdicts: dict[int, str] = {}
    results: dict[int, tuple] = {}  # абзац → (TruthResult, найденные куски)
    calls = quick_calls = 0
    llm_seconds = 0.0
    hallucinated = 0
    for p in fragment_ids:
        fragment = derived.paragraphs[p]
        found = [chunks[h.index] for h in index.search(fragment, k=args.top_k)]
        if not found:
            verdicts[p] = "not_covered"
            continue
        texts = [c.for_model() for c in found]
        try:
            quick = ""
            if args.fast:
                quick_calls += 1
                verdict, latency = classify_truth(args.model, quick_prompt, texts, fragment)
                llm_seconds += latency
                if verdict not in NEEDS_ATTENTION:
                    verdicts[p] = verdict
                    print(f"абзац {p:>3}: {MARK[verdict]} (быстро {latency:.1f}s)")
                    continue
                quick = f"быстро {latency:.1f}s + "
            calls += 1
            result, latency = analyze_truth(args.model, prompt, texts, fragment)
            llm_seconds += latency
        except Exception as exc:  # один упавший вызов не должен останавливать прогон
            print(f"абзац {p:>3}: ОШИБКА {repr(exc).splitlines()[0]}")
            continue
        verdicts[p] = result.verdict
        results[p] = (result, found)
        bad = bad_quotes(result, found) if result.verdict in NEEDS_ATTENTION else []
        hallucinated += len(bad)
        warn = f" ⚠ цитат не из источника: {len(bad)}" if bad else ""
        print(f"абзац {p:>3}: {MARK[result.verdict]} ({quick}{latency:.1f}s){warn}")

    print(f"\nВызовов LLM: полных {calls}" + (f" + коротких: {quick_calls}" if args.fast else ""))
    print(f"Время: всего {time.perf_counter() - started:.0f}s, из них в LLM {llm_seconds:.0f}s")
    print("Вердикты:", ", ".join(f"{k} {v}" for k, v in Counter(verdicts.values()).most_common()))
    if hallucinated:
        print(f"Цитат, которых нет дословно в источнике: {hallucinated}")

    print("\n=== Итог: что проверить в", args.derived)
    for p, (r, found) in results.items():
        if r.verdict not in NEEDS_ATTENTION:
            continue
        print(f"\nАбзац {p}: {derived.paragraphs[p]}")
        if r.verdict == "contradicts":
            print(f"  → {r.new_text}")
        else:
            print("  ⚠ источники противоречат друг другу, правку не предлагаем:")
        for file, text in quoted_sources(r, found):
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
        for p, (r, found) in results.items():
            quotes = quoted_sources(r, found)
            if r.verdict == "contradicts":
                suggestions.append(syncback.truth_suggestion(section_of.get(p, ""), derived.raw[p], r.analysis, r.new_text, quotes))
            elif r.verdict == "sources_conflict":
                flags.append(syncback.conflict_flag(section_of.get(p, ""), derived.raw[p], quotes))
        out = syncback.response(suggestions, flags)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сохранено: {args.json} (правок: {len(out['suggestions'])}, пометок: {len(flags)})")


if __name__ == "__main__":
    main()
