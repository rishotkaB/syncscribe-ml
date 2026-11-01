"""Режим «источник истины» на старых eval-наборах: та же разметка, но модель видит только НОВУЮ версию источника.

    python scripts/eval_truth.py                                   # отложенная выборка gold_v1_holdout
    python scripts/eval_truth.py --data data/eval/gold_v0.jsonl

Пример затронут в режиме diff ⇔ фрагмент противоречит новой версии источника. Поэтому разметку берём как есть:
impacted=true → ждём contradicts, impacted=false → ждём consistent или not_covered.
Так сравниваем с режимом diff на тех же примерах: насколько хуже модель без подсказки «вот что изменилось».
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from syncscribe_ml.llm import analyze_truth, load_prompt
from syncscribe_ml.metrics import quote_in_source

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--data", default=ROOT / "data/eval/gold_v1_holdout.jsonl", type=Path)
    parser.add_argument("--prompt", default=ROOT / "prompts/truth_v2.md", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    examples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = load_prompt(args.prompt)
    tp = fp = tn = fn = errors = quotes_bad = 0
    verdicts: Counter = Counter()
    latencies = []
    for i, ex in enumerate(examples, 1):
        try:
            r, latency = analyze_truth(args.model, prompt, [f"(source.md) {ex['source_new']}"], ex["fragment"])
        except Exception as exc:
            errors += 1
            print(f"[{i:>2}/{len(examples)}] ERR  {ex['id']} {repr(exc).splitlines()[0]}")
            continue
        latencies.append(latency)
        verdicts[r.verdict] += 1
        said = r.verdict == "contradicts"
        tp += said and ex["impacted"]
        fp += said and not ex["impacted"]
        fn += not said and ex["impacted"]
        tn += not said and not ex["impacted"]
        if said:
            quotes_bad += sum(not quote_in_source(q.text, ex["source_new"]) for q in r.quotes)
        mark = "ok  " if said == ex["impacted"] else "MISS"
        print(f"[{i:>2}/{len(examples)}] {mark} {ex['id']} gold={ex['impacted']!s:<5} {r.verdict:<16} {latency:.1f}s")

    total = tp + fp + tn + fn + errors
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nAccuracy {(tp + tn) / total:.0%}, Precision {precision:.0%}, Recall {recall:.0%} "
          f"(TP {tp}, FP {fp}, FN {fn}, TN {tn}, ошибок вызова {errors})")
    print("Вердикты:", ", ".join(f"{k} {v}" for k, v in verdicts.most_common()))
    print(f"Цитат не из источника: {quotes_bad}")
    if latencies:
        print(f"Среднее время: {sum(latencies) / len(latencies):.1f}s")


if __name__ == "__main__":
    main()
