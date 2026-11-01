"""Прогон модели на eval-наборе.

Пример:
    python scripts/run_eval.py --model qwen3:8b
    python scripts/run_eval.py --model qwen3:8b --prompt prompts/impact_v1.md --think
    python scripts/run_eval.py --edits-from results/<run>.jsonl   # пересобрать edits.md без вызова модели

Что получится в results/:
    <run>.jsonl     — каждый пример с ответом модели (для разбора ошибок);
    <run>.summary.json — метрики;
    <run>.edits.md  — ожидаемые и предложенные правки рядом, для ручной оценки Edit Usefulness (1–5).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from syncscribe_ml.llm import add_llm_args, analyze, configure, load_prompt
from syncscribe_ml.metrics import compute

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    add_llm_args(parser)
    parser.add_argument("--data", default=ROOT / "data/eval/gold_v0.jsonl", type=Path)
    parser.add_argument("--prompt", default=ROOT / "prompts/impact_v2.md", type=Path)
    parser.add_argument("--think", action="store_true", help="Включить режим рассуждений модели (медленнее)")
    parser.add_argument("--limit", type=int, default=None, help="Прогнать только первые N примеров")
    parser.add_argument("--edits-from", type=Path, default=None,
                        help="Не запускать модель, а пересобрать edits.md из готового results/<run>.jsonl")
    args = parser.parse_args()
    configure(args)

    # В консоли Windows иначе бывают проблемы с кириллицей.
    sys.stdout.reconfigure(encoding="utf-8")

    if args.edits_from:
        rows = [json.loads(line) for line in args.edits_from.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = args.edits_from.with_suffix(".edits.md")
        if out.exists():  # не затираем файл, в котором уже могут стоять оценки
            out = args.edits_from.with_suffix(".edits.new.md")
        write_edits_review(rows, out)
        print(f"Пересобран {out}")
        return

    examples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        examples = examples[: args.limit]
    system_prompt = load_prompt(args.prompt)

    # Имя прогона: дата + модель + версия промпта (+ think). ":" в имени файла Windows не допускает.
    run_name = f"{datetime.now():%Y%m%d-%H%M}_{re.sub(r'[^A-Za-z0-9.-]', '-', args.model)}_{args.prompt.stem}"
    if args.think:
        run_name += "_think"
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for i, ex in enumerate(examples, 1):
        try:
            result, latency, thinking = analyze(
                args.model, system_prompt, ex["source_old"], ex["source_new"], ex["fragment"], think=args.think
            )
            pred, error = result.model_dump(), None
        except Exception as exc:  # один упавший пример не должен останавливать прогон
            pred, latency, thinking, error = None, None, "", repr(exc).splitlines()[0]

        rows.append({"gold": ex, "pred": pred, "latency": latency, "thinking": thinking, "error": error})
        if pred is None:
            mark = "ERR"
        else:
            mark = "ok " if pred["impacted"] == ex["impacted"] else "MISS"
        said = "—" if pred is None else pred["impacted"]
        print(f"[{i:>2}/{len(examples)}] {mark} {ex['id']} {ex['change_type']:<10} gold={ex['impacted']!s:<5} pred={said!s:<5}"
              + (f" {latency:.1f}s" if latency is not None else f" {error}"))

    summary = compute(rows)
    summary.update({"provider": args.provider, "model": args.model, "prompt": args.prompt.name, "think": args.think, "data": args.data.name})

    with open(out_dir / f"{run_name}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{run_name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_edits_review(rows, out_dir / f"{run_name}.edits.md")

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nРезультаты: results/{run_name}.*")


def write_edits_review(rows: list[dict], path: Path) -> None:
    """Файл для ручной оценки правок: там, где модель и эталон согласны, что фрагмент затронут."""
    lines = ["# Правки для ручной оценки\n", "Поставь оценку 1–5 в строке «Оценка» (5 = можно принять без изменений).\n"]
    for row in rows:
        gold, pred = row["gold"], row["pred"]
        if not (pred and pred["impacted"] and gold["impacted"]):
            continue
        lines += [
            f"## {gold['id']} ({gold['change_type']})\n",
            f"**Источник (было):** {gold['source_old']}\n",
            f"**Источник (стало):** {gold['source_new']}\n",
            f"**Фрагмент:** {gold['fragment']}\n",
            f"**Эталон:** {gold['expected_edit']}\n",
            f"**Модель:** {pred['new_text']}\n",
            f"**Цитата модели:** {pred['rationale_quote']}\n",
            "**Оценка:** \n",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
