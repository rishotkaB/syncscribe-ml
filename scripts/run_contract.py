"""Контракт 2.0 с SyncBack: JSON-запрос (документ блоками + источник) → JSON-ответ с правками.

    python scripts/run_contract.py data/contract/example_request.json -o out.json
    python scripts/run_contract.py request.json -o out.json --no-add          # без правок «добавить»
    python scripts/run_contract.py request.json -o out.json --full            # без короткого вызова «только вердикт»
    python scripts/run_contract.py request.json -o out.json --labels labels.json   # точность и полнота по разметке

Формат входа и выхода — в src/syncscribe_ml/contract.py и docs/contract_v2.md.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from syncscribe_ml.contract import run
from syncscribe_ml.llm import add_llm_args, configure, describe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, help="JSON-запрос от SyncBack")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Куда сохранить ответ (по умолчанию <запрос>.response.json)")
    add_llm_args(parser)
    parser.add_argument("--full", action="store_true", help="Сразу полный вызов на каждый блок (медленнее)")
    parser.add_argument("--no-add", action="store_true", help="Не искать правки «добавить»")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--workers", type=int, default=1, help="Сколько блоков проверять параллельно (vLLM: 8–16)")
    parser.add_argument("--labels", type=Path, default=None, help="Разметка {block_id: contradicts|sources_conflict}")
    args = parser.parse_args()
    configure(args)
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    request = json.loads(args.request.read_text(encoding="utf-8"))
    print(f"Модель {args.model} ({describe()}), блоков: {len(request.get('document', {}).get('blocks', []))}")
    response = run(request, args.model, fast=not args.full, top_k=args.top_k, mode=args.mode,
                   adds=not args.no_add, workers=args.workers)
    out = args.output or args.request.with_suffix(".response.json")
    out.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    if response["status"] != "completed":
        print(f"Ошибка: {response['error']}")
        sys.exit(1)
    m = response["meta"]
    print(f"Вызовов: коротких {m.get('quick', 0)}, полных {m.get('full', 0)}, проверок спора {m.get('conflict', 0)}, "
          f"проверок добавления {m.get('add_checks', 0)}. Время {m['seconds_total']}s, из них в LLM {m['llm_seconds']}s")
    kinds = {}
    for s in response["suggestions"]:
        kinds[s["change_type"]] = kinds.get(s["change_type"], 0) + 1
    print(f"Правок: {len(response['suggestions'])} {kinds}, пометок «проверьте вручную»: {len(response['flags'])}")
    if m.get("errors"):
        print(f"Ошибок на отдельных блоках: {m['errors']}")
    for s in response["suggestions"]:
        print(f"\n[{s['suggestion_id']} {s['change_type']} {s['target_block_id']} conf={s['confidence_score']}]")
        if s["old_text"]:
            print(f"  было:  {s['old_text']}")
        else:
            print(f"  после: {s['insert_after']}")
        print(f"  стало: {s['new_text']}\n  {s['rationale']}")
    for f in response["flags"]:
        print(f"\n[{f['flag_id']} ПРОВЕРЬТЕ ВРУЧНУЮ {f['target_block_id']}] {f['old_text']}")
        for q in f["quotes"]:
            print(f"  {q['source']}: «{q['text']}»")

    if args.labels:
        gold = json.loads(args.labels.read_text(encoding="utf-8"))
        truth = set(gold)
        got = {s["target_block_id"] for s in response["suggestions"] if s["change_type"] == "modify"}
        got |= {f["target_block_id"] for f in response["flags"]}
        tp, fp, fn = len(got & truth), len(got - truth), len(truth - got)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"\nПо разметке (modify + пометки): нашли {tp} из {len(truth)}, ложных {fp}. Точность {p:.0%}, полнота {r:.0%}")
        if fp:
            print("Ложные блоки:", ", ".join(sorted(got - truth, key=lambda b: int(b.split('-')[-1]))))
        if fn:
            print("Пропущены:", ", ".join(sorted(truth - got, key=lambda b: int(b.split('-')[-1]))))
    print(f"\nОтвет сохранён: {out}")


if __name__ == "__main__":
    main()
