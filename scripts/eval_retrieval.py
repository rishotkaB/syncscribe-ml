"""Насколько хорошо поиск находит затронутый фрагмент.

Берём фрагменты из обоих eval-наборов как одну «базу документации» (~50 абзацев на похожие темы,
это нарочно трудные соседи). Для каждого примера с impacted=true ищем по изменению его источника
и смотрим, на каком месте оказался его фрагмент.

    recall@k — доля примеров, где нужный фрагмент попал в top-k (только они дойдут до LLM);
    MRR      — среднее 1/место, чувствует, насколько высоко он стоит.

Пример:
    python scripts/eval_retrieval.py              # гибрид, нужна `ollama pull bge-m3`
    python scripts/eval_retrieval.py --mode bm25  # только лексический, без модели
"""

import argparse
import json
import sys
from pathlib import Path

from syncscribe_ml.diff import to_source_pairs
from syncscribe_ml.retrieval import FragmentIndex, pair_query

ROOT = Path(__file__).resolve().parent.parent
SETS = ["data/eval/gold_v0.jsonl", "data/eval/gold_v1_holdout.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--show-misses", type=int, default=3, help="Показать примеры, где фрагмент не попал в top-k")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    examples = [json.loads(l) for s in SETS for l in (ROOT / s).read_text(encoding="utf-8").splitlines() if l.strip()]
    fragments = list(dict.fromkeys(ex["fragment"] for ex in examples))  # без повторов, порядок сохраняем
    index = FragmentIndex(fragments, mode=args.mode, embed_model=args.embed_model)

    ranks = []  # место нужного фрагмента (1 = первый), None = не нашёлся вообще
    rows = []
    for ex in examples:
        if not ex["impacted"]:
            continue
        pair = to_source_pairs(ex["source_old"], ex["source_new"])[0]
        hits = index.search(pair_query(pair), k=len(fragments))
        target = fragments.index(ex["fragment"])
        rank = next((place for place, h in enumerate(hits, 1) if h.index == target), None)
        ranks.append(rank)
        rows.append((ex, rank, hits))

    n = len(ranks)
    print(f"Режим: {args.mode} · фрагментов в базе: {len(fragments)} · запросов: {n}\n")
    for k in (1, 3, 5, 10):
        hit = sum(1 for r in ranks if r is not None and r <= k)
        print(f"recall@{k:<2} {hit / n:6.1%}  ({hit}/{n})")
    print(f"MRR       {sum(1 / r for r in ranks if r) / n:.3f}")

    misses = [(ex, rank, hits) for ex, rank, hits in rows if rank is None or rank > 3]
    for ex, rank, hits in misses[: args.show_misses]:
        print(f"\n--- {ex['id']}: нужный фрагмент на месте {rank or '—'}")
        print(f"изменение: {ex['source_old']}  →  {ex['source_new']}")
        print(f"нужный:    {ex['fragment']}")
        for h in hits[:3]:
            print(f"  top: {fragments[h.index][:90]}")


if __name__ == "__main__":
    main()
