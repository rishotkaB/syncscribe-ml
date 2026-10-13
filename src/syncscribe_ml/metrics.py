"""Метрики по результатам прогона.

Каждая запись в rows: {"gold": пример из eval-набора, "pred": ответ модели (dict) или None, "latency": сек}.
pred = None означает, что вызов упал (сеть, таймаут, невалидный ответ):
такой пример считаем ошибкой модели, а не пропускаем, иначе метрики выйдут лучше реальных.
"""

import re
from statistics import median


def normalize(text: str) -> str:
    """Для сравнения цитат: нижний регистр, одинаковые кавычки и пробелы."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[«»“”\"']", '"', text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_source(quote: str, source_new: str) -> bool:
    """Traceability: цитата должна дословно (с точностью до пробелов и кавычек) найтись в новой версии источника."""
    return bool(quote.strip()) and normalize(quote) in normalize(source_new)


def compute(rows: list[dict]) -> dict:
    tp = fp = tn = fn = errors = 0
    by_type: dict[str, dict[str, int]] = {}
    quotes_ok = quotes_total = 0

    for row in rows:
        gold, pred = row["gold"], row["pred"]
        truth = gold["impacted"]
        if pred is None:
            errors += 1
            said = not truth  # упавший вызов = неверный ответ
        else:
            said = pred["impacted"]

        if truth and said:
            tp += 1
        elif truth and not said:
            fn += 1
        elif not truth and said:
            fp += 1
        else:
            tn += 1

        stats = by_type.setdefault(gold["change_type"], {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(truth == said)

        # Цитату проверяем только там, где модель предложила правку.
        if pred is not None and pred["impacted"]:
            quotes_total += 1
            quotes_ok += int(quote_in_source(pred["rationale_quote"], gold["source_new"]))

    n = len(rows)
    latencies = [row["latency"] for row in rows if row["pred"] is not None]
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "errors": errors,
        "accuracy_by_type": {k: v["correct"] / v["total"] for k, v in by_type.items()},
        "traceability": quotes_ok / quotes_total if quotes_total else 0.0,
        "latency_median_s": median(latencies) if latencies else None,
    }
