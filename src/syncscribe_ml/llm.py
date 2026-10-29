"""Один вызов модели на одну пару (изменение источника, фрагмент).

Локально работаем через Ollama. Позже на сервере будет vLLM, но меняется
только эта функция: промпт, схема и скрипт метрик остаются теми же.
"""

import time
from pathlib import Path

import ollama
from pydantic import BaseModel

from .schema import ImpactResult, QuickResult, TruthQuick, TruthResult


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_user_message(source_old: str, source_new: str, fragment: str) -> str:
    # Источник идёт перед фрагментом: на сервере много фрагментов проверяются
    # против одного изменения, и общее начало промпта vLLM закэширует.
    return (
        f"ИСТОЧНИК (БЫЛО):\n{source_old}\n\n"
        f"ИСТОЧНИК (СТАЛО):\n{source_new}\n\n"
        f"ФРАГМЕНТ:\n{fragment}"
    )


def analyze(
    model: str,
    system_prompt: str,
    source_old: str,
    source_new: str,
    fragment: str,
    think: bool = False,
) -> tuple[ImpactResult, float, str]:
    """Возвращает (ответ модели, время ответа в секундах, рассуждения модели или "")."""
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(source_old, source_new, fragment)},
        ],
        # Ollama ограничивает генерацию этой схемой: невалидный JSON невозможен.
        format=ImpactResult.model_json_schema(),
        # think=True: модель (Qwen3) сначала рассуждает отдельным текстом, потом отвечает JSON.
        # Обычно точнее, но медленнее. По умолчанию выключено, рассуждением служит поле analysis.
        think=think,
        # temperature=0: одинаковый вход даёт одинаковый ответ, прогоны сравнимы.
        options={"temperature": 0},
    )
    elapsed = time.perf_counter() - started
    thinking = response.message.thinking or ""
    return ImpactResult.model_validate_json(response.message.content), elapsed, thinking


def classify(model: str, system_prompt: str, source_old: str, source_new: str, fragment: str) -> tuple[bool, float]:
    """Быстрый режим, шаг 1: только «затронут / нет». Возвращает (impacted, время в секундах).

    Почти всё время полного вызова уходит на генерацию analysis и new_text. Здесь ответ —
    `{"impacted": false}`, поэтому вызов в разы короче. Полный analyze зовём только для «да».
    """
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(source_old, source_new, fragment)},
        ],
        format=QuickResult.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    elapsed = time.perf_counter() - started
    return QuickResult.model_validate_json(response.message.content).impacted, elapsed


# ---------- режим «источник истины» ----------

def build_truth_message(chunks: list[str], fragment: str) -> str:
    """Куски источников нумеруем: по номеру модель ссылается на кусок в цитате, а мы проверяем, что цитата из него."""
    sources = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    return f"ИСТОЧНИКИ ИСТИНЫ:\n{sources}\n\nФРАГМЕНТ:\n{fragment}"


def _chat(model: str, system_prompt: str, user: str, schema: type[BaseModel]):
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        format=schema.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    return schema.model_validate_json(response.message.content), time.perf_counter() - started


def analyze_truth(model: str, system_prompt: str, chunks: list[str], fragment: str) -> tuple[TruthResult, float]:
    return _chat(model, system_prompt, build_truth_message(chunks, fragment), TruthResult)


def classify_truth(model: str, system_prompt: str, chunks: list[str], fragment: str) -> tuple[str, float]:
    result, elapsed = _chat(model, system_prompt, build_truth_message(chunks, fragment), TruthQuick)
    return result.verdict, elapsed
