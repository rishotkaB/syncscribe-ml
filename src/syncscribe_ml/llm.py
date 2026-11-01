"""Вызов модели: одна функция _chat, за ней два провайдера.

    ollama  — локально (по умолчанию), библиотека ollama;
    openai  — любой OpenAI-совместимый API: vLLM на нашем сервере, Ollama через /v1,
              облачные API (OpenRouter и т. п.).

Провайдер выбирается флагами скриптов (--provider, --base-url) или переменными окружения
LLM_PROVIDER, LLM_BASE_URL. Ключ — только из переменной LLM_API_KEY, чтобы он не попадал в историю команд.

ВАЖНО: облачные API — только для разработки и только на синтетических данных.
Данные клиентов под NDA уходят только в on-prem модель (vLLM у клиента).

Промпты, схемы ответов и скрипты метрик от провайдера не зависят.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel

from .schema import ConflictCheck, ImpactResult, QuickResult, TruthQuick, TruthResult

PROVIDERS = ("ollama", "openai")


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


# ---------- выбор провайдера ----------

def add_llm_args(parser) -> None:
    """Общие флаги для всех скриптов, которые зовут модель."""
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3:8b"))
    parser.add_argument("--provider", choices=PROVIDERS, default=os.environ.get("LLM_PROVIDER", "ollama"),
                        help="ollama — локально; openai — любой OpenAI-совместимый API (vLLM, облако)")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""),
                        help="Адрес OpenAI-совместимого API, например http://localhost:8000/v1 (ключ — в LLM_API_KEY)")


def configure(args) -> None:
    """Запомнить выбор провайдера из флагов скрипта."""
    os.environ["LLM_PROVIDER"] = args.provider
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.provider == "openai" and not os.environ.get("LLM_BASE_URL"):
        raise SystemExit("Для --provider openai нужен --base-url или переменная LLM_BASE_URL")


def describe() -> str:
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    return provider if provider == "ollama" else f"{provider} ({os.environ.get('LLM_BASE_URL', '')})"


def _ollama(model: str, messages: list[dict], schema: type[BaseModel], think: bool) -> tuple[str, str]:
    import ollama

    response = ollama.chat(
        model=model,
        messages=messages,
        # Ollama ограничивает генерацию этой схемой: невалидный JSON невозможен.
        format=schema.model_json_schema(),
        # think=True: модель (Qwen3) сначала рассуждает отдельным текстом, потом отвечает JSON.
        think=think,
        # temperature=0: одинаковый вход даёт одинаковый ответ, прогоны сравнимы.
        options={"temperature": 0},
    )
    return response.message.content, response.message.thinking or ""


def _openai(model: str, messages: list[dict], schema: type[BaseModel], think: bool) -> tuple[str, str]:
    import httpx  # приходит вместе с библиотекой ollama, отдельная зависимость не нужна

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        # Structured outputs: vLLM и большинство облачных API ограничивают генерацию схемой, как Ollama.
        "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
    }
    if "qwen3" in model.lower():
        # Qwen3 на vLLM / Ollama: включить или выключить рассуждения. Остальные API лишний ключ игнорируют.
        body["chat_template_kwargs"] = {"enable_thinking": think}
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"} if os.environ.get("LLM_API_KEY") else {}
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    response = httpx.post(url, json=body, headers=headers, timeout=180)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return message.get("content") or "", message.get("reasoning_content") or message.get("reasoning") or ""


def _parse(content: str, schema: type[BaseModel]) -> BaseModel:
    """Если API схему не соблюдает (обернул JSON в текст или ```), берём от первой { до последней }."""
    try:
        return schema.model_validate_json(content)
    except ValueError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise
        return schema.model_validate(json.loads(match.group(0)))


def _chat(model: str, system_prompt: str, user: str, schema: type[BaseModel], think: bool = False):
    """→ (ответ по схеме, время в секундах, рассуждения модели или "")."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    call = _openai if os.environ.get("LLM_PROVIDER", "ollama") == "openai" else _ollama
    started = time.perf_counter()
    content, thinking = call(model, messages, schema, think)
    return _parse(content, schema), time.perf_counter() - started, thinking


# ---------- режим diff: пара (изменение источника, фрагмент) ----------

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
    return _chat(model, system_prompt, build_user_message(source_old, source_new, fragment), ImpactResult, think)


def classify(model: str, system_prompt: str, source_old: str, source_new: str, fragment: str) -> tuple[bool, float]:
    """Быстрый режим, шаг 1: только «затронут / нет». Возвращает (impacted, время в секундах).

    Почти всё время полного вызова уходит на генерацию analysis и new_text. Здесь ответ —
    `{"impacted": false}`, поэтому вызов в разы короче. Полный analyze зовём только для «да».
    """
    result, elapsed, _ = _chat(model, system_prompt, build_user_message(source_old, source_new, fragment), QuickResult)
    return result.impacted, elapsed


# ---------- режим «источник истины» ----------

def build_truth_message(chunks: list[str], fragment: str) -> str:
    """Куски источников нумеруем: по номеру модель ссылается на кусок в цитате, а мы проверяем, что цитата из него."""
    sources = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    return f"ИСТОЧНИКИ ИСТИНЫ:\n{sources}\n\nФРАГМЕНТ:\n{fragment}"


def analyze_truth(model: str, system_prompt: str, chunks: list[str], fragment: str) -> tuple[TruthResult, float]:
    result, elapsed, _ = _chat(model, system_prompt, build_truth_message(chunks, fragment), TruthResult)
    return result, elapsed


def classify_truth(model: str, system_prompt: str, chunks: list[str], fragment: str) -> tuple[str, float]:
    result, elapsed, _ = _chat(model, system_prompt, build_truth_message(chunks, fragment), TruthQuick)
    return result.verdict, elapsed


def confirm_conflict(model: str, system_prompt: str, fragment: str, quotes: list[tuple[str, str]]) -> tuple[bool, float]:
    """Спор источников подтверждаем отдельным коротким вопросом: модель видит только фрагмент и цитаты.

    Правило «один файл подтверждает, другой противоречит» срабатывает и тогда, когда модель по ошибке
    сочла противоречием кусок про ДРУГОЙ объект («размер файла» вместо «размера выгрузки»).
    Узкий вопрос «это про одно и то же и значения разные?» модель решает надёжнее.
    """
    lines = "\n".join(f"- ({file}) {text}" for file, text in quotes)
    user = f"ФРАГМЕНТ:\n{fragment}\n\nЦИТАТЫ ИЗ ИСТОЧНИКОВ:\n{lines}"
    result, elapsed, _ = _chat(model, system_prompt, user, ConflictCheck)
    return result.same_object and result.values_differ, elapsed
