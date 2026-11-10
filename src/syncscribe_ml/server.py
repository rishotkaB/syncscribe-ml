"""HTTP-сервис контракта 2.0 для SyncBack: тонкая обёртка над contract.run, логика анализа та же.

    pip install -e .[server]
    uvicorn syncscribe_ml.server:app --host 0.0.0.0 --port 8000

POST /v1/analyze — тело: JSON запроса 2.0; ответ: 200 и JSON ответа 2.0. Ошибки запроса и модели
                   возвращаются в теле (status=failed, error.code invalid_request | pipeline_error),
                   как у run(). HTTP-ошибки — только интеграционные: 400 не JSON-объект, 401 неверный ключ,
                   429 все слоты заняты (SyncBack повторит после Retry-After).
GET  /health     — 200, когда сервис принимает запросы: промпты прочитаны, модели прогреты
                   (если SYNCSCRIBE_ML_WARMUP=1). Без авторизации.

Параметры run() берутся из переменных окружения, SyncBack их не передаёт и не знает:
    SYNCSCRIBE_ML_MODEL           модель LLM (по умолчанию LLM_MODEL или qwen3:8b, как у run_contract.py)
    SYNCSCRIBE_ML_WORKERS         блоков параллельно внутри одного запроса (1)
    SYNCSCRIBE_ML_TOP_K           кусков источника на блок (3)
    SYNCSCRIBE_ML_MODE            hybrid | dense | bm25 (hybrid)
    SYNCSCRIBE_ML_FAST            1 — сначала короткий вызов «только вердикт» (1)
    SYNCSCRIBE_ML_ADDS            1 — искать правки «добавить» (1)
    SYNCSCRIBE_ML_MAX_CONCURRENT  запросов одновременно, сверх — 429 (1)
    SYNCSCRIBE_ML_RETRY_AFTER     секунд в Retry-After при 429 (30)
    SYNCSCRIBE_ML_API_KEY         если задан, нужен заголовок Authorization: Bearer <ключ>
    SYNCSCRIBE_ML_WARMUP          1 — загрузить модель эмбеддингов на старте, а не на первом запросе (0)
Провайдер LLM и эмбеддинги настраиваются как раньше: LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, EMBED_BACKEND.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .contract import CONTRACT_VERSION, Prompts, run

logger = logging.getLogger("syncscribe_ml.server")

MODES = ("hybrid", "dense", "bm25")


def _int(name: str, default: int) -> int:
    value = int(os.environ.get(name) or default)
    if value < 1:
        raise ValueError(f"{name} должен быть не меньше 1, а не {value}")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model: str
    workers: int
    top_k: int
    mode: str
    fast: bool
    adds: bool
    max_concurrent: int
    retry_after: int
    api_key: str
    warmup: bool

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.environ.get("SYNCSCRIBE_ML_MODE") or "hybrid"
        if mode not in MODES:
            raise ValueError(f"SYNCSCRIBE_ML_MODE должен быть одним из {MODES}, а не {mode!r}")
        return cls(
            model=os.environ.get("SYNCSCRIBE_ML_MODEL") or os.environ.get("LLM_MODEL") or "qwen3:8b",
            workers=_int("SYNCSCRIBE_ML_WORKERS", 1),
            top_k=_int("SYNCSCRIBE_ML_TOP_K", 3),
            mode=mode,
            fast=_flag("SYNCSCRIBE_ML_FAST", True),
            adds=_flag("SYNCSCRIBE_ML_ADDS", True),
            max_concurrent=_int("SYNCSCRIBE_ML_MAX_CONCURRENT", 1),
            retry_after=_int("SYNCSCRIBE_ML_RETRY_AFTER", 30),
            api_key=os.environ.get("SYNCSCRIBE_ML_API_KEY", ""),
            warmup=_flag("SYNCSCRIBE_ML_WARMUP", False),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Всё, что может не загрузиться (промпты, модель эмбеддингов), проверяем до приёма запросов:
    # пока lifespan не завершился, /health недоступен, и SyncBack не шлёт задачи в неготовый сервис.
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.prompts = Prompts.default()
    app.state.slots = asyncio.Semaphore(settings.max_concurrent)
    if settings.warmup:
        from .retrieval import embed

        await asyncio.to_thread(embed, ["прогрев"])
    logger.info("сервис готов: модель %s, слотов %d", settings.model, settings.max_concurrent)
    yield


app = FastAPI(title="syncscribe-ml", version=CONTRACT_VERSION, lifespan=lifespan)


def _authorized(header: str, api_key: str) -> bool:
    return hmac.compare_digest(header.encode(), f"Bearer {api_key}".encode())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "contract_version": CONTRACT_VERSION}


@app.post("/v1/analyze")
async def analyze(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    if settings.api_key and not _authorized(request.headers.get("authorization", ""), settings.api_key):
        return JSONResponse({"detail": "неверный ключ"}, status_code=401)
    try:
        payload = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"detail": "тело не JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"detail": "тело должно быть JSON-объектом запроса 2.0"}, status_code=400)

    slots: asyncio.Semaphore = request.app.state.slots
    # Проверка и захват идут без await между ними: в одном цикле событий гонки здесь нет.
    if slots.locked():
        return JSONResponse({"detail": "все слоты заняты"}, status_code=429,
                            headers={"Retry-After": str(settings.retry_after)})
    async with slots:
        # run() синхронный и долгий: в отдельном потоке, чтобы /health и 429 отвечали во время прогона.
        response = await asyncio.to_thread(
            run, payload, settings.model, request.app.state.prompts, fast=settings.fast, top_k=settings.top_k,
            mode=settings.mode, adds=settings.adds, workers=settings.workers,
        )
    # Только идентификатор и счётчики: текст документа и источников в лог не пишем.
    logger.info("request_id=%s status=%s suggestions=%d flags=%d seconds=%s",
                request.headers.get("x-request-id") or response.get("request_id"), response["status"],
                len(response["suggestions"]), len(response["flags"]), response["meta"].get("seconds_total"))
    return JSONResponse(response)
