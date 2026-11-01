"""Реализация порта LLMClient из SyncBack поверх нашего конвейера (syncscribe_ml.service.suggest).

Файл предназначен для репозитория SyncBack: app/infrastructure/llm/ml_pipeline_client.py.
Как подключить — integrations/syncback/README.md. API, воркер, маппер, экспорт и UI не меняются.

Чем отличается от OpenAICompatibleClient:
- не один вызов модели на весь документ, а поиск + короткий вызов на каждый абзац (подробнее в README нашего репо);
- модель та же: OpenAI-совместимый API из LLM_ENDPOINT / LLM_MODEL / LLM_API_KEY;
- эмбеддинги BGE-M3: EMBED_BASE_URL (если не задан — LLM_ENDPOINT) и EMBED_MODEL (по умолчанию bge-m3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

from app.core.config import Settings, get_settings
from app.domain.exceptions import LLMInvalidResponseError, LLMTimeoutError
from app.domain.interfaces.llm_client import LLMSuggestionBatch, LLMSuggestionItem

logger = logging.getLogger("syncscribe.llm.ml_pipeline")


class MLPipelineClient:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        if self._settings.llm_auth == "oauth_gigachat":
            raise ValueError("LLM_PROVIDER=ml_pipeline пока поддерживает только LLM_AUTH=bearer или none")
        # Наш конвейер читает настройки модели из окружения (так же работают его скрипты).
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_BASE_URL"] = self._settings.llm_endpoint
        if self._settings.llm_auth == "bearer" and self._settings.llm_api_key:
            os.environ["LLM_API_KEY"] = self._settings.llm_api_key

    async def generate_suggestions(
        self, document_text: str, source_text: str, document_format: str
    ) -> LLMSuggestionBatch:
        from syncscribe_ml.service import suggest

        try:
            # Конвейер синхронный (много коротких HTTP-вызовов), в воркере уводим его в поток.
            result = await asyncio.to_thread(
                suggest, document_text, source_text, document_format, self._settings.llm_model
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise LLMTimeoutError(f"Модель или сервис эмбеддингов недоступны: {type(exc).__name__}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMInvalidResponseError(f"Модель ответила HTTP {exc.response.status_code}") from exc
        except ValueError as exc:  # ответ не прошёл схему
            raise LLMInvalidResponseError(f"Ответ модели не прошёл схему: {type(exc).__name__}") from exc

        items = [
            LLMSuggestionItem(
                section_ref=s["section_ref"],
                change_type=s["change_type"],
                old_text=s["old_text"],
                new_text=s["new_text"],
                rationale=s["rationale"],
                confidence_score=s["confidence_score"],
                insert_after=s["insert_after"] or None,
            )
            for s in result["suggestions"]
        ]
        # Пометки «источник противоречит сам себе» в формате SyncBack выразить нечем:
        # пока только в логе и raw_response. Как показывать их в UI — вопрос к продукту.
        logger.info(
            "ml_pipeline: правок %d, пометок %d", len(items), len(result["flags"]),
            extra={"stats": result["stats"]},
        )
        return LLMSuggestionBatch(items=items, raw_response=json.dumps(result, ensure_ascii=False))

    async def health_check(self) -> bool:
        headers = {"Authorization": f"Bearer {self._settings.llm_api_key}"} if self._settings.llm_api_key else {}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._settings.llm_endpoint.rstrip('/')}/models", headers=headers)
                return response.is_success
        except (httpx.HTTPError, OSError):
            logger.warning("Проверка доступности модели не прошла", exc_info=True)
            return False
