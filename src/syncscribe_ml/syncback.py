"""Перевод результата нашего конвейера в формат ответа SyncBack (бэкенд коллег).

SyncBack ждёт от модели JSON вида {"suggestions": [...]} (app/infrastructure/llm/schemas.py)
и дальше сам проверяет и применяет правки (app/workers/pipeline/suggestion_mapper.py).
Главное условие маппера: old_text должен ДОСЛОВНО найтись в тексте документа,
иначе правка отбрасывается. Поэтому в old_text кладём абзац так, как он записан в файле.

Пометки «раздел источника удалён» в их формате выразить нечем (у них только add / modify / delete),
поэтому они идут отдельным ключом "flags": pydantic-схема SyncBack лишние ключи пропускает.
"""

from __future__ import annotations

from .schema import ImpactResult


def suggestion(section_ref: str, old_text: str, result: ImpactResult) -> dict:
    """Одна правка «абзац стал неверным» → правка modify."""
    rationale = result.analysis.strip()
    if result.rationale_quote:
        rationale += f" Источник: «{result.rationale_quote}»"
    return {
        "section_ref": section_ref,
        "change_type": "modify",
        "old_text": old_text,
        "new_text": result.new_text,
        "insert_after": "",
        "rationale": rationale,
        # confidence модели не откалиброван (у qwen3:8b всегда 0.9–0.95), показывать его как
        # «уверенность ИИ» нечестно. Появится вердикт судьи (этап 5) — подставим его.
        "confidence_score": None,
    }


def deletion_flag(section_ref: str, old_text: str, source_section: str) -> dict:
    return {
        "section_ref": section_ref,
        "old_text": old_text,
        "reason": "source_section_deleted",
        "rationale": f"Раздел источника «{source_section}» удалён. Проверьте фрагмент вручную, черновика правки нет.",
    }


def dedupe(suggestions: list[dict]) -> list[dict]:
    """Один абзац могут затронуть несколько изменений источника, и модель предложит одну и ту же правку дважды.

    Одинаковые правки (тот же old_text и new_text) склеиваем, обоснования объединяем.
    Разные правки одного абзаца оставляем обе: какую принять, решает человек. Объединять их
    в одну правку — отдельная задача (см. риски в документе к ревью).
    """
    merged: dict[tuple[str, str], dict] = {}
    for s in suggestions:
        key = (s["old_text"], s["new_text"])
        if key in merged:
            if s["rationale"] not in merged[key]["rationale"]:
                merged[key]["rationale"] += "\n" + s["rationale"]
        else:
            merged[key] = dict(s)
    return list(merged.values())


def response(suggestions: list[dict], flags: list[dict]) -> dict:
    return {"suggestions": dedupe(suggestions), "flags": flags}
