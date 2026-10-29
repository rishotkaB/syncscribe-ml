"""Схема ответа модели.

Эта схема одновременно:
1) передаётся в Ollama/vLLM как JSON Schema, и модель физически не может
   вернуть ничего другого (guided decoding);
2) валидирует ответ на нашей стороне через pydantic.

Порядок полей важен: модель генерирует их сверху вниз. Поэтому сначала
короткий анализ («подумай»), потом решение, потом правка и цитата.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ImpactResult(BaseModel):
    analysis: str = Field(
        description="1-3 предложения: что изменилось в источнике и касается ли это фрагмента"
    )
    impacted: bool = Field(description="Затрагивает ли изменение источника этот фрагмент")
    confidence: float = Field(ge=0, le=1, description="Уверенность в решении impacted, от 0 до 1")
    change_type: Literal["factual", "structural", "semantic", "none"] = Field(
        description="Тип изменения; none, если фрагмент не затронут"
    )
    new_text: str = Field(
        description="Полный новый текст фрагмента; пустая строка, если не затронут"
    )
    rationale_quote: str = Field(
        description="Дословная цитата из НОВОЙ версии источника, обосновывающая правку; пустая строка, если не затронут"
    )


class QuickResult(BaseModel):
    """Быстрый режим: только решение. Ответ в несколько токенов вместо ~150, правку пишем потом и только для «да»."""

    impacted: bool = Field(description="Сделало ли изменение источника фрагмент неверным")


# ---------- режим «источник истины» ----------
# Старой версии источника нет: модель сверяет абзац с найденными кусками источников истины.

Verdict = Literal["consistent", "contradicts", "not_covered", "sources_conflict"]


class Quote(BaseModel):
    source: int = Field(description="Номер куска источника из запроса: [1], [2], ...")
    text: str = Field(description="Дословная цитата из этого куска")


class TruthResult(BaseModel):
    analysis: str = Field(description="1-3 предложения: что утверждает фрагмент и что говорят источники")
    verdict: Verdict = Field(
        description="consistent — согласуется; contradicts — противоречит, нужна правка; "
        "not_covered — источники об этом молчат; sources_conflict — источники противоречат друг другу"
    )
    new_text: str = Field(description="Исправленный фрагмент, только для contradicts; иначе пустая строка")
    quotes: list[Quote] = Field(
        description="Для contradicts — цитата, на которой основана правка; для sources_conflict — "
        "цитаты из каждого спорящего источника; иначе пустой список"
    )


class TruthQuick(BaseModel):
    """Быстрый режим: только вердикт, полный вызов — для contradicts и sources_conflict."""

    verdict: Verdict
