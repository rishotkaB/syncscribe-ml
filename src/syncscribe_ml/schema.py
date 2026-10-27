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
