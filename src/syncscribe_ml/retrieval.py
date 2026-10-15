"""Этап 4: для каждого изменения источника найти несколько фрагментов, которые оно может затронуть.

Без поиска пришлось бы звать LLM на каждую пару «изменение × абзац» — на реальных документах это тысячи вызовов.
Поиск дешёвый и отсекает почти всё; LLM вызывается только на top-k кандидатах.

Гибридный поиск, потому что у каждого способа свои слепые зоны:
- dense (BGE-M3, эмбеддинги): ловит смысл и перефраз, в том числе через язык
  («Session tokens expire» ↔ «сессия завершается»), но плохо различает конкретные числа и названия;
- BM25 (лексический): ловит точные совпадения слов, названий, чисел («Confluence», «8 ядер»),
  но не видит синонимов.
Два списка сливаем через RRF (reciprocal rank fusion): складываем 1/(60 + место) из каждого списка.
RRF не требует подгонять шкалы оценок друг к другу, поэтому это стандартный выбор.

Локально эмбеддинги считает Ollama (`ollama pull bge-m3`), индекс в памяти.
В сервисе то же самое ляжет в Qdrant (dense + sparse векторы), логика поиска не меняется.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .diff import SourcePair

Mode = Literal["hybrid", "dense", "bm25"]
EMBED_MODEL = "bge-m3"
RRF_K = 60


# ---------- BM25 ----------

def tokenize(text: str) -> list[str]:
    """Слова и числа. Грубый стемминг: обрезаем слово до 5 символов («ядрами», «ядра» → «ядра»/«ядрам»).

    Для русского это заметно лучше, чем без стемминга, и не тянет зависимостей.
    Числа не обрезаем: «15» и «150» — разные вещи.
    """
    tokens = re.findall(r"\w+", text.lower().replace("ё", "е"))
    return [t if t.isdigit() else t[:5] for t in tokens]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg_len = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len))
            out.append(s)
        return out


# ---------- dense ----------

def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    import ollama

    return [list(v) for v in ollama.embed(model=model, input=texts).embeddings]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


# ---------- гибрид ----------

@dataclass
class Hit:
    index: int            # номер фрагмента в списке, который индексировали
    rrf: float            # итоговая оценка (чем больше, тем выше в выдаче)
    dense: float | None   # косинус BGE-M3, если считали
    bm25: float | None


class FragmentIndex:
    def __init__(self, fragments: list[str], mode: Mode = "hybrid", embed_model: str = EMBED_MODEL):
        self.fragments = fragments
        self.mode = mode
        self.embed_model = embed_model
        self.bm25 = BM25(fragments) if mode in ("hybrid", "bm25") else None
        self.vectors = embed(fragments, embed_model) if mode in ("hybrid", "dense") else None

    def search(self, query: str, k: int = 3) -> list[Hit]:
        n = len(self.fragments)
        dense = None
        if self.vectors is not None:
            q = embed([query], self.embed_model)[0]
            dense = [cosine(v, q) for v in self.vectors]
        bm25 = self.bm25.scores(query) if self.bm25 else None

        rrf = [0.0] * n
        for scores in (dense, bm25):
            if scores is None:
                continue
            ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for place, i in enumerate(ranked):
                # Фрагмент без единого общего слова не получает очков от BM25, иначе он «найдётся» по месту в списке.
                if scores is bm25 and scores[i] == 0:
                    continue
                rrf[i] += 1 / (RRF_K + place + 1)

        order = sorted(range(n), key=lambda i: rrf[i], reverse=True)
        return [
            Hit(i, rrf[i], dense[i] if dense else None, bm25[i] if bm25 else None)
            for i in order[:k]
            if rrf[i] > 0
        ]


def pair_query(pair: SourcePair) -> str:
    """Запрос для поиска: только изменённые единицы (и до, и после), а не весь раздел.

    Весь раздел размывает запрос: в нём много неизменённого текста, и найдутся фрагменты про него.
    Старый текст в запросе нужен, потому что производный документ написан по старой версии
    («8 ядер», а не «16 vCPU»). Заголовок раздела добавляем: фрагменты часто на него ссылаются.
    """
    parts: list[str] = []
    if pair.section:
        parts.append(pair.section[-1])
    for c in pair.changes:
        for unit in (c.old, c.new):
            if unit is not None and unit.text not in parts:
                parts.append(unit.text)
    return "\n".join(parts)


# ---------- удалённые разделы ----------

# Пороги «фрагмент опирается на удалённый текст». Подобраны на глаз, не откалиброваны:
# у BGE-M3 близкие по смыслу абзацы обычно дают косинус 0.6+, посторонние 0.3–0.5.
DELETED_MIN_DENSE = 0.6
DELETED_MIN_BM25 = 3.0


def is_deletion(pair: SourcePair) -> bool:
    """Раздел удалён целиком: в новой версии нечего цитировать, черновик правки модель не напишет."""
    return not pair.source_new.strip()


def deletion_flags(hits: list[Hit]) -> list[Hit]:
    """Какие из найденных фрагментов пометить «источник удалён, проверьте вручную».

    Это правило, а не LLM: что делать с текстом про удалённую функцию (убрать, заменить, оставить),
    решает человек. Наша задача — не дать такому фрагменту остаться незамеченным.
    """
    if any(h.dense is not None for h in hits):
        return [h for h in hits if h.dense is not None and h.dense >= DELETED_MIN_DENSE]
    return [h for h in hits if h.bm25 is not None and h.bm25 >= DELETED_MIN_BM25]
