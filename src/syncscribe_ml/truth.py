"""Режим «источник истины»: правим документ по одному или нескольким файлам с верными данными.

В отличие от режима diff, старой версии источника нет, и мы не знаем, что именно поменялось.
Поэтому идём от документа: для каждого его абзаца ищем подходящие куски источников и спрашиваем
модель, согласуется ли абзац с ними. У модели четыре ответа:
    consistent        — согласуется, не трогаем;
    not_covered       — источники об этом молчат, не трогаем (иначе модель «исправит» всё подряд);
    contradicts       — противоречит, предлагаем правку с цитатой;
    sources_conflict  — источники противоречат друг другу: правку НЕ предлагаем,
                        показываем цитаты из каждого источника, решает человек.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .diff import parse_markdown
from .metrics import quote_in_source
from .schema import TruthResult

NEEDS_ATTENTION = ("contradicts", "sources_conflict")


@dataclass
class Chunk:
    file: str      # имя файла источника, чтобы показать человеку, откуда цитата
    section: str   # заголовок раздела
    text: str

    def for_model(self) -> str:
        where = f"{self.file}, раздел «{self.section}»" if self.section else self.file
        return f"({where}) {self.text}"


def load_chunks(sources: dict[str, str]) -> list[Chunk]:
    """{имя файла: текст} → куски = абзацы (заголовки отдельно не берём, они идут в подпись куска)."""
    chunks = []
    for name, text in sources.items():
        doc = parse_markdown(text)
        section_of = {u.paragraph: (u.section[-1] if u.section else "") for u in doc.units}
        headings = {u.paragraph for u in doc.units if u.kind == "heading"}
        for i, paragraph in enumerate(doc.paragraphs):
            if i not in headings:
                chunks.append(Chunk(Path(name).name, section_of.get(i, ""), paragraph))
    return chunks


def search_text(chunk: Chunk) -> str:
    """По чему ищем кусок: заголовок раздела + текст (фрагменты часто называют раздел, а не повторяют его)."""
    return f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text


def bad_quotes(result: TruthResult, chunks: list[Chunk]) -> list[str]:
    """Цитаты, которых нет дословно в куске, на который сослалась модель. Дешёвая проверка на выдумку."""
    bad = []
    for q in result.quotes:
        ok = 1 <= q.source <= len(chunks) and quote_in_source(q.text, chunks[q.source - 1].text)
        if not ok:
            bad.append(q.text)
    return bad


def quoted_sources(result: TruthResult, chunks: list[Chunk]) -> list[tuple[str, str]]:
    """[(файл, цитата)] для показа человеку."""
    return [(chunks[q.source - 1].file if 1 <= q.source <= len(chunks) else "?", q.text) for q in result.quotes]
