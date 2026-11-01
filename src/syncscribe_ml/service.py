"""Точка входа для продукта: документ + источник (текст) → правки в формате SyncBack.

SyncBack (бэкенд коллег) отдаёт в модель документ и ОДИН источник целиком: release notes, протокол встречи,
страницу. Старой версии источника нет, поэтому это наш режим «источник истины» с одним файлом
(src/syncscribe_ml/truth.py). Адаптер под их порт LLMClient лежит в integrations/syncback/.

    from syncscribe_ml.service import suggest
    result = suggest(document_text, source_text, "docx", model="qwen3:8b")
    result["suggestions"]  # [{section_ref, change_type, old_text, new_text, insert_after, rationale, confidence_score}]
    result["flags"]        # пометки без правки: «источник противоречит сам себе», см. syncback.conflict_flag

Главное требование маппера SyncBack: old_text должен ДОСЛОВНО найтись в document_text, а для docx
ещё и лежать внутри одного абзаца (без «\\n»). Поэтому абзацы берём кусками самого document_text,
а перед возвратом отбрасываем всё, что этому условию не удовлетворяет.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import syncback
from .diff import parse_markdown
from .retrieval import FragmentIndex
from .truth import Chunk, NEEDS_ATTENTION, Prompts, by_file, judge, search_text

logger = logging.getLogger("syncscribe_ml.service")
MARKDOWN_FORMATS = {"md", "markdown"}
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")


@dataclass
class Paragraph:
    text: str     # дословный кусок исходного текста
    section: str  # заголовок раздела, в котором стоит абзац ("" если не определили)


def _looks_like_heading(line: str) -> bool:
    """В plain text из docx стили заголовков потеряны. Короткая строка без знака препинания в конце
    и не пункт списка почти всегда заголовок. Ошибка здесь безопасна: заголовок просто не проверяем
    и подставляем его в section_ref."""
    line = line.strip()
    return 0 < len(line) <= 80 and not re.search(r"[.!?:;…,)»\"]$", line) and not _LIST_ITEM.match(line)


def split_document(text: str, document_format: str) -> list[Paragraph]:
    """Абзацы, которые проверяем (без заголовков).

    md          — наш parse_markdown (заголовки «#», абзацы через пустую строку);
    docx, txt   — у SyncBack это строки, склеенные через «\\n»: один абзац = одна строка.
                  Если в тексте есть пустые строки (обычный txt), абзацы — блоки между ними.
    """
    if document_format.lower() in MARKDOWN_FORMATS:
        doc = parse_markdown(text)
        headings = {u.paragraph for u in doc.units if u.kind == "heading"}
        section_of = {u.paragraph: (u.section[-1] if u.section else "") for u in doc.units}
        return [Paragraph(doc.raw[i], section_of.get(i, "")) for i in range(len(doc.raw)) if i not in headings]

    blocks = re.split(r"\n\s*\n", text) if re.search(r"\n\s*\n", text) else text.split("\n")
    paragraphs, section = [], ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if "\n" not in block and _looks_like_heading(block):
            section = block
            continue
        paragraphs.append(Paragraph(block, section))
    return paragraphs


def source_chunks(source_text: str, name: str = "источник") -> list[Chunk]:
    """Источник из SyncBack — просто текст (формат не известен), режем так же, как txt.

    Список (типичные release notes) режем на пункты: один пункт — одно изменение, так точнее
    и поиск, и цитата в обосновании правки.
    """
    fmt = "md" if re.search(r"^#{1,6}\s", source_text, re.M) else "txt"
    chunks = []
    for p in split_document(source_text, fmt):
        lines = [l.strip() for l in p.text.split("\n") if l.strip()]
        if len(lines) > 1 and all(_LIST_ITEM.match(l) for l in lines):
            chunks += [Chunk(name, p.section, l) for l in lines]
        else:
            chunks.append(Chunk(name, p.section, p.text))
    return chunks


def suggest(
    document_text: str,
    source_text: str,
    document_format: str,
    model: str,
    *,
    fast: bool = True,
    top_k: int = 3,
    mode: str = "hybrid",
    prompts: Prompts | None = None,
    workers: int | None = None,
) -> dict:
    """→ {"suggestions": [...], "flags": [...], "stats": {...}}.

    Переводит в формат SyncBack: contradicts → modify с цитатой источника в rationale;
    sources_conflict → пометка в flags без правки (решение Alex 29.10: истину не предлагаем).
    workers — сколько абзацев проверять параллельно (ML_PIPELINE_WORKERS, по умолчанию 1).
    Ollama по умолчанию обрабатывает запросы по одному, vLLM — пачкой: на сервере ставим 8–16.
    """
    prompts = prompts or Prompts.default()
    paragraphs = split_document(document_text, document_format)
    chunks = source_chunks(source_text)
    stats: Counter = Counter(paragraphs=len(paragraphs), source_chunks=len(chunks))
    if not paragraphs or not chunks:
        return {"suggestions": [], "flags": [], "stats": dict(stats)}

    index = FragmentIndex([search_text(c) for c in chunks], mode=mode)
    single_paragraph = document_format.lower() in {"docx", "doc"}
    suggestions, flags = [], []
    workers = workers or int(os.environ.get("ML_PIPELINE_WORKERS", "1"))

    done = [0]

    def check(p: Paragraph):
        groups = by_file([chunks[h.index] for h in index.search(p.text, k=top_k)])
        j = judge(model, prompts, p.text, groups, fast)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(paragraphs):
            logger.info("проверено абзацев: %d из %d", done[0], len(paragraphs))
        return j

    with ThreadPoolExecutor(max_workers=workers) as pool:
        judgements = list(pool.map(check, paragraphs))  # порядок абзацев сохраняется

    for p, j in zip(paragraphs, judgements):
        stats.update(j.calls)
        stats["llm_seconds"] += round(j.seconds, 2)
        stats[j.verdict] += 1
        if j.verdict not in NEEDS_ATTENTION:
            continue
        # Страховка под маппер SyncBack: то, что он всё равно отбросит, не отдаём.
        old_text = p.text.strip()
        if old_text not in document_text or (single_paragraph and "\n" in old_text):
            stats["dropped_not_verbatim"] += 1
            continue
        quotes = [("", text) for _, text in j.quotes]  # источник один, имя файла в обосновании лишнее
        if j.verdict == "contradicts":
            suggestions.append(syncback.truth_suggestion(p.section, old_text, j.result.analysis, j.result.new_text, quotes))
        else:
            flags.append(syncback.conflict_flag(p.section, old_text, quotes))
    out = syncback.response(suggestions, flags)
    out["stats"] = dict(stats)
    return out
