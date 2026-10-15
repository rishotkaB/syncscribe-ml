"""Этап 3: что именно изменилось между двумя версиями источника.

Идея: режем обе версии на мелкие единицы (предложения, пункты списков, строки таблиц),
запоминаем для каждой раздел, в котором она стоит, и выравниваем две последовательности.
Дальше в LLM уходят только изменённые места, а не весь документ.

Почему не эмбеддинги: косинусная близость «лимит 50 МБ» и «лимит 100 МБ» почти 1,
изменение числа она не видит. Для поиска изменений нужен лексический diff (difflib),
эмбеддинги пригодятся на этапе 4, для поиска затронутых фрагментов.

Виды изменений:
    modified — единица переписана (нашли похожую в новой версии);
    added    — появилась новая;
    deleted  — исчезла;
    moved    — тот же текст, но в другом месте (перестановка шагов или разделов).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

# Если текст раздела длиннее, в LLM отдаём не весь раздел, а только изменённые абзацы.
MAX_CONTEXT_CHARS = 2000

# Порог похожести, при котором удалённую и добавленную единицу считаем одной изменённой.
# 0.5 подобран на наших примерах: замена числа или слова даёт 0.7–0.95,
# разные предложения из одного документа — обычно ниже 0.4.
MODIFIED_THRESHOLD = 0.5

ChangeKind = Literal["modified", "added", "deleted", "moved"]


@dataclass
class Unit:
    """Минимальная единица сравнения."""

    text: str
    section: tuple[str, ...]  # путь заголовков: ("Установка", "Требования")
    paragraph: int            # номер абзаца, чтобы потом отдать LLM весь абзац как контекст
    kind: Literal["heading", "sentence", "list_item", "table_row"]

    @property
    def key(self) -> str:
        """По этому ключу ищем совпадения: регистр, ё и лишние пробелы не считаются изменением."""
        return re.sub(r"\s+", " ", self.text.lower().replace("ё", "е")).strip()


@dataclass
class Document:
    units: list[Unit]
    paragraphs: list[str]  # исходный текст абзацев, индекс = Unit.paragraph


@dataclass
class Change:
    kind: ChangeKind
    old: Unit | None
    new: Unit | None
    inline: str = ""  # пословный diff: «лимит [-50-]{+100+} МБ»

    @property
    def section(self) -> tuple[str, ...]:
        return (self.new or self.old).section


@dataclass
class SourcePair:
    """То, что уходит в LLM: старый и новый текст изменённого места (с контекстом абзаца)."""

    section: tuple[str, ...]
    source_old: str
    source_new: str
    changes: list[Change] = field(default_factory=list)


# ---------- разбор текста на единицы ----------

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# Конец предложения: . ! ? … и пробел, дальше заглавная буква, цифра или открывающая кавычка.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    # «Обновление: 1. Остановите сервис.» режется на «Обновление: 1.» и «Остановите сервис.»:
    # кусок, который кончается голым номером, приклеиваем к следующему.
    merged: list[str] = []
    for part in parts:
        if merged and re.search(r"(?:^|\s)\d{1,3}(?:\.\d{1,3})*\.$", merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def parse_markdown(text: str) -> Document:
    """Markdown (его же отдаёт Docling для docx/pdf) или обычный текст → единицы сравнения.

    Абзацы разделены пустой строкой; заголовки, пункты списков и строки таблиц — отдельные абзацы.
    """
    units: list[Unit] = []
    paragraphs: list[str] = []
    section: list[tuple[int, str]] = []  # стек (уровень, заголовок)
    buffer: list[str] = []

    def path() -> tuple[str, ...]:
        return tuple(title for _, title in section)

    def flush() -> None:
        if not buffer:
            return
        paragraph = " ".join(line.strip() for line in buffer)
        buffer.clear()
        paragraphs.append(paragraph)
        for sentence in split_sentences(paragraph):
            units.append(Unit(sentence, path(), len(paragraphs) - 1, "sentence"))

    def add_single(line: str, kind) -> None:
        paragraphs.append(line)
        units.append(Unit(line, path(), len(paragraphs) - 1, kind))

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
        elif heading := _HEADING.match(line):
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            while section and section[-1][0] >= level:
                section.pop()
            section.append((level, title))
            # Сам заголовок — тоже единица своего раздела: перенумерация «4.» → «5.» должна быть видна.
            add_single(title, "heading")
        elif _TABLE_SEPARATOR.match(line):
            flush()
        elif _TABLE_ROW.match(line):
            flush()
            add_single(line.strip(), "table_row")
        elif _LIST_ITEM.match(line):
            flush()
            add_single(line.strip(), "list_item")
        else:
            buffer.append(line)
    flush()
    return Document(units, paragraphs)


# ---------- выравнивание ----------

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def inline_diff(old: str, new: str) -> str:
    """Пословный diff для человека и для логов: [-удалено-]{+добавлено+}."""
    a, b = old.split(), new.split()
    out: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out += a[i1:i2]
            continue
        if i2 > i1:
            out.append("[-" + " ".join(a[i1:i2]) + "-]")
        if j2 > j1:
            out.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(out)


def _pair_similar(removed: list[Unit], added: list[Unit]) -> tuple[list[tuple[Unit, Unit]], list[Unit], list[Unit]]:
    """Жадно сводим удалённые и добавленные единицы в пары «было → стало» по похожести."""
    candidates = sorted(
        ((similarity(r.key, a.key), i, j) for i, r in enumerate(removed) for j, a in enumerate(added)),
        reverse=True,
    )
    used_r: set[int] = set()
    used_a: set[int] = set()
    pairs: list[tuple[Unit, Unit]] = []
    for score, i, j in candidates:
        if score < MODIFIED_THRESHOLD:
            break
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        pairs.append((removed[i], added[j]))
    rest_r = [u for i, u in enumerate(removed) if i not in used_r]
    rest_a = [u for j, u in enumerate(added) if j not in used_a]
    return pairs, rest_r, rest_a


def diff_documents(old: Document, new: Document) -> list[Change]:
    a = [u.key for u in old.units]
    b = [u.key for u in new.units]
    blocks: list[tuple[list[Unit], list[Unit]]] = []  # (удалённые, добавленные) для каждого неравного участка
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            blocks.append((old.units[i1:i2], new.units[j1:j2]))

    def change(kind: ChangeKind, o: Unit | None, n: Unit | None) -> Change:
        return Change(kind, o, n, inline_diff(o.text, n.text) if kind == "modified" else "")

    changes: list[Change] = []

    # 1. Перестановки: тот же текст удалён в одном месте и добавлен в другом.
    removed_all = [u for r, _ in blocks for u in r]
    added_all = [u for _, ad in blocks for u in ad]
    moved_keys = {u.key for u in removed_all} & {u.key for u in added_all}
    for key in moved_keys:
        old_u = next(u for u in removed_all if u.key == key)
        new_u = next(u for u in added_all if u.key == key)
        changes.append(change("moved", old_u, new_u))

    # 2. Изменения внутри одного участка: сначала ищем пары там, где они стоят рядом.
    leftovers_r: list[Unit] = []
    leftovers_a: list[Unit] = []
    for removed, added in blocks:
        removed = [u for u in removed if u.key not in moved_keys]
        added = [u for u in added if u.key not in moved_keys]
        pairs, rest_r, rest_a = _pair_similar(removed, added)
        changes += [change("modified", r, ad) for r, ad in pairs]
        leftovers_r += rest_r
        leftovers_a += rest_a

    # 3. Изменённая единица, которая к тому же переехала в другой участок.
    pairs, rest_r, rest_a = _pair_similar(leftovers_r, leftovers_a)
    changes += [change("modified", r, ad) for r, ad in pairs]
    changes += [change("deleted", u, None) for u in rest_r]
    changes += [change("added", None, u) for u in rest_a]

    # Порядок как в новом документе (удалённые — по месту в старом).
    order = {id(u): i for i, u in enumerate(new.units)}
    old_order = {id(u): i for i, u in enumerate(old.units)}
    changes.sort(key=lambda c: (order[id(c.new)] if c.new else old_order[id(c.old)] - 0.5))
    return changes


def diff_texts(old_text: str, new_text: str) -> list[Change]:
    return diff_documents(parse_markdown(old_text), parse_markdown(new_text))


# ---------- что отдаём в LLM ----------

def _section_paragraphs(doc: Document, paths: set[tuple[str, ...]]) -> list[int]:
    return sorted({u.paragraph for u in doc.units if u.section in paths})


def to_source_pairs(old_text: str, new_text: str) -> list[SourcePair]:
    """Изменения → пары (БЫЛО, СТАЛО) в формате, который уже понимает llm.analyze.

    Одна пара = один изменённый раздел = один вызов LLM. В пару кладём текст раздела целиком:
    модели нужен контекст («шаг 4» без соседнего нового «шаг 3» непонятен), а перефраз
    «удалили одно предложение, добавили другое» должен попасть в один вызов, а не в два.
    Если раздел длиннее MAX_CONTEXT_CHARS, кладём только изменённые абзацы.
    """
    old_doc, new_doc = parse_markdown(old_text), parse_markdown(new_text)
    groups: dict[tuple[str, ...], list[Change]] = {}
    for c in diff_documents(old_doc, new_doc):
        groups.setdefault(c.section, []).append(c)  # раздел новой версии; для удалённого — старой

    pairs = []
    for path, changes in groups.items():
        # Старая версия раздела могла называться иначе («3. Настройка интеграций» → «4. ...»).
        old_paths = {path} | {c.old.section for c in changes if c.old}
        old_p = _section_paragraphs(old_doc, old_paths)
        new_p = _section_paragraphs(new_doc, {path})
        old_ctx = "\n".join(old_doc.paragraphs[i] for i in old_p)
        new_ctx = "\n".join(new_doc.paragraphs[i] for i in new_p)
        if len(old_ctx) + len(new_ctx) > 2 * MAX_CONTEXT_CHARS:
            old_ctx = "\n".join(old_doc.paragraphs[i] for i in sorted({c.old.paragraph for c in changes if c.old}))
            new_ctx = "\n".join(new_doc.paragraphs[i] for i in sorted({c.new.paragraph for c in changes if c.new}))
        pairs.append(SourcePair(path, old_ctx, new_ctx, changes))
    return pairs
