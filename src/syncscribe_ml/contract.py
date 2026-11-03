"""Контракт 2.0 с SyncBack: обмен JSON-файлами.

Вход (от SyncBack, один файл на документ):
    {"request_id", "analysis_job_id",
     "document": {"id", "name", "format", "blocks": [{"block_id", "type", "text", "editable"}, ...]},
     "source": {"id", "name", "type", "text"}}          # или "sources": [ {...}, {...} ]

Выход (от нас, один файл):
    {"contract_version": "2.0", "request_id", "analysis_job_id", "status": "completed" | "failed",
     "suggestions": [{"suggestion_id", "change_type", "target_block_id", "old_text", "new_text",
                      "insert_after", "rationale", "confidence_score"}],
     "flags": [...],   # «проверьте вручную»: источники спорят. Правку не предлагаем (решение Alex 29.10)
     "meta": {...},    # статистика прогона, SyncBack может игнорировать
     "error": null | {"code", "message"}}

Как работает (режим «источник истины», тот же, что scripts/check_truth.py):
1. modify. Для каждого редактируемого блока ищем 3 ближайших куска источника (BGE-M3 + BM25),
   короткий вызов «только вердикт», полный вызов с правкой, цитатой и confidence_score.
   Если источников несколько и они спорят — пометка в flags вместо правки.
2. add. Куски источника, на которые не опирается ни одна правка, проверяем отдельно: нет ли в источнике
   факта, которого в документе нет совсем (например, «добавлена оплата по счёту»). Если есть — правка add
   после ближайшего по смыслу блока.
3. delete пока не предлагаем: удаление без замены в наших данных не встречалось, сначала нужен пример.

Требование SyncBack: old_text и insert_after дословно лежат в тексте блока. Мы их не генерируем моделью,
а берём из блока сами (целиком или одно изменённое предложение), поэтому это условие выполнено всегда.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .llm import analyze_truth, classify_truth, confirm_conflict, load_prompt, propose_addition
from .metrics import normalize, quote_in_source
from .retrieval import FragmentIndex
from .schema import TruthResultV3
from .truth import NEEDS_ATTENTION, Chunk, bad_quotes, by_file, combine, search_text

logger = logging.getLogger("syncscribe_ml.contract")

CONTRACT_VERSION = "2.0"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[A-ZА-ЯЁ0-9])")

# Ограничения на confidence_score поверх оценки модели (модель склонна ставить 0.9+ всему подряд).
CAP_BAD_QUOTE = 0.4   # цитаты нет в источнике дословно: правка может быть выдумана
CAP_ADD_SAME_TEXT = 0.0


class ContractError(ValueError):
    """Запрос не соответствует контракту: отвечаем status=failed."""


@dataclass
class Block:
    block_id: str
    type: str
    text: str
    editable: bool
    section: str  # ближайший заголовок выше блока

    @property
    def checkable(self) -> bool:
        return self.editable and self.type != "heading" and bool(self.text.strip())

    def for_model(self) -> str:
        return f"(раздел «{self.section}») {self.text}" if self.section else self.text


@dataclass
class Prompts:
    full: str
    quick: str
    conflict: str
    add: str

    @classmethod
    def default(cls) -> "Prompts":
        return cls(*(load_prompt(PROMPTS_DIR / name) for name in
                     ("truth_v3.md", "truth_quick_v2.md", "conflict_v1.md", "add_v1.md")))


@dataclass
class BlockResult:
    verdict: str = "not_covered"
    result: TruthResultV3 | None = None
    quotes: list[tuple[str, str]] = field(default_factory=list)
    used_chunks: set[int] = field(default_factory=set)  # куски источника, на которые опирается вердикт
    calls: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    bad_quotes: int = 0
    error: str = ""


# ---------- разбор входа ----------

def parse_request(request: dict) -> tuple[list[Block], list[tuple[str, str]]]:
    """→ (блоки документа, [(имя источника, текст)])."""
    try:
        raw_blocks = request["document"]["blocks"]
    except (KeyError, TypeError) as exc:
        raise ContractError("нет document.blocks") from exc
    if not isinstance(raw_blocks, list):
        raise ContractError("document.blocks должен быть списком")

    blocks, section, seen = [], "", set()
    for i, b in enumerate(raw_blocks):
        if not isinstance(b, dict) or "block_id" not in b or "text" not in b:
            raise ContractError(f"блок №{i + 1}: нужны поля block_id и text")
        if b["block_id"] in seen:
            raise ContractError(f"блок {b['block_id']}: block_id повторяется")
        seen.add(b["block_id"])
        kind = b.get("type", "paragraph")
        if kind == "heading":
            section = b["text"].strip()
        blocks.append(Block(b["block_id"], kind, b["text"], bool(b.get("editable", True)), section))

    raw_sources = request.get("sources") or ([request["source"]] if request.get("source") else [])
    sources = []
    for i, s in enumerate(raw_sources):
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            raise ContractError(f"источник №{i + 1}: пустой text")
        sources.append((s.get("name") or s.get("id") or f"источник {i + 1}", s["text"]))
    if not sources:
        raise ContractError("нет source (или sources)")
    return blocks, sources


def source_chunks(name: str, text: str) -> list[Chunk]:
    """Источник — просто текст: протокол встречи, release notes, страница руководства.

    Режем на абзацы по пустым строкам; markdown-заголовки задают раздел; список режем на пункты
    (один пункт — одно изменение: так точнее и поиск, и цитата в обосновании).
    """
    chunks, section = [], ""
    blocks = re.split(r"\n\s*\n", text) if re.search(r"\n\s*\n", text) else text.split("\n")
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        heading = _MD_HEADING.match(lines[0])
        if heading:
            section = heading.group(1).strip()
            lines = lines[1:]
            if not lines:
                continue
        if len(lines) > 1 and any(_LIST_ITEM.match(l) for l in lines):
            # «Решения:» + пункты списка: подпись над списком считаем разделом пунктов
            intro = [l for l in lines if not _LIST_ITEM.match(l)]
            items = [l for l in lines if _LIST_ITEM.match(l)]
            sec = " / ".join(x for x in (section, " ".join(intro).rstrip(":")) if x)
            chunks += [Chunk(name, sec, item) for item in items]
        else:
            chunks.append(Chunk(name, section, " ".join(lines)))
    return chunks


# ---------- шаг 1: modify ----------

def judge_block(model: str, prompts: Prompts, block: Block, groups: dict[str, list[Chunk]],
                index_of: dict[int, int], fast: bool) -> BlockResult:
    """Вердикт по одному блоку. Логика та же, что в scripts/check_truth.py."""
    out = BlockResult()
    if not groups:
        return out
    fragment = block.text
    files = list(groups)
    if fast:
        quick = {}
        for f in groups:
            quick[f], latency = classify_truth(model, prompts.quick, [c.for_model() for c in groups[f]], fragment)
            out.calls["quick"] += 1
            out.seconds += latency
        flagged = [f for f, v in quick.items() if v in NEEDS_ATTENTION]
        if not flagged:
            out.verdict = "consistent" if "consistent" in quick.values() else "not_covered"
            return out
        files = flagged + [f for f, v in quick.items() if v == "consistent" and len(groups) > 1]

    per_file = {}
    for f in files:
        per_file[f], latency = analyze_truth(model, prompts.full, [c.for_model() for c in groups[f]],
                                             fragment, schema=TruthResultV3)
        out.calls["full"] += 1
        out.seconds += latency
    verdict, result, quotes = combine(per_file, groups)
    if verdict == "sources_conflict" and len({f for f, _ in quotes}) > 1:
        out.calls["conflict"] += 1
        try:
            confirmed, latency = confirm_conflict(model, prompts.conflict, fragment, quotes)
            out.seconds += latency
        except Exception:  # не смогли проверить — пусть человек посмотрит
            confirmed = True
        if not confirmed:
            out.calls["conflict_rejected"] += 1
            verdict, result, quotes = combine(per_file, groups, allow_conflict=False)

    out.verdict, out.result, out.quotes = verdict, result, quotes
    out.bad_quotes = sum(len(bad_quotes(r, groups[f])) for f, r in per_file.items() if r.verdict == "contradicts")
    # Какие куски источника «потрачены» на этот блок: на них не ищем добавлений.
    if verdict in NEEDS_ATTENTION or verdict == "consistent":
        for f, r in per_file.items():
            for q in r.quotes:
                if 1 <= q.source <= len(groups[f]):
                    out.used_chunks.add(index_of[id(groups[f][q.source - 1])])
    return out


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def narrow(old: str, new: str) -> tuple[str, str]:
    """Если модель переписала абзац, а поменялось одно предложение, отдаём только его (так просит SyncBack).

    old_text всегда берём из текста блока, поэтому он дословно есть в документе.
    """
    a, b = split_sentences(old), split_sentences(new)
    if len(a) > 1 and len(a) == len(b):
        changed = [i for i in range(len(a)) if normalize(a[i]) != normalize(b[i])]
        if len(changed) == 1 and old.count(a[changed[0]]) == 1:
            return a[changed[0]], b[changed[0]]
    return old, new


def modify_suggestion(block: Block, r: BlockResult) -> dict | None:
    res = r.result
    if res is None or not res.new_text.strip() or normalize(res.new_text) == normalize(block.text):
        return None
    old_text, new_text = narrow(block.text, res.new_text.strip())
    rationale = (res.rationale or res.analysis).strip()
    quotes = " ".join(f"«{text}» ({file})" for file, text in r.quotes)
    if quotes:
        rationale += f" Источник: {quotes}"
    score = round(float(res.confidence_score), 2)
    if r.bad_quotes:
        score = min(score, CAP_BAD_QUOTE)
        rationale += " Цитата не найдена в источнике дословно, проверьте правку внимательно."
    return {
        "change_type": "modify",
        "target_block_id": block.block_id,
        "old_text": old_text,
        "new_text": new_text,
        "insert_after": None,
        "rationale": rationale,
        "confidence_score": score,
    }


def conflict_flag(block: Block, r: BlockResult) -> dict:
    return {
        "reason": "sources_conflict",
        "target_block_id": block.block_id,
        "old_text": block.text,
        "rationale": "Источники противоречат друг другу, правку не предлагаем, проверьте вручную.",
        "quotes": [{"source": file, "text": text} for file, text in r.quotes],
    }


# ---------- шаг 2: add ----------

def last_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[-1] if sentences else text


def addition(model: str, prompts: Prompts, chunk: Chunk, candidates: list[Block], doc_text: str) -> tuple[dict | None, float]:
    res, latency = propose_addition(model, prompts.add, chunk.for_model(), [b.for_model() for b in candidates])
    if not res.needed or not res.new_text.strip() or not 1 <= res.after <= len(candidates):
        return None, latency
    if normalize(res.new_text) in normalize(doc_text):  # уже есть в документе дословно
        return None, latency
    block = candidates[res.after - 1]
    rationale = res.rationale.strip() or res.analysis.strip()
    score = round(float(res.confidence_score), 2)
    if res.quote and quote_in_source(res.quote, chunk.text):
        rationale += f" Источник: «{res.quote}» ({chunk.file})"
    else:
        score = min(score, CAP_BAD_QUOTE)
        rationale += f" Источник ({chunk.file}): «{chunk.text}»"
    return {
        "change_type": "add",
        "target_block_id": block.block_id,
        "old_text": None,
        "new_text": res.new_text.strip(),
        "insert_after": last_sentence(block.text),
        "rationale": rationale,
        "confidence_score": score,
    }, latency


# ---------- весь запрос ----------

def run(request: dict, model: str, prompts: Prompts | None = None, *, fast: bool = True, top_k: int = 3,
        mode: str = "hybrid", adds: bool = True, workers: int = 1) -> dict:
    """Запрос контракта 2.0 → ответ контракта 2.0. Не бросает исключений: ошибка → status=failed."""
    started = time.perf_counter()
    response = {
        "contract_version": CONTRACT_VERSION,
        "request_id": (request or {}).get("request_id"),
        "analysis_job_id": (request or {}).get("analysis_job_id"),
        "status": "completed",
        "suggestions": [],
        "flags": [],
        "meta": {},
        "error": None,
    }
    try:
        prompts = prompts or Prompts.default()
        blocks, sources = parse_request(request)
        suggestions, flags, meta = _analyze(blocks, sources, model, prompts, fast, top_k, mode, adds, workers)
    except ContractError as exc:
        response.update(status="failed", error={"code": "invalid_request", "message": str(exc)})
        return response
    except Exception as exc:  # модель недоступна, ответ не по схеме и т. п.
        logger.exception("контракт 2.0: прогон упал")
        response.update(status="failed", error={"code": "pipeline_error", "message": f"{type(exc).__name__}: {exc}"})
        return response

    for i, s in enumerate(suggestions, 1):
        response["suggestions"].append({"suggestion_id": f"s-{i}", **s})
    response["flags"] = [{"flag_id": f"f-{i}", **f} for i, f in enumerate(flags, 1)]
    meta["seconds_total"] = round(time.perf_counter() - started, 1)
    response["meta"] = meta
    return response


def _analyze(blocks, sources, model, prompts, fast, top_k, mode, adds, workers):
    chunks = [c for name, text in sources for c in source_chunks(name, text)]
    checkable = [b for b in blocks if b.checkable]
    meta: Counter = Counter(blocks=len(blocks), checked_blocks=len(checkable), source_chunks=len(chunks))
    if not chunks or not checkable:
        return [], [], dict(meta)

    index = FragmentIndex([search_text(c) for c in chunks], mode=mode)
    index_of = {id(c): i for i, c in enumerate(chunks)}
    hits_of = index.search_many([b.text for b in checkable], k=top_k)  # эмбеддинги одним пакетом

    def check(pair):
        block, hits = pair
        try:
            return judge_block(model, prompts, block, by_file([chunks[h.index] for h in hits]), index_of, fast)
        except Exception as exc:  # один упавший блок не роняет весь документ
            logger.warning("блок %s: %s", block.block_id, exc)
            return BlockResult(error=f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(check, zip(checkable, hits_of)))

    position = {b.block_id: i for i, b in enumerate(blocks)}
    found: list[tuple[int, int, dict]] = []  # (позиция блока, порядок внутри, правка) — для сортировки по документу
    flags, used = [], set()
    llm_seconds = 0.0
    for block, r in zip(checkable, results):
        meta.update(r.calls)
        meta[r.verdict if not r.error else "errors"] += 1
        meta["bad_quotes"] += r.bad_quotes
        llm_seconds += r.seconds
        used |= r.used_chunks
        if r.verdict == "contradicts":
            s = modify_suggestion(block, r)
            if s:
                found.append((position[block.block_id], 0, s))
        elif r.verdict == "sources_conflict":
            flags.append(conflict_flag(block, r))

    if adds:
        rest = [c for i, c in enumerate(chunks) if i not in used]
        block_index = FragmentIndex([b.for_model() for b in checkable], mode=mode)
        doc_text = "\n".join(b.text for b in blocks)
        seen = set()
        for chunk, hits in zip(rest, block_index.search_many([search_text(c) for c in rest], k=top_k)):
            candidates = [checkable[h.index] for h in hits]
            if not candidates:
                continue
            try:
                s, latency = addition(model, prompts, chunk, candidates, doc_text)
            except Exception as exc:
                logger.warning("добавление по куску «%s»: %s", chunk.text[:40], exc)
                meta["errors"] += 1
                continue
            meta["add_checks"] += 1
            llm_seconds += latency
            if s and (s["target_block_id"], normalize(s["new_text"])) not in seen:
                seen.add((s["target_block_id"], normalize(s["new_text"])))
                found.append((position[s["target_block_id"]], 1, s))
                meta["add"] += 1

    found.sort(key=lambda t: (t[0], t[1]))
    meta = dict(meta)
    meta["llm_seconds"] = round(llm_seconds, 1)
    return [s for _, _, s in found], flags, meta
