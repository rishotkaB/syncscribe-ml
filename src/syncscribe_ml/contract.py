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
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .llm import analyze_truth, classify_truth, confirm_conflict, load_prompt, propose_addition
from .metrics import normalize, quote_in_source
from .retrieval import FragmentIndex, tokenize
from .schema import TruthResultV3
from .truth import NEEDS_ATTENTION, Chunk, bad_quotes, by_file, combine, search_text

logger = logging.getLogger("syncscribe_ml.contract")

CONTRACT_VERSION = "2.0"
# В Docker-образе пакет ставится в site-packages, и путь от __file__ до корня репозитория теряется:
# там папку промптов задают переменной SYNCSCRIBE_ML_PROMPTS_DIR.
PROMPTS_DIR = Path(os.environ.get("SYNCSCRIBE_ML_PROMPTS_DIR") or Path(__file__).resolve().parents[2] / "prompts")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[A-ZА-ЯЁ0-9])")

# confidence_score считаем в коде по проверяемым признакам. Оценку модели используем, только если
# она сама сомневается (ниже 0.7): прогоны 03.11 показали, что и qwen3:8b, и gpt-4o-mini ставят
# всем правкам одно и то же (0.95 / 0.85), то есть их оценка ничего не различает.
SCORE_VALUE_TWO_SOURCES = 0.97  # меняется только значение, его подтверждают два источника
SCORE_VALUE = 0.95              # меняется только значение (число, срок, версия), оно есть в цитате
SCORE_REWRITE = 0.75            # модель переписала формулировку, а не одно значение
SCORE_CONFLICT_REJECTED = 0.6   # источники расходились, проверка решила, что спора нет
SCORE_VALUE_NOT_IN_QUOTE = 0.6  # нового значения нет в цитате источника
SCORE_BAD_QUOTE = 0.4           # цитаты нет в источнике дословно: правка может быть выдумана
SCORE_ADD = 0.8                 # добавление: текст придуман моделью, место выбрано ею же
SCORE_ADD_MANY_FACTS = 0.6      # добавление склеивает несколько предложений
MODEL_DOUBT = 0.7               # ниже этого оценку модели учитываем
MAX_CHANGED_TOKENS = 4          # «только значение»: изменилось не больше 4 слов/чисел
COVERED_SHARE = 0.8             # факт считаем уже описанным, если 80% его слов есть в ближайших блоках


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
    used_quotes: list[str] = field(default_factory=list)  # цитаты источника, на которые опирается вердикт
    calls: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    bad_quotes: int = 0
    error: str = ""
    conflict_rejected: bool = False


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
    out.conflict_rejected = bool(out.calls["conflict_rejected"])
    # Какие факты источника «потрачены» на этот блок: по ним не ищем добавлений.
    if verdict in NEEDS_ATTENTION or verdict == "consistent":
        out.used_quotes = [q.text for r in per_file.values() for q in r.quotes]
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


def changed_tokens(old: str, new: str) -> tuple[list[str], list[str]]:
    """Какие слова/числа ушли и какие появились (со стеммингом, как в поиске)."""
    a, b = Counter(tokenize(old)), Counter(tokenize(new))
    return list((a - b).elements()), list((b - a).elements())


def modify_score(old: str, new: str, r: BlockResult, model_score: float) -> tuple[float, str]:
    """→ (оценка, причина сомнения или ""). Признаки проверяемые, поэтому оценка объяснима."""
    removed, added = changed_tokens(old, new)
    quote_tokens = set(tokenize(" ".join(text for _, text in r.quotes)))
    files = {file for file, _ in r.quotes}
    options = []
    if len(removed) + len(added) <= 2 * MAX_CHANGED_TOKENS and len(added) <= MAX_CHANGED_TOKENS:
        options.append((SCORE_VALUE_TWO_SOURCES if len(files) > 1 else SCORE_VALUE, ""))
    else:
        options.append((SCORE_REWRITE, "Правка переписывает формулировку, а не одно значение."))
    value_only = options[0][0] != SCORE_REWRITE
    # Новые значения должны быть в цитате. Если правка — замена значения, проверяем все новые слова;
    # если переписана формулировка, только числа (слова формулировки модель берёт не из источника).
    must_quote = [t for t in added if value_only or t.isdigit()]
    if must_quote and not set(must_quote) <= quote_tokens:
        options.append((SCORE_VALUE_NOT_IN_QUOTE, "Нового значения нет в цитате источника дословно."))
    if r.conflict_rejected:
        options.append((SCORE_CONFLICT_REJECTED, "Источники по этому месту расходились, проверка сочла это разными объектами."))
    if r.bad_quotes:
        options.append((SCORE_BAD_QUOTE, "Цитата не найдена в источнике дословно."))
    if model_score < MODEL_DOUBT:
        options.append((model_score, "Модель сама сомневается в правке."))
    score, reason = min(options, key=lambda o: o[0])
    return round(score, 2), reason


def modify_suggestion(block: Block, r: BlockResult) -> dict | None:
    res = r.result
    if res is None or not res.new_text.strip() or normalize(res.new_text) == normalize(block.text):
        return None
    old_text, new_text = narrow(block.text, res.new_text.strip())
    rationale = (res.rationale or res.analysis).strip()
    quotes = " ".join(f"«{text}» ({file})" for file, text in r.quotes)
    if quotes:
        rationale += f" Источник: {quotes}"
    score, reason = modify_score(old_text, new_text, r, float(res.confidence_score))
    if reason and score < MODEL_DOUBT and reason not in rationale:
        rationale += f" Проверьте внимательно: {reason[0].lower()}{reason[1:]}"
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

def covered(fact: str, blocks: list[Block]) -> bool:
    """Почти все слова и числа факта уже есть в ближайших блоках документа → описан, модель не зовём."""
    words = [t for t in tokenize(fact) if len(t) > 2 or t.isdigit()]
    if not words:
        return True
    for b in blocks:  # по каждому блоку отдельно: иначе «50» из одного абзаца и «лицензия» из другого дают ложное «описан»
        present = set(tokenize(b.text))
        numbers_ok = all(t in present for t in words if t.isdigit())
        if numbers_ok and sum(t in present for t in words) / len(words) >= COVERED_SHARE:
            return True
    return False


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
    options = [(SCORE_ADD, "")]
    if len(split_sentences(res.new_text)) > 1:
        options.append((SCORE_ADD_MANY_FACTS, "Добавление объединяет несколько фактов."))
    if res.quote and quote_in_source(res.quote, chunk.text):
        rationale += f" Источник: «{res.quote}» ({chunk.file})"
    else:
        options.append((SCORE_BAD_QUOTE, "Цитата не найдена в источнике дословно."))
        rationale += f" Источник ({chunk.file}): «{chunk.text}»"
    if res.confidence_score < MODEL_DOUBT:
        options.append((float(res.confidence_score), "Модель сама сомневается в добавлении."))
    score, reason = min(options, key=lambda o: o[0])
    score = round(score, 2)
    if reason and score < MODEL_DOUBT:
        rationale += f" Проверьте внимательно: {reason[0].lower()}{reason[1:]}"
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
    flags, used = [], []
    llm_seconds = 0.0
    for block, r in zip(checkable, results):
        meta.update(r.calls)
        meta[r.verdict if not r.error else "errors"] += 1
        meta["bad_quotes"] += r.bad_quotes
        llm_seconds += r.seconds
        used += [normalize(q) for q in r.used_quotes if q.strip()]
        if r.verdict == "contradicts":
            s = modify_suggestion(block, r)
            if s:
                found.append((position[block.block_id], 0, s))
        elif r.verdict == "sources_conflict":
            flags.append(conflict_flag(block, r))

    if adds:
        # Добавления ищем по отдельным фактам (предложениям), а не по кускам: иначе модель склеивает
        # в одну вставку 3–4 факта из абзаца руководства (прогон 03.11).
        facts = [Chunk(c.file, c.section, sentence) for c in chunks for sentence in split_sentences(c.text)]
        rest = [f for f in facts if not any(q in normalize(f.text) or normalize(f.text) in q for q in used)]
        meta["add_facts_unused"] = len(rest)
        block_index = FragmentIndex([b.for_model() for b in checkable], mode=mode)
        doc_text = "\n".join(b.text for b in blocks)
        todo = []
        for fact, hits in zip(rest, block_index.search_many([search_text(f) for f in rest], k=top_k)):
            candidates = [checkable[h.index] for h in hits]
            if not candidates:
                continue
            if covered(fact.text, candidates):  # дешёвая проверка без модели: факт уже описан в документе
                meta["add_skipped_covered"] += 1
                continue
            todo.append((fact, candidates))

        def check_add(item):
            fact, candidates = item
            try:
                return addition(model, prompts, fact, candidates, doc_text)
            except Exception as exc:
                logger.warning("добавление по факту «%s»: %s", fact.text[:40], exc)
                return "error", 0.0

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            added = list(pool.map(check_add, todo))
        seen = set()
        for s, latency in added:
            if s == "error":
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
