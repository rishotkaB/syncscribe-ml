"""Сквозная проверка адаптера на коде SyncBack: наш MLPipelineClient → их маппер правок.

Ничего в SyncBack не меняет: берёт их код из указанной папки только на чтение.

    python integrations/syncback/check_with_syncback.py --syncback ../SyncBack-main \
        --document data/large/truth/derived.md --source data/syncback/release_notes.md \
        --endpoint http://localhost:11434/v1 --model qwen3:8b

1. Документ .md превращаем в настоящий .docx (заголовки — стилем Heading) и разбираем ИХ DocxParser:
   так мы получаем ровно тот plain_text, который видит их воркер.
2. Зовём наш адаптер generate_suggestions(document_text, source_text, "docx").
3. Отдаём результат ИХ map_to_suggestions и считаем, сколько правок он принял и сколько отбросил.
Для SyncBack нужны его зависимости (pip install -e <SyncBack>) или хотя бы sqlalchemy, pydantic-settings, python-docx.
"""

import argparse
import asyncio
import io
import json
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def md_to_docx(markdown: str) -> bytes:
    from docx import Document

    doc = Document()
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            level = len(block) - len(block.lstrip("#"))
            doc.add_heading(block.lstrip("# ").strip(), level=min(level, 9))
        else:
            doc.add_paragraph(" ".join(block.split("\n")))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syncback", required=True, type=Path, help="Папка с кодом SyncBack (там, где app/)")
    parser.add_argument("--document", required=True, type=Path, help="Документ .md: станет .docx")
    parser.add_argument("--source", required=True, type=Path, help="Источник: текст изменений (release notes и т. п.)")
    parser.add_argument("--endpoint", required=True, help="OpenAI-совместимый API модели, например http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--api-key", default="", help="Если API требует ключ")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sys.path.insert(0, str(args.syncback.resolve()))
    sys.path.insert(0, str(HERE))
    from app.infrastructure.parsers.docx_parser import DocxParser
    from app.workers.pipeline.suggestion_mapper import map_to_suggestions
    from ml_pipeline_client import MLPipelineClient

    parsed = DocxParser().parse(md_to_docx(args.document.read_text(encoding="utf-8")))
    source_text = args.source.read_text(encoding="utf-8")
    settings = SimpleNamespace(llm_endpoint=args.endpoint, llm_model=args.model, llm_api_key=args.api_key,
                               llm_auth="bearer" if args.api_key else "none")
    client = MLPipelineClient(settings)
    batch = asyncio.run(client.generate_suggestions(parsed.plain_text, source_text, "docx"))
    raw = json.loads(batch.raw_response)

    kept = map_to_suggestions(batch, uuid.uuid4(), uuid.uuid4(), document_text=parsed.plain_text,
                              document_format="docx", diagnostics_enabled=True)
    print(f"Абзацев: {raw['stats'].get('paragraphs')}, кусков источника: {raw['stats'].get('source_chunks')}")
    print("Вызовы:", {k: v for k, v in raw["stats"].items() if k in ("quick", "full", "conflict")},
          f"время в LLM {raw['stats'].get('llm_seconds', 0):.0f}s")
    print(f"Наш ответ: правок {len(batch.items)}, пометок {len(raw['flags'])}")
    print(f"Маппер SyncBack принял {len(kept)} из {len(batch.items)}")
    for s in kept[:5]:
        print(f"\n[{s.section_ref}] {s.original_text}\n  → {s.suggested_text}\n  {s.rationale}")


if __name__ == "__main__":
    main()
