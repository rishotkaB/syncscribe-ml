"""Загрузка документа в текст (markdown).

.md и .txt читаем как есть. .docx и .pdf превращаем в markdown через Docling:
он сохраняет заголовки, списки и таблицы, а diff.parse_markdown опирается именно на них.
Docling тяжёлый (тянет torch), поэтому он не в основных зависимостях:
    pip install -e .[docling]
"""

from pathlib import Path


def load_document(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise SystemExit(f"Для {path.suffix} нужен Docling: pip install -e .[docling]") from exc
    return DocumentConverter().convert(str(path)).document.export_to_markdown()
