"""Собрать JSON-запрос контракта 2.0 из наших markdown-примеров (документ → блоки).

    python scripts/md_to_request.py data/large/truth/derived.md data/large/truth/admin_guide.md data/large/truth/faq.md \
        -o data/contract/large_request.json --labels data/large/truth/labels.json --labels-out data/contract/large_labels.json

Заголовки становятся блоками type=heading, editable=false; абзацы — type=paragraph, editable=true.
Каждый файл-источник — элемент "sources" (у SyncBack пока один "source", мы понимаем оба варианта).
"""

import argparse
import json
import uuid
from pathlib import Path

from syncscribe_ml.diff import parse_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="Наша разметка [{paragraph, expected}]")
    parser.add_argument("--labels-out", type=Path, default=None, help="Разметка по block_id для run_contract.py")
    args = parser.parse_args()

    doc = parse_markdown(args.document.read_text(encoding="utf-8"))
    headings = {u.paragraph for u in doc.units if u.kind == "heading"}
    blocks = []
    for i, text in enumerate(doc.paragraphs):
        heading = i in headings
        blocks.append({"block_id": f"d-{i + 1}", "type": "heading" if heading else "paragraph",
                       "text": text, "editable": not heading})
    request = {
        "request_id": str(uuid.uuid4()),
        "analysis_job_id": str(uuid.uuid4()),
        "document": {"id": str(uuid.uuid4()), "name": args.document.name, "format": "md", "blocks": blocks},
        "sources": [{"id": str(uuid.uuid4()), "name": s.name, "type": "file", "text": s.read_text(encoding="utf-8")}
                    for s in args.sources],
    }
    args.output.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Запрос: {args.output} (блоков {len(blocks)}, источников {len(args.sources)})")

    if args.labels and args.labels_out:
        expected = {l["paragraph"]: l["expected"] for l in json.loads(args.labels.read_text(encoding="utf-8"))}
        labels = {b["block_id"]: expected[b["text"]] for b in blocks
                  if b["editable"] and expected.get(b["text"], "ok") != "ok"}
        args.labels_out.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Разметка: {args.labels_out} ({len(labels)} блоков)")


if __name__ == "__main__":
    main()
