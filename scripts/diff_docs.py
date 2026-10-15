"""Этап 3: показать, что изменилось между двумя версиями источника.

Пример:
    python scripts/diff_docs.py data/diff/admin_guide_v1.md data/diff/admin_guide_v2.md
    python scripts/diff_docs.py old.docx new.docx --pairs   # что уйдёт в LLM
"""

import argparse
import sys

from syncscribe_ml.diff import diff_texts, to_source_pairs
from syncscribe_ml.parse import load_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--pairs", action="store_true", help="Показать пары (БЫЛО, СТАЛО) по разделам, как их увидит LLM")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    old_text, new_text = load_document(args.old), load_document(args.new)
    if args.pairs:
        for i, pair in enumerate(to_source_pairs(old_text, new_text), 1):
            print(f"=== Пара {i}: {' / '.join(pair.section) or '(без раздела)'} · изменений: {len(pair.changes)}")
            print(f"--- БЫЛО:\n{pair.source_old or '(нет)'}\n--- СТАЛО:\n{pair.source_new or '(нет)'}\n")
        return

    changes = diff_texts(old_text, new_text)
    for c in changes:
        where = " / ".join(c.section) or "(без раздела)"
        text = c.inline or (c.new or c.old).text
        print(f"{c.kind:<9} [{where}]\n          {text}")
    print(f"\nВсего изменений: {len(changes)}")


if __name__ == "__main__":
    main()
