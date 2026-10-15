"""Сквозной прогон этапов 2+3: две версии источника + производный документ → какие абзацы править.

Пример:
    python scripts/check_docs.py data/diff/admin_guide_v1.md data/diff/admin_guide_v2.md data/diff/user_guide.md

Пока перебираем все пары (изменённый раздел × абзац производного документа): для примера это
5 × 7 = 35 вызовов. На этапе 4 вместо перебора будет поиск (BGE-M3 + Qdrant): для каждого изменения
берём только несколько похожих фрагментов.
"""

import argparse
import sys
from pathlib import Path

from syncscribe_ml.diff import parse_markdown, to_source_pairs
from syncscribe_ml.llm import analyze, load_prompt
from syncscribe_ml.parse import load_document

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("derived", help="Производный документ, который проверяем")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--prompt", default=ROOT / "prompts/impact_v2.md", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pairs = to_source_pairs(load_document(args.old), load_document(args.new))
    derived = parse_markdown(load_document(args.derived))
    # Фрагмент = абзац производного документа; заголовки не проверяем.
    heading_paragraphs = {u.paragraph for u in derived.units if u.kind == "heading"}
    fragments = [(i, p) for i, p in enumerate(derived.paragraphs) if i not in heading_paragraphs]
    system_prompt = load_prompt(args.prompt)
    print(f"Изменённых разделов: {len(pairs)}, фрагментов: {len(fragments)}, вызовов: {len(pairs) * len(fragments)}\n")

    hits: dict[int, list] = {}
    for pair in pairs:
        where = pair.section[-1] if pair.section else "(без раздела)"
        for i, fragment in fragments:
            try:
                result, latency, _ = analyze(args.model, system_prompt, pair.source_old, pair.source_new, fragment)
            except Exception as exc:  # один упавший вызов не должен останавливать прогон
                print(f"[{where[:30]:<30}] абзац {i:>2}: ОШИБКА {repr(exc).splitlines()[0]}")
                continue
            mark = "ЗАТРОНУТ" if result.impacted else "-"
            print(f"[{where[:30]:<30}] абзац {i:>2}: {mark} ({latency:.1f}s)")
            if result.impacted:
                hits.setdefault(i, []).append((where, result))

    print("\n=== Итог: что править в", args.derived)
    for i, fragment in fragments:
        if i not in hits:
            continue
        print(f"\nАбзац {i}: {fragment}")
        for where, r in hits[i]:
            print(f"  из-за [{where}]\n  → {r.new_text}\n  цитата: {r.rationale_quote}")
    print(f"\nЗатронуто абзацев: {len(hits)} из {len(fragments)}")


if __name__ == "__main__":
    main()
