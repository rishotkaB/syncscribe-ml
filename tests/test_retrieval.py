"""Проверки поиска (только BM25: он не требует модели). Запуск: python tests/test_retrieval.py"""

from pathlib import Path

from syncscribe_ml.diff import parse_markdown, to_source_pairs
from syncscribe_ml.retrieval import BM25, FragmentIndex, deletion_flags, is_deletion, pair_query, tokenize

ROOT = Path(__file__).resolve().parent.parent


def test_tokenize_stems_words_not_numbers():
    assert tokenize("Ядрами, ядра и 150 ГБ") == ["ядрам", "ядра", "и", "150", "гб"]


def test_bm25_prefers_overlap():
    scores = BM25(["Confluence опрашивается раз в 15 минут.", "Jira опрашивается раз в час."]).scores("Confluence 15 минут")
    assert scores[0] > scores[1]


def test_guide_candidates_and_deletion():
    d = ROOT / "data/diff"
    pairs = to_source_pairs((d / "admin_guide_v1.md").read_text(encoding="utf-8"),
                            (d / "admin_guide_v2.md").read_text(encoding="utf-8"))
    derived = parse_markdown((d / "user_guide.md").read_text(encoding="utf-8"))
    fragments = [p for p in derived.paragraphs if not p.startswith("Быстрый старт")]
    index = FragmentIndex(fragments, mode="bm25")
    found = {pair.section[-1]: [fragments[h.index] for h in index.search(pair_query(pair), k=3)] for pair in pairs}

    assert any("8 ядрами" in f for f in found["1. Требования"])
    assert any("скачайте дистрибутив" in f for f in found["2. Установка"])
    assert any("разделе 3" in f for f in found["4. Настройка интеграций"])
    assert any("15 минут" in f for f in found["4. Настройка интеграций"])

    export = next(p for p in pairs if p.section[-1] == "4. Экспорт")
    assert is_deletion(export)
    flagged = [fragments[h.index] for h in deletion_flags(index.search(pair_query(export), k=3))]
    assert flagged == ["Готовый документ можно выгрузить в Word."]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
