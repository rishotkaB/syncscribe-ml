"""Проверки режима «источник истины» (без модели). Запуск: python tests/test_truth.py"""

from syncscribe_ml import syncback
from syncscribe_ml.llm import build_truth_message
from syncscribe_ml.retrieval import FragmentIndex
from syncscribe_ml.schema import Quote, TruthResult
from syncscribe_ml.truth import bad_quotes, by_file, combine, load_chunks, quoted_sources, search_text

SOURCES = {
    "docs/admin.md": "# Руководство\n\n## Сессии\n\nВремя жизни сессии: 8 часов.\n\n## API\n\nЛимит API: 60 запросов в минуту.\n",
    "faq.md": "## Вопросы\n\nВремя жизни сессии: 24 часа.\n",
}


def test_chunks_keep_file_and_section():
    chunks = load_chunks(SOURCES)
    assert [(c.file, c.section) for c in chunks] == [("admin.md", "Сессии"), ("admin.md", "API"), ("faq.md", "Вопросы")]
    assert chunks[0].for_model() == "(admin.md, раздел «Сессии») Время жизни сессии: 8 часов."


def test_search_finds_both_disputed_chunks():
    chunks = load_chunks(SOURCES)
    index = FragmentIndex([search_text(c) for c in chunks], mode="bm25")
    found = {chunks[h.index].file for h in index.search("Сессия живёт 24 часа, время жизни сессии фиксировано.", k=2)}
    assert found == {"admin.md", "faq.md"}


def test_message_numbers_chunks():
    msg = build_truth_message(["(a.md) X.", "(b.md) Y."], "Фрагмент.")
    assert "[1] (a.md) X.\n\n[2] (b.md) Y." in msg and msg.endswith("ФРАГМЕНТ:\nФрагмент.")


def test_bad_quotes_checks_the_cited_chunk():
    chunks = load_chunks(SOURCES)
    r = TruthResult(analysis="", verdict="sources_conflict", new_text="", quotes=[
        Quote(source=1, text="Время жизни сессии: 8 часов."),
        Quote(source=1, text="Время жизни сессии: 24 часа."),  # есть в источнике, но не в куске [1]
        Quote(source=9, text="что угодно"),                    # куска с таким номером нет
    ])
    assert bad_quotes(r, chunks) == ["Время жизни сессии: 24 часа.", "что угодно"]
    assert quoted_sources(r, chunks)[0] == ("admin.md", "Время жизни сессии: 8 часов.")


def test_conflict_flag_has_no_edit():
    flag = syncback.conflict_flag("Сессии", "Сессия живёт 24 часа.", [("admin.md", "8 часов."), ("faq.md", "24 часа.")])
    assert flag["reason"] == "sources_conflict" and "new_text" not in flag
    assert flag["quotes"][1] == {"file": "faq.md", "text": "24 часа."}


def _r(verdict, new_text="", quote=""):
    return TruthResult(analysis="", verdict=verdict, new_text=new_text, quotes=[Quote(source=1, text=quote)] if quote else [])


def test_combine_one_agrees_other_contradicts_is_conflict():
    groups = by_file(load_chunks(SOURCES))
    per_file = {"admin.md": _r("contradicts", "Сессия живёт 8 часов.", "Время жизни сессии: 8 часов."),
                "faq.md": _r("consistent", quote="Время жизни сессии: 24 часа.")}
    verdict, result, quotes = combine(per_file, groups)
    assert verdict == "sources_conflict" and result is None
    assert quotes == [("admin.md", "Время жизни сессии: 8 часов."), ("faq.md", "Время жизни сессии: 24 часа.")]


def test_combine_different_edits_is_conflict_same_edit_is_not():
    groups = by_file(load_chunks(SOURCES))
    a, b = _r("contradicts", "8 часов."), _r("contradicts", "24 часа.")
    assert combine({"admin.md": a, "faq.md": b}, groups)[0] == "sources_conflict"
    verdict, result, _ = combine({"admin.md": a, "faq.md": _r("contradicts", "8 часов.")}, groups)
    assert verdict == "contradicts" and result.new_text == "8 часов."


def test_combine_not_covered_file_does_not_create_conflict():
    groups = by_file(load_chunks(SOURCES))
    verdict, result, _ = combine({"admin.md": _r("contradicts", "X."), "faq.md": _r("not_covered")}, groups)
    assert verdict == "contradicts" and result.new_text == "X."
    assert combine({"admin.md": _r("not_covered"), "faq.md": _r("consistent")}, groups)[0] == "consistent"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
