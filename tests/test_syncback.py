"""Проверки перевода в формат SyncBack. Запуск: python tests/test_syncback.py"""

from syncscribe_ml import syncback
from syncscribe_ml.diff import parse_markdown
from syncscribe_ml.schema import ImpactResult


def _result(new_text: str) -> ImpactResult:
    return ImpactResult(analysis="Лимит вырос.", impacted=True, confidence=0.9, change_type="factual",
                        new_text=new_text, rationale_quote="Лимит — 100 МБ.")


def test_suggestion_fields():
    s = syncback.suggestion("Загрузка", "Лимит — 50 МБ.", _result("Лимит — 100 МБ."))
    assert s["change_type"] == "modify" and s["old_text"] == "Лимит — 50 МБ." and s["new_text"] == "Лимит — 100 МБ."
    assert "«Лимит — 100 МБ.»" in s["rationale"]
    assert s["confidence_score"] is None  # пока нет судьи, неоткалиброванное число не отдаём


def test_dedupe_same_edit():
    a = syncback.suggestion("Р", "X.", _result("Y."))
    b = dict(a, rationale="Другое изменение.")
    c = syncback.suggestion("Р", "X.", _result("Z."))
    out = syncback.response([a, b, c], [])["suggestions"]
    assert len(out) == 2 and "Другое изменение." in out[0]["rationale"]


def test_raw_paragraph_is_verbatim():
    # Абзац из двух строк: в old_text он должен попасть так, как записан в файле, иначе SyncBack его не найдёт.
    text = "# Раздел\n\nПервая строка абзаца\nвторая строка абзаца.\n"
    doc = parse_markdown(text)
    assert doc.raw[1] in text and doc.paragraphs[1] not in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
