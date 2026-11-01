"""Проверки точки входа для SyncBack (без модели). Запуск: python tests/test_service.py"""

from syncscribe_ml import service
from syncscribe_ml.schema import Quote, TruthResult
from syncscribe_ml.truth import Judgement, Prompts

DOCX_TEXT = "Установка\nСкачайте пакет и запустите установщик.\nЛимиты\nМаксимальный размер файла — 50 МБ.\nКонтакты: support@example.com."


def test_docx_lines_are_paragraphs_and_headings_are_sections():
    ps = service.split_document(DOCX_TEXT, "docx")
    assert [(p.section, p.text) for p in ps] == [
        ("Установка", "Скачайте пакет и запустите установщик."),
        ("Лимиты", "Максимальный размер файла — 50 МБ."),
        ("Лимиты", "Контакты: support@example.com."),
    ]


def test_markdown_and_blank_line_txt():
    md = service.split_document("# Лимиты\n\nФайл до 50 МБ.\n\nДокумент до 300 страниц.\n", "md")
    assert [(p.section, p.text) for p in md] == [("Лимиты", "Файл до 50 МБ."), ("Лимиты", "Документ до 300 страниц.")]
    txt = service.split_document("Первый абзац\nв две строки.\n\nВторой абзац.", "txt")
    assert [p.text for p in txt] == ["Первый абзац\nв две строки.", "Второй абзац."]


def test_release_notes_list_split_into_items():
    chunks = service.source_chunks("Релиз 2.4\n\n- Лимит файла: 100 МБ.\n- Поддержка PDF.\n")
    assert [c.text for c in chunks] == ["- Лимит файла: 100 МБ.", "- Поддержка PDF."]
    assert chunks[0].section == "Релиз 2.4"


def test_suggest_returns_syncback_items():
    def fake_judge(model, prompts, fragment, groups, fast):
        if "50 МБ" in fragment:
            r = TruthResult(analysis="Лимит вырос.", verdict="contradicts",
                            new_text="Максимальный размер файла — 100 МБ.", quotes=[Quote(source=1, text="Лимит файла: 100 МБ.")])
            return Judgement("contradicts", r, [("источник", "Лимит файла: 100 МБ.")])
        return Judgement("consistent", None, [])

    original = service.judge
    service.judge = fake_judge
    try:
        out = service.suggest(DOCX_TEXT, "- Лимит файла: 100 МБ.\n- Поддержка PDF.", "docx", "m",
                              mode="bm25", prompts=Prompts("", "", ""))
    finally:
        service.judge = original
    [s] = out["suggestions"]
    assert s["section_ref"] == "Лимиты" and s["change_type"] == "modify"
    assert s["old_text"] in DOCX_TEXT and "\n" not in s["old_text"]
    assert s["rationale"] == "Лимит вырос. Источник: «Лимит файла: 100 МБ.»"
    assert out["stats"]["contradicts"] == 1 and out["flags"] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
