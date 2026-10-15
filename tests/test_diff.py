"""Проверки diff. Запуск: python tests/test_diff.py (или pytest, если установлен)."""

import json
from pathlib import Path

from syncscribe_ml.diff import diff_texts, split_sentences, to_source_pairs

ROOT = Path(__file__).resolve().parent.parent


def _guide():
    d = ROOT / "data/diff"
    return (d / "admin_guide_v1.md").read_text(encoding="utf-8"), (d / "admin_guide_v2.md").read_text(encoding="utf-8")


def test_sentences_keep_numbered_steps():
    assert split_sentences("Обновление: 1. Остановите сервис. 2. Сделайте копию.") == [
        "Обновление: 1. Остановите сервис.",
        "2. Сделайте копию.",
    ]
    assert split_sentences("Раздел 6.1. Двухфакторная аутентификация.") == ["Раздел 6.1. Двухфакторная аутентификация."]


def test_no_changes():
    old, _ = _guide()
    assert diff_texts(old, old) == []
    # Регистр, ё и лишние пробелы изменением не считаются.
    assert diff_texts("Сервер  ещё работает.", "сервер еще работает.") == []


def test_guide_changes():
    changes = diff_texts(*_guide())
    inline = [c.inline for c in changes if c.kind == "modified"]
    assert any("[-8-] {+16+}" in s and "[-32-] {+64+}" in s for s in inline)  # требования
    assert any("[-14-] {+16+}" in s for s in inline)                         # строка таблицы
    assert any("[-15-] {+5+}" in s for s in inline)                          # Confluence
    added = {c.new.text for c in changes if c.kind == "added"}
    assert "3. Укажите токен доступа в agent.conf." in added
    assert "3. Настройка SSO" in added
    deleted = {c.old.text for c in changes if c.kind == "deleted"}
    assert deleted == {"4. Экспорт", "Готовый документ можно выгрузить в Word."}
    # Неизменённые места в изменения не попали.
    assert not any("Резервные копии" in (c.new or c.old).text for c in changes)


def test_guide_pairs_one_per_section():
    pairs = to_source_pairs(*_guide())
    sections = [p.section[-1] for p in pairs]
    assert sections == ["1. Требования", "2. Установка", "3. Настройка SSO", "4. Экспорт", "4. Настройка интеграций"]
    integrations = pairs[-1]
    assert integrations.source_old.startswith("3. Настройка интеграций")  # старое имя раздела нашлось
    assert "каждые 5 минут" in integrations.source_new


def test_moved():
    changes = diff_texts("Шаг А.\n\nШаг Б.\n\nШаг В.", "Шаг Б.\n\nШаг А.\n\nШаг В.")
    assert [c.kind for c in changes] == ["moved"]


def test_eval_sources_roundtrip():
    """На всех 50 примерах этапа 2 diff находит изменение и отдаёт в LLM ровно те тексты, на которых мы мерили модель."""
    norm = lambda t: " ".join(t.split())
    for name in ["gold_v0.jsonl", "gold_v1_holdout.jsonl"]:
        for line in (ROOT / "data/eval" / name).read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            pairs = to_source_pairs(ex["source_old"], ex["source_new"])
            if ex["source_old"] == ex["source_new"]:
                assert pairs == [], ex["id"]
                continue
            assert len(pairs) == 1, ex["id"]
            assert norm(pairs[0].source_old) == norm(ex["source_old"]), ex["id"]
            assert norm(pairs[0].source_new) == norm(ex["source_new"]), ex["id"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
