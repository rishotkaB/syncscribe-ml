"""Проверки контракта 2.0 без модели. Запуск: python tests/test_contract.py"""

import json
from pathlib import Path

from syncscribe_ml.contract import (BlockResult, ContractError, modify_suggestion, narrow, parse_request, run,
                                    source_chunks)
from syncscribe_ml.schema import Quote, TruthResultV3

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "data/contract/example_request.json").read_text(encoding="utf-8"))


def test_parse_example():
    blocks, sources = parse_request(EXAMPLE)
    assert [b.block_id for b in blocks if b.checkable] == ["d-2", "d-4"]
    assert blocks[3].section == "Оплата"
    assert sources[0][0] == "Решения встречи от 02.11.2026"


def test_sources_list_is_accepted():
    req = dict(EXAMPLE, sources=[EXAMPLE["source"], dict(EXAMPLE["source"], name="FAQ")])
    del req["source"]
    assert [name for name, _ in parse_request(req)[1]] == ["Решения встречи от 02.11.2026", "FAQ"]


def test_bad_request_gives_failed_status():
    for bad in ({}, {"document": {"blocks": []}}, {"document": {"blocks": [{"text": "x"}]}, "source": {"text": "y"}}):
        out = run(bad, "qwen3:8b")
        assert out["status"] == "failed" and out["error"]["code"] == "invalid_request", out
        assert out["contract_version"] == "2.0"
    try:
        parse_request({"document": {"blocks": [{"block_id": "a", "text": "1"}, {"block_id": "a", "text": "2"}]}, "source": {"text": "s"}})
        raise AssertionError("повтор block_id должен быть ошибкой")
    except ContractError:
        pass


def test_list_items_become_chunks():
    chunks = source_chunks("встреча", EXAMPLE["source"]["text"])
    texts = [c.text for c in chunks]
    assert "- Срок ответа на обращение: 5 рабочих дней." in texts
    assert "- Добавлена оплата по счёту для юридических лиц." in texts
    assert chunks[-1].section == "Решения"


def test_narrow_to_changed_sentence():
    old = "Оплата принимается картой. Срок возврата — 10 дней. Чек приходит на почту."
    new = "Оплата принимается картой. Срок возврата — 14 дней. Чек приходит на почту."
    assert narrow(old, new) == ("Срок возврата — 10 дней.", "Срок возврата — 14 дней.")
    assert narrow("Одно предложение.", "Другое предложение.") == ("Одно предложение.", "Другое предложение.")


def test_bad_quote_caps_confidence():
    blocks, _ = parse_request(EXAMPLE)
    res = TruthResultV3(analysis="a", verdict="contradicts", new_text="Срок ответа на обращение — 5 рабочих дней.",
                        quotes=[Quote(source=1, text="выдумка")], rationale="Срок сократили.", confidence_score=0.97)
    s = modify_suggestion(blocks[1], BlockResult("contradicts", res, [("встреча", "выдумка")], bad_quotes=1))
    assert s["confidence_score"] == 0.4 and s["old_text"] == blocks[1].text and s["insert_after"] is None
    same = TruthResultV3(analysis="a", verdict="contradicts", new_text=blocks[1].text, quotes=[], rationale="", confidence_score=0.9)
    assert modify_suggestion(blocks[1], BlockResult("contradicts", same)) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
