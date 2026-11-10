"""Проверки HTTP-сервиса без модели. Запуск: python tests/test_server.py (нужен pip install -e .[server])"""

import json
import os
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from syncscribe_ml import server

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "data/contract/example_request.json").read_text(encoding="utf-8"))
REAL_RUN = server.run
ENV = ("SYNCSCRIBE_ML_MODEL", "SYNCSCRIBE_ML_API_KEY", "SYNCSCRIBE_ML_MAX_CONCURRENT", "SYNCSCRIBE_ML_MODE")


def fake_response(request: dict) -> dict:
    return {"contract_version": "2.0", "request_id": request.get("request_id"), "analysis_job_id": None,
            "status": "completed", "suggestions": [], "flags": [], "meta": {"seconds_total": 0.0}, "error": None}


def client(env: dict | None = None, fake=None) -> TestClient:
    for name in ENV:
        os.environ.pop(name, None)
    os.environ.update(env or {})
    server.run = fake or REAL_RUN
    return TestClient(server.app)



def test_analyze_returns_run_response_and_model_from_env():
    seen = {}

    def fake(request, model, prompts, **kwargs):
        seen.update(model=model, kwargs=kwargs)
        return fake_response(request)

    with client({"SYNCSCRIBE_ML_MODEL": "gpt-4o-mini"}, fake) as c:
        r = c.post("/v1/analyze", json=EXAMPLE, headers={"X-Request-ID": EXAMPLE.get("request_id") or "r-1"})
    assert r.status_code == 200, r.text
    assert r.json()["request_id"] == EXAMPLE.get("request_id")
    assert seen["model"] == "gpt-4o-mini"
    assert seen["kwargs"] == {"fast": True, "top_k": 3, "mode": "hybrid", "adds": True, "workers": 1}


def test_contract_errors_stay_in_body_with_200():
    # Настоящий run(): некорректный запрос отбрасывается до вызова модели.
    with client() as c:
        r = c.post("/v1/analyze", json={"request_id": "r-2", "document": {"blocks": []}})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "failed" and r.json()["error"]["code"] == "invalid_request"
    assert r.json()["request_id"] == "r-2"


def test_not_json_object_is_400():
    with client(fake=lambda *a, **k: fake_response({})) as c:
        assert c.post("/v1/analyze", content=b"not json").status_code == 400
        assert c.post("/v1/analyze", json=[1, 2]).status_code == 400


def test_api_key():
    with client({"SYNCSCRIBE_ML_API_KEY": "secret"}, lambda request, *a, **k: fake_response(request)) as c:
        assert c.post("/v1/analyze", json=EXAMPLE).status_code == 401
        assert c.post("/v1/analyze", json=EXAMPLE, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/v1/analyze", json=EXAMPLE, headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_busy_gives_429_and_health_still_answers():
    started, release = threading.Event(), threading.Event()

    def slow(request, *a, **k):
        started.set()
        release.wait(5)
        return fake_response(request)

    with client({"SYNCSCRIBE_ML_MAX_CONCURRENT": "1"}, slow) as c:
        first = {}
        worker = threading.Thread(target=lambda: first.update(r=c.post("/v1/analyze", json=EXAMPLE)))
        worker.start()
        assert started.wait(5)
        busy = c.post("/v1/analyze", json=EXAMPLE)
        health = c.get("/health")
        release.set()
        worker.join(5)
    assert busy.status_code == 429 and busy.headers["Retry-After"] == "30"
    assert health.status_code == 200 and health.json()["contract_version"] == "2.0"
    assert first["r"].status_code == 200


def test_bad_settings_fail_on_start():
    try:
        with client({"SYNCSCRIBE_ML_MODE": "magic"}):
            pass
    except ValueError as exc:
        assert "SYNCSCRIBE_ML_MODE" in str(exc)
    else:
        raise AssertionError("сервис не должен стартовать с неверным SYNCSCRIBE_ML_MODE")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
