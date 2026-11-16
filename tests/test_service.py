import asyncio
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_serving.app import create_app
from llm_serving.backends import get_backend
from llm_serving.backends.echo import EchoBackend
from llm_serving.config import Settings


@pytest.fixture
def client():
    with TestClient(create_app(Settings())) as client:
        yield client


def payload(**changes):
    return {"model": "echo-1", "prompt": "one two", **changes}


def assert_error(response, status):
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    assert set(response.json()["error"]) == {"type", "message"}
    assert all(isinstance(value, str) and value for value in response.json()["error"].values())
    assert response.headers["x-request-id"]


def test_health_ready_models(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/v1/models").json() == {
        "object": "list", "data": [{"id": "echo-1", "object": "model", "owned_by": "local"}]
    }


def test_not_ready():
    class NotReady(EchoBackend):
        def is_ready(self):
            return False

    with TestClient(create_app(Settings(), NotReady())) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}
        assert response.headers["x-request-id"]
        assert client.get("/healthz").status_code == 200


def test_success(client):
    response = client.post("/v1/completions", json=payload(max_tokens=5))
    assert response.status_code == 200
    body = response.json()
    assert re.fullmatch(r"cmpl-[0-9a-f]{32}", body.pop("id"))
    assert type(body.pop("created")) is int
    assert body == {
        "object": "text_completion", "model": "echo-1",
        "choices": [{"index": 0, "text": "one two one two one", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
    }
    assert client.post("/v1/completions", json=payload()).json()["usage"]["completion_tokens"] == 16


@pytest.mark.parametrize("stop,text,count", [("two", "one ", 1), (["two", "one"], "", 0)])
def test_stop(client, stop, text, count):
    body = client.post("/v1/completions", json=payload(stop=stop)).json()
    assert body["choices"][0] == {"index": 0, "text": text, "finish_reason": "stop"}
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": count, "total_tokens": 2 + count}


@pytest.mark.parametrize("changes,status", [
    ({"max_tokens": 0}, 422), ({"max_tokens": -1}, 422), ({"max_tokens": 513}, 422),
    ({"prompt": ""}, 422), ({"prompt": " \n\t"}, 422), ({"prompt": "x" * 8193}, 413),
    ({"model": "other"}, 404), ({"stream": True}, 400), ({"unknown": 1}, 422),
    ({"temperature": -0.1}, 422), ({"temperature": 2.1}, 422),
    ({"stop": ""}, 422), ({"stop": [""]}, 422), ({"stop": ["a"] * 5}, 422),
])
def test_invalid_requests(client, changes, status):
    response = client.post("/v1/completions", json=payload(**changes))
    assert_error(response, status)
    if changes.get("stream"):
        assert response.json()["error"]["message"] == "streaming not supported yet"


@pytest.mark.parametrize("body", [{}, {"model": "echo-1"}, {"prompt": "hi"}])
def test_required_fields(client, body):
    assert_error(client.post("/v1/completions", json=body), 422)


def test_boundaries_and_custom_settings():
    with TestClient(create_app(Settings(model_id="custom", max_tokens_limit=2, max_prompt_chars=3))) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "custom"
        for count in [1, 2]:
            response = client.post("/v1/completions", json=payload(model="custom", prompt="abc", max_tokens=count))
            assert response.status_code == 200
            assert response.json()["model"] == "custom"
        assert_error(client.post("/v1/completions", json=payload(model="custom", prompt="abc", max_tokens=3)), 422)


@pytest.mark.parametrize("request_id", [None, "", "a" * 129, "bad\tvalue", "chosen-id", "a" * 128])
def test_request_id(client, request_id):
    headers = {} if request_id is None else {"X-Request-ID": request_id}
    for path in ["/healthz", "/missing"]:
        response = client.get(path, headers=headers)
        value = response.headers["x-request-id"]
        if request_id and len(request_id) <= 128 and request_id.isprintable():
            assert value == request_id
        else:
            assert re.fullmatch(r"[0-9a-f]{32}", value)


def test_framework_errors(client):
    assert_error(client.get("/missing"), 404)
    assert_error(client.get("/v1/completions"), 405)
    assert_error(client.post("/v1/completions", content="{", headers={"Content-Type": "application/json"}), 422)


def test_concurrency_and_per_app_limit():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Blocking(EchoBackend):
            async def generate(self, *args):
                entered.set()
                await release.wait()
                return await super().generate(*args)

        app = create_app(Settings(max_concurrency=1), Blocking())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/completions", json=payload()))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                response = await asyncio.wait_for(client.post("/v1/completions", json=payload()), 2)
                assert_error(response, 503)
                other = create_app(Settings(max_concurrency=1))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(other), base_url="http://test") as other_client:
                    assert (await other_client.post("/v1/completions", json=payload())).status_code == 200
            finally:
                release.set()
                response = await asyncio.wait_for(first, 2)
            assert response.status_code == 200
            assert (await client.post("/v1/completions", json=payload())).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("failure,status", [("timeout", 504), ("exception", 500)])
def test_failures_release_slot(failure, status, caplog):
    class FailsOnce(EchoBackend):
        failed = False

        async def generate(self, *args):
            if not self.failed:
                self.failed = True
                if failure == "timeout":
                    await asyncio.Event().wait()
                raise RuntimeError("private backend detail")
            return await super().generate(*args)

    with TestClient(create_app(Settings(max_concurrency=1, request_timeout_s=1), FailsOnce())) as client:
        response = client.post("/v1/completions", json=payload())
        assert_error(response, status)
        assert "private backend detail" not in response.text
        assert "Traceback" not in response.text
        assert client.post("/v1/completions", json=payload()).status_code == 200
    if failure == "exception":
        assert "private backend detail" in caplog.text


def test_readiness_exception_is_generic():
    class Broken(EchoBackend):
        def is_ready(self):
            raise RuntimeError("secret")

    with TestClient(create_app(Settings(), Broken())) as client:
        response = client.get("/readyz")
        assert_error(response, 500)
        assert "secret" not in response.text


@pytest.mark.parametrize("prompt,maximum,stop,text,reason,prompt_count,completion_count", [
    ("a b", 5, None, "a b a b a", "length", 2, 5),
    (" a\t b\n", 1, None, "a", "length", 2, 1),
    ("abc def", 4, ["def", "bc"], "a", "stop", 2, 1),
    ("a b", 4, ["a"], "", "stop", 2, 0),
    ("a b", 2, ["missing"], "a b", "length", 2, 2),
    (" \t", 2, None, "", "length", 0, 0),
])
def test_echo(prompt, maximum, stop, text, reason, prompt_count, completion_count):
    backend = EchoBackend()
    result = asyncio.run(backend.generate(prompt, maximum, 0, stop))
    assert result == asyncio.run(backend.generate(prompt, maximum, 2, stop))
    assert (result.text, result.finish_reason, result.prompt_tokens, result.completion_tokens) == (
        text, reason, prompt_count, completion_count
    )
    assert backend.is_ready()
    assert backend.model_id == "echo-1"


@pytest.fixture
def clean_env(monkeypatch):
    for name in Settings.__dataclass_fields__:
        monkeypatch.delenv(f"LLM_{name.upper()}", raising=False)


def test_config_defaults_and_valid_env(clean_env, monkeypatch):
    assert Settings.from_env() == Settings()
    values = dict(backend="echo", model_id="custom", max_prompt_chars=123,
                  max_tokens_limit=12, max_concurrency=2, request_timeout_s=3)
    for name, value in values.items():
        monkeypatch.setenv(f"LLM_{name.upper()}", str(value))
    assert Settings.from_env() == Settings(**values)
    with TestClient(create_app()) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "custom"


@pytest.mark.parametrize("name", ["MAX_PROMPT_CHARS", "MAX_TOKENS_LIMIT", "MAX_CONCURRENCY", "REQUEST_TIMEOUT_S"])
@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1", ""])
def test_invalid_config(clean_env, monkeypatch, name, value):
    monkeypatch.setenv(f"LLM_{name}", value)
    with pytest.raises(ValueError, match=f"LLM_{name} must be a positive integer"):
        create_app()


def test_unknown_backend(clean_env, monkeypatch):
    assert isinstance(get_backend("echo"), EchoBackend)
    monkeypatch.setenv("LLM_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        create_app()
