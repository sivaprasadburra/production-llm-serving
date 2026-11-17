"""llama-server adapter tests. No model weights: the upstream is a programmable fake."""
import asyncio
import json
import threading

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from llm_serving.app import create_app
from llm_serving.backends import get_backend
from llm_serving.backends.llama_server import LlamaServerBackend
from llm_serving.config import Settings
from test_service import BODY, asgi_post, assert_error, payload

SETTINGS = dict(backend="llama_server", model_id="tiny", max_concurrency=1)
COMPLETION = {
    "choices": [{"index": 0, "text": " there was a cat", "finish_reason": "length"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


class FakeLlamaServer:
    """Minimal ASGI stand-in for llama-server with scriptable behaviour."""

    def __init__(self):
        self.health = (200, {"status": "ok"})
        self.props = (200, {"total_slots": 1})
        self.completion = (200, COMPLETION)  # (status, json) | (status, raw bytes) | "hang"
        self.requests = []  # (path, headers, parsed json body)
        self.entered = asyncio.Event()
        self.aborted = False  # generation was cut off by the caller going away

    async def __call__(self, scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        self.requests.append((scope["path"], headers, json.loads(body) if body else None))
        if scope["path"] == "/health":
            return await self._respond(send, *self.health)
        if scope["path"] == "/props":
            return await self._respond(send, *self.props)
        if self.completion == "hang":
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.aborted = True
                raise
        await self._respond(send, *self.completion)

    @staticmethod
    async def _respond(send, status, content):
        raw = content if isinstance(content, bytes) else json.dumps(content).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": raw})


def make_app(fake=None, transport=None, **overrides):
    settings = Settings(**{**SETTINGS, **overrides})
    transport = transport or httpx.ASGITransport(app=fake)
    return create_app(settings, LlamaServerBackend(settings, transport=transport))


def post(fake=None, transport=None, body=None, **overrides):
    with TestClient(make_app(fake, transport, **overrides)) as client:
        return client.post("/v1/completions", json=body or payload(model="tiny"))


def test_completion_maps_request_and_response():
    fake = FakeLlamaServer()
    response = post(fake, body=payload(model="tiny", prompt="Once upon a time", max_tokens=4,
                                       temperature=0.5, stop=["\n"]),
                    llama_server_api_key="s3cret")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tiny"
    assert body["choices"] == [{"index": 0, "text": " there was a cat", "finish_reason": "length"}]
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    path, headers, sent = fake.requests[-1]
    assert path == "/v1/completions"
    assert sent == {"prompt": "Once upon a time", "max_tokens": 4, "temperature": 0.5,
                    "stream": False, "stop": ["\n"]}
    assert headers["authorization"] == "Bearer s3cret"


def test_no_api_key_and_no_stop_are_omitted():
    fake = FakeLlamaServer()
    fake.completion = (200, {**COMPLETION, "choices": [{"text": "x", "finish_reason": "stop"}]})
    response = post(fake)
    assert response.json()["choices"][0]["finish_reason"] == "stop"
    _, headers, sent = fake.requests[-1]
    assert "authorization" not in headers
    assert "stop" not in sent


@pytest.mark.parametrize("health,status,body", [
    ((200, {"status": "ok"}), 200, {"status": "ready"}),
    ((503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}),
     503, {"status": "not_ready"}),
    ((500, b"oops"), 503, {"status": "not_ready"}),
])
def test_readiness_follows_upstream_health(health, status, body):
    fake = FakeLlamaServer()
    fake.health = health
    with TestClient(make_app(fake)) as client:
        response = client.get("/readyz")
        assert (response.status_code, response.json()) == (status, body)
        assert client.get("/healthz").status_code == 200  # liveness is independent of upstream


def test_slot_mismatch_is_warned_once_per_upstream_start(caplog):
    fake = FakeLlamaServer()
    with TestClient(make_app(fake, max_concurrency=4)) as client:
        assert client.get("/readyz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert caplog.text.count("exceeds llama-server slots (--parallel 1)") == 1
        fake.health = (503, {})  # llama-server restarts, possibly with new settings
        assert client.get("/readyz").status_code == 503
        fake.health = (200, {"status": "ok"})
        assert client.get("/readyz").status_code == 200
        assert caplog.text.count("exceeds llama-server slots") == 2


@pytest.mark.parametrize("props", [(200, {"total_slots": 4}), (200, {"total_slots": 8}),
                                   (404, b""), (200, b"nope"), (200, {"other": 1})])
def test_no_slot_warning_when_matched_or_props_unavailable(caplog, props):
    fake = FakeLlamaServer()
    fake.props = props
    with TestClient(make_app(fake, max_concurrency=4)) as client:
        assert client.get("/readyz").status_code == 200
    assert "exceeds llama-server slots" not in caplog.text


def failing_transport(exc):
    def handler(request):
        raise exc
    return httpx.MockTransport(handler)


def test_upstream_down_is_not_ready_and_503():
    """The service starts and stays alive while llama-server is down."""
    transport = failing_transport(httpx.ConnectError("refused"))
    with TestClient(make_app(transport=transport)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        response = client.post("/v1/completions", json=payload(model="tiny"))
        assert_error(response, 503)
        assert response.json()["error"]["message"] == "model backend unavailable"


@pytest.mark.parametrize("exc,status", [
    (httpx.ConnectTimeout("slow"), 503),
    (httpx.ReadError("connection reset mid-generation"), 502),
    (httpx.RemoteProtocolError("garbage"), 502),
])
def test_transport_failures(exc, status):
    response = post(transport=failing_transport(exc))
    assert_error(response, status)
    assert str(exc) not in response.text


@pytest.mark.parametrize("completion,status,message", [
    ((503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}),
     503, "model backend unavailable"),
    ((400, {"error": {"code": 400, "message": "request exceeds the available context size",
                      "type": "exceed_context_size_error"}}),
     400, "request exceeds the available context size"),
    ((400, {"error": {"message": "m" * 5000}}), 400, "m" * 200),
    ((400, b"not json"), 400, "request rejected by model backend"),
    ((400, {"error": "flat string"}), 400, "request rejected by model backend"),
    ((401, {"error": {"message": "Invalid API Key"}}), 502, "model backend error"),
    ((500, {"error": {"message": "internal detail"}}), 502, "model backend error"),
    ((200, b"<html>not json</html>"), 502, "model backend error"),
    ((200, {"choices": []}), 502, "model backend error"),
    ((200, {"choices": [{"text": "x"}]}), 502, "model backend error"),
    ((200, {**COMPLETION, "choices": [{"text": None}]}), 502, "model backend error"),
    ((200, {**COMPLETION, "usage": {"prompt_tokens": "5", "completion_tokens": 4}}), 502,
     "model backend error"),
    ((200, [1, 2, 3]), 502, "model backend error"),
])
def test_upstream_responses(completion, status, message):
    fake = FakeLlamaServer()
    fake.completion = completion
    response = post(fake, llama_server_api_key="s3cret")
    assert_error(response, status)
    assert response.json()["error"]["message"] == message
    assert "s3cret" not in response.text


def test_slot_is_released_after_backend_errors():
    fake = FakeLlamaServer()
    with TestClient(make_app(fake)) as client:
        fake.completion = (500, b"")
        assert client.post("/v1/completions", json=payload(model="tiny")).status_code == 502
        fake.completion = (200, COMPLETION)
        assert client.post("/v1/completions", json=payload(model="tiny")).status_code == 200


TINY_BODY = BODY.replace(b"echo-1", b"tiny")


def test_timeout_aborts_upstream_generation():
    async def scenario():
        fake = FakeLlamaServer()
        fake.completion = "hang"
        app = make_app(fake, request_timeout_s=0.05)
        sent = await asyncio.wait_for(asgi_post(app, TINY_BODY), 2)
        assert sent[0]["status"] == 504
        assert fake.aborted
        fake.completion = (200, COMPLETION)  # slot is free again
        assert (await asgi_post(app, TINY_BODY))[0]["status"] == 200

    asyncio.run(scenario())


def test_client_disconnect_aborts_upstream_generation():
    async def scenario():
        fake = FakeLlamaServer()
        fake.completion = "hang"
        app = make_app(fake)
        disconnect = asyncio.Event()
        request = asyncio.create_task(asgi_post(app, TINY_BODY, disconnect))
        await asyncio.wait_for(fake.entered.wait(), 2)
        assert (await asgi_post(app, TINY_BODY))[0]["status"] == 503  # at capacity, fail fast
        disconnect.set()
        assert (await asyncio.wait_for(request, 2))[0]["status"] == 499
        assert fake.aborted

    asyncio.run(scenario())


def test_real_socket_timeout_closes_upstream_connection():
    """Over real TCP: a timed-out request must drop the upstream connection.

    llama-server stops generating (and frees its slot) when the connection closes,
    so this is the property that makes our timeout actually release CPU.
    """
    saw_disconnect = threading.Event()

    async def hanging_upstream(scope, receive, send):
        if scope["type"] != "http":
            return
        while (await receive())["type"] != "http.disconnect":
            pass
        saw_disconnect.set()

    server = uvicorn.Server(uvicorn.Config(hanging_upstream, host="127.0.0.1", port=0,
                                           log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            threading.Event().wait(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        settings = Settings(**SETTINGS, request_timeout_s=0.2,
                            llama_server_url=f"http://127.0.0.1:{port}")
        with TestClient(create_app(settings)) as client:
            assert_error(client.post("/v1/completions", json=payload(model="tiny")), 504)
        assert saw_disconnect.wait(5)
    finally:
        server.should_exit = True
        thread.join(5)


def test_closed_local_port_fails_within_the_connect_timeout():
    """Nothing listens on the port: the gateway must answer 503 quickly on every OS.

    Linux refuses loopback connections at once. Windows retries them, so the
    failure only arrives when our connect timeout fires (measured on a
    GitHub-hosted Windows runner: 2.0 s with the default 2 s timeout). The
    connect timeout is therefore the upper bound for noticing an outage.
    """
    import socket
    import time

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # free now, and closed once the socket is gone
    settings = Settings(**SETTINGS, llama_server_url=f"http://127.0.0.1:{port}",
                        llama_server_connect_timeout_s=0.3)
    with TestClient(create_app(settings)) as client:
        started = time.perf_counter()
        assert_error(client.post("/v1/completions", json=payload(model="tiny")), 503)
        assert client.get("/readyz").status_code == 503
        assert time.perf_counter() - started < 2.0  # two attempts, each bounded by 0.3 s
        assert client.get("/healthz").status_code == 200


def test_client_is_closed_on_shutdown_and_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    settings = Settings(**SETTINGS)
    backend = get_backend(settings)
    assert isinstance(backend, LlamaServerBackend)
    assert not backend._client.trust_env
    with TestClient(create_app(settings, backend)):
        assert not backend._client.is_closed
    assert backend._client.is_closed


def test_llama_settings(monkeypatch):
    for name in Settings.__dataclass_fields__:
        monkeypatch.delenv(f"LLM_{name.upper()}", raising=False)
    monkeypatch.setenv("LLM_LLAMA_SERVER_URL", "http://10.0.0.5:9000")
    monkeypatch.setenv("LLM_LLAMA_SERVER_API_KEY", "s3cret")
    monkeypatch.setenv("LLM_LLAMA_SERVER_CONNECT_TIMEOUT_S", "0.5")
    settings = Settings.from_env()
    assert settings.llama_server_url == "http://10.0.0.5:9000"
    assert settings.llama_server_connect_timeout_s == 0.5
    assert "s3cret" not in repr(settings)
    for bad in ["localhost:8080", "ftp://x", ""]:
        with pytest.raises(ValueError, match="LLM_LLAMA_SERVER_URL"):
            Settings(llama_server_url=bad)
    with pytest.raises(ValueError, match="LLM_LLAMA_SERVER_CONNECT_TIMEOUT_S"):
        Settings(llama_server_connect_timeout_s=0)
