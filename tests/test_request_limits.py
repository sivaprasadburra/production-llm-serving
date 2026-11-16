import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_serving.app import create_app
from llm_serving.config import Settings


BODY = json.dumps({"model": "echo-1", "prompt": "hello"}).encode()


def test_boolean_max_tokens():
    with TestClient(create_app(Settings())) as client:
        response = client.post("/v1/completions", json={
            "model": "echo-1", "prompt": "hello", "max_tokens": True,
        })
        assert response.status_code == 422


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("over_limit", [False, True])
def test_body_limit(streamed, over_limit):
    async def scenario():
        settings = Settings(max_request_bytes=len(BODY) - int(over_limit))
        consumed = 0

        async def chunks():
            nonlocal consumed
            for chunk in [BODY[:10], BODY[10:]]:
                consumed += 1
                yield chunk
            if over_limit:
                pytest.fail("should stop reading as soon as the limit is exceeded")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(settings)), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", content=chunks() if streamed else BODY,
                headers={"Content-Type": "application/json", "X-Request-ID": "body-test"},
            )
        assert response.status_code == (413 if over_limit else 200)
        assert response.headers["x-request-id"] == "body-test"
        if over_limit:
            assert response.json() == {"error": {
                "type": "http_error", "message": "request body exceeds maximum byte count",
            }}
        if streamed:
            assert "content-length" not in response.request.headers
            assert consumed == 2

    asyncio.run(scenario())


def test_content_length_rejected_before_reading():
    async def scenario():
        settings = Settings(max_request_bytes=10)

        async def unread_body():
            pytest.fail("oversized Content-Length must be rejected without reading")
            yield b""

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(settings)), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", content=unread_body(), headers={"Content-Length": "11"},
            )
        assert response.status_code == 413
        assert len(response.headers["x-request-id"]) == 32
        assert response.json()["error"]["type"] == "http_error"

    asyncio.run(scenario())


def test_limit_does_not_send_second_response():
    from llm_serving.app import RequestMiddleware
    from fastapi import HTTPException

    async def scenario():
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await receive()

        async def receive():
            return {"type": "http.request", "body": b"too large", "more_body": False}

        async def send(message):
            sent.append(message)

        middleware = RequestMiddleware(app, max_request_bytes=1)
        with pytest.raises(HTTPException) as error:
            await middleware({"type": "http", "headers": []}, receive, send)
        assert error.value.status_code == 413
        assert len(sent) == 1
        assert sent[0]["status"] == 200

    asyncio.run(scenario())
