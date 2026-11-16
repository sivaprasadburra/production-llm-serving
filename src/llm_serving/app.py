import asyncio
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.datastructures import Headers, MutableHeaders

from .backends import Backend, get_backend
from .config import Settings
from .schemas import Choice, CompletionRequest, CompletionResponse, ModelInfo, ModelsResponse, Usage

logger = logging.getLogger(__name__)


class RequestMiddleware:
    def __init__(self, app, max_request_bytes: int):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = headers.get("X-Request-ID", "")
        if not request_id or len(request_id) > 128 or not request_id.isprintable():
            request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = False
        received = 0

        async def send_with_id(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_request_bytes:
                    raise HTTPException(413, "request body exceeds maximum byte count")
            return message

        try:
            try:
                content_length = int(headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0  # Still enforce the actual received byte count.
            if content_length > self.max_request_bytes:
                raise HTTPException(413, "request body exceeds maximum byte count")
            await self.app(scope, limited_receive, send_with_id)
        except Exception as exc:
            if started:
                raise
            if isinstance(exc, StarletteHTTPException):
                status, message = exc.status_code, str(exc.detail)
            else:
                logger.exception("Request failed request_id=%s", request_id)
                status, message = 500, "internal server error"
            response = JSONResponse(
                {"error": {"type": "http_error", "message": message}}, status_code=status,
            )
            await response(scope, receive, send_with_id)


async def _disconnected(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _generate(request: Request, coro, timeout: float):
    """Await coro, cancelling it on timeout or client disconnect.

    Returns only once the backend task has actually finished, so the caller's
    concurrency slot is never released while generation is still running.
    """
    generation = asyncio.ensure_future(coro)
    watcher = asyncio.ensure_future(_disconnected(request))
    try:
        done, _ = await asyncio.wait(
            {generation, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if generation in done:
            return generation.result()
        if watcher in done:
            raise HTTPException(499, "client disconnected")
        raise HTTPException(504, "generation timed out")
    finally:
        generation.cancel()
        watcher.cancel()
        await asyncio.gather(generation, watcher, return_exceptions=True)


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    backend = backend if backend is not None else get_backend(settings.backend)
    in_flight = 0  # generations running in this app instance
    app = FastAPI()

    app.add_middleware(RequestMiddleware, max_request_bytes=settings.max_request_bytes)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            {"error": {"type": "http_error", "message": str(exc.detail)}},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Field names are client-controlled: bound what gets echoed back.
        errors = exc.errors()
        parts = [
            f"{'.'.join(str(part) for part in error['loc'])[:100]}: {error['msg'][:200]}"
            for error in errors[:5]
        ]
        if len(errors) > 5:
            parts.append(f"and {len(errors) - 5} more errors")
        message = "; ".join(parts)
        return JSONResponse({"error": {"type": "validation_error", "message": message}}, status_code=422)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if backend.is_ready():
            return {"status": "ready"}
        return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models():
        return ModelsResponse(data=[ModelInfo(id=settings.model_id)])

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def complete(body: CompletionRequest, request: Request):
        nonlocal in_flight
        if body.model != settings.model_id:
            raise HTTPException(404, "model not found")
        if body.stream:
            raise HTTPException(400, "streaming not supported yet")
        if len(body.prompt) > settings.max_prompt_chars:
            raise HTTPException(413, "prompt exceeds maximum character count")
        if body.max_tokens > settings.max_tokens_limit:
            raise HTTPException(422, "max_tokens exceeds configured limit")
        if in_flight >= settings.max_concurrency:  # fail fast, never queue
            raise HTTPException(503, "generation capacity exhausted")
        in_flight += 1
        try:
            result = await _generate(
                request,
                backend.generate(body.prompt, body.max_tokens, body.temperature, body.stop),
                settings.request_timeout_s,
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "Backend generation failed request_id=%s", request.state.request_id
            )
            raise HTTPException(500, "internal server error") from None
        finally:
            in_flight -= 1
        return CompletionResponse(
            id=f"cmpl-{uuid4().hex}", created=int(time.time()), model=settings.model_id,
            choices=[Choice(text=result.text, finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens),
        )

    return app
