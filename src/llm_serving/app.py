import asyncio
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .backends import Backend, get_backend
from .config import Settings
from .schemas import Choice, CompletionRequest, CompletionResponse, ModelInfo, ModelsResponse, Usage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    backend = backend if backend is not None else get_backend(settings.backend)
    limiter = asyncio.Semaphore(settings.max_concurrency)
    app = FastAPI()

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        value = request.headers.get("X-Request-ID", "")
        if not value or len(value) > 128 or not value.isprintable():
            value = uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Request failed")
            response = JSONResponse(
                {"error": {"type": "http_error", "message": "internal server error"}},
                status_code=500,
            )
        response.headers["X-Request-ID"] = value
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            {"error": {"type": "http_error", "message": str(exc.detail)}},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
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
    async def complete(body: CompletionRequest):
        if body.model != settings.model_id:
            raise HTTPException(404, "model not found")
        if body.stream:
            raise HTTPException(400, "streaming not supported yet")
        if len(body.prompt) > settings.max_prompt_chars:
            raise HTTPException(413, "prompt exceeds maximum character count")
        if body.max_tokens > settings.max_tokens_limit:
            raise HTTPException(422, "max_tokens exceeds configured limit")
        # No suspension between checking availability and acquiring a free slot.
        if limiter.locked():
            raise HTTPException(503, "generation capacity exhausted")
        await limiter.acquire()
        try:
            result = await asyncio.wait_for(
                backend.generate(body.prompt, body.max_tokens, body.temperature, body.stop),
                timeout=settings.request_timeout_s,
            )
            return CompletionResponse(
                id=f"cmpl-{uuid4().hex}", created=int(time.time()), model=settings.model_id,
                choices=[Choice(text=result.text, finish_reason=result.finish_reason)],
                usage=Usage(prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            total_tokens=result.prompt_tokens + result.completion_tokens),
            )
        except TimeoutError:
            raise HTTPException(504, "generation timed out") from None
        except Exception:
            logger.exception("Backend generation failed")
            raise HTTPException(500, "internal server error") from None
        finally:
            limiter.release()

    return app
