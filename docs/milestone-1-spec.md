# Milestone 1 — Minimal CPU inference service

Goal: a small, fully tested FastAPI service that defines the serving contract.
No model weights, no GPU, no network calls. A real model backend is milestone 2.

## Layout

```
pyproject.toml              # package "llm_serving", Python >=3.11
src/llm_serving/
  __init__.py
  config.py                 # Settings dataclass loaded from env vars
  schemas.py                # Pydantic v2 request/response models
  backends/__init__.py      # Backend Protocol + get_backend(name) factory
  backends/echo.py          # Deterministic CPU EchoBackend
  app.py                    # create_app(settings=None, backend=None) factory
tests/                      # pytest, uses fastapi.testclient
README.md                   # how to install, run, test, curl examples
```

Runtime deps: `fastapi`, `uvicorn[standard]`, `pydantic>=2`.
Dev extra (`[dev]`): `pytest`, `httpx`. Nothing else.

## Config (env vars, prefix `LLM_`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `echo` | backend name; unknown name -> startup error |
| `LLM_MODEL_ID` | `echo-1` | model id reported by `/v1/models` |
| `LLM_MAX_REQUEST_BYTES` | `1048576` | larger HTTP body -> 413 before parsing (Content-Length or streamed) |
| `LLM_MAX_PROMPT_CHARS` | `8192` | longer prompt -> 413 |
| `LLM_MAX_TOKENS_LIMIT` | `512` | upper bound for `max_tokens` |
| `LLM_MAX_CONCURRENCY` | `4` | in-flight generations; extra requests get 503 immediately |
| `LLM_REQUEST_TIMEOUT_S` | `30` | generation timeout -> 504 |

Invalid values (non-int, <=0) must raise a clear error at startup.

## Backend contract

```python
class Backend(Protocol):
    model_id: str
    def is_ready(self) -> bool: ...
    async def generate(self, prompt: str, max_tokens: int, temperature: float,
                       stop: list[str] | None) -> GenerationResult: ...
```
`GenerationResult(text: str, prompt_tokens: int, completion_tokens: int, finish_reason: Literal["stop","length"])`.

EchoBackend: tokens = whitespace split. Output = prompt tokens repeated
cyclically until `max_tokens` (finish_reason `length`), truncated at the first
`stop` string if present (finish_reason `stop`). Empty output is allowed.
Deterministic; ignores temperature.

## HTTP API

- `GET /healthz` -> 200 `{"status":"ok"}` (process liveness).
- `GET /readyz` -> 200 `{"status":"ready"}` or 503 `{"status":"not_ready"}` from `backend.is_ready()` (the one exception to the error envelope rule below).
- `GET /v1/models` -> `{"object":"list","data":[{"id":..., "object":"model", "owned_by":"local"}]}`.
- `POST /v1/completions`, request:
  - `model: str` (required; must equal the configured model id, else 404)
  - `prompt: str` (required, non-empty after strip)
  - `max_tokens: int = 16` (strict int, 1..LLM_MAX_TOKENS_LIMIT, else 422)
  - `temperature: float = 1.0` (0..2)
  - `stop: str | list[str] | None` (max 4 non-empty items)
  - `stream: bool = false`; `true` -> 400 "streaming not supported yet"
  - unknown fields -> 422 (`extra="forbid"`)
- Response (OpenAI-compatible subset):
  `{"id":"cmpl-<uuid hex>","object":"text_completion","created":<unix int>,"model":...,
    "choices":[{"index":0,"text":...,"finish_reason":...}],
    "usage":{"prompt_tokens":..,"completion_tokens":..,"total_tokens":..}}`

Errors: all non-2xx responses use `{"error":{"type":..., "message":...}}`,
including FastAPI's validation errors (422) and HTTPExceptions. Do not leak
stack traces; unexpected backend exceptions -> 500 with a generic message
and the exception logged.

Middleware: accept `X-Request-ID` (reuse if present and <=128 printable chars,
else generate uuid4 hex) and echo it on every response.

Concurrency: an `asyncio.Semaphore`-style limiter created per app instance
(not module-global). If no slot is free, return 503 immediately (do not
queue). Always release the slot, including on timeout/exception.

## Tests (pytest, no network, must run in < 10s)

Cover at least: health/ready (including a not-ready fake backend), models list,
successful completion + usage math, stop handling, max_tokens bounds, empty
prompt, oversized prompt (413), wrong model (404), stream=true (400), extra
field (422), error envelope shape, request-ID echo + generation, concurrency
limit returns 503 (use a fake slow backend, deterministic — no sleeps-based
races), timeout returns 504, backend exception -> 500 without leaking message,
slot released after errors, config parsing (valid + invalid env), EchoBackend
unit tests.

## Non-goals for milestone 1
Streaming, auth, real models, metrics endpoint, Docker, CI.
