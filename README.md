# Minimal CPU inference service

A FastAPI service with a deterministic echo backend. It repeats whitespace-split
prompt tokens up to `max_tokens`. No model weights, GPU, or external API calls.

## Install

Requires Python 3.11+ with venv support.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run

```sh
.venv/bin/uvicorn llm_serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

Configuration comes from environment variables at app creation:

| Variable | Default |
| --- | --- |
| `LLM_BACKEND` | `echo` |
| `LLM_MODEL_ID` | `echo-1` |
| `LLM_MAX_REQUEST_BYTES` | `1048576` |
| `LLM_MAX_PROMPT_CHARS` | `8192` |
| `LLM_MAX_TOKENS_LIMIT` | `512` |
| `LLM_MAX_CONCURRENCY` | `4` |
| `LLM_REQUEST_TIMEOUT_S` | `30` |

HTTP request bodies exceeding `LLM_MAX_REQUEST_BYTES` return 413 before parsing.
The limit applies to both declared Content-Length and streamed body bytes.

Numeric settings must be positive integers, including the timeout in seconds.
Unknown backends fail at startup. The configured model ID is authoritative for
both model discovery and completion requests.

## Try it

```sh
curl -i http://127.0.0.1:8000/healthz
curl -i http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/v1/models

curl -i http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: example-1' \
  -d '{"model":"echo-1","prompt":"hello world","max_tokens":5}'

curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"echo-1","prompt":"hello world","max_tokens":5,"temperature":0,"stop":["world"]}'

curl -i http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"echo-1","prompt":"hello","stream":true}'
```

The first completion produces `hello world hello world hello`, with 2 prompt
and 5 completion tokens. Stop strings truncate at the earliest match, excluding
the match; whitespace before the match is preserved. Usage counts whitespace-split
tokens in the final text. Empty stop lists mean no stopping; whitespace-only stop
strings are valid. Temperature is validated but ignored by the echo backend.

Errors use `{"error":{"type":"...","message":"..."}}`. The explicit readiness
contract is the exception: an unready backend returns 503 with
`{"status":"not_ready"}`. Over-capacity requests return 503 immediately;
generation timeouts return 504. Every response includes `X-Request-ID`.
Streaming is not supported (400).

## Test

```sh
.venv/bin/pytest -q
```

Tests use in-process clients and fake backends, with no network calls or weights.
Concurrency tests coordinate with events; the timeout test waits for a one-second
deadline. The suite should finish in under 10 seconds.
