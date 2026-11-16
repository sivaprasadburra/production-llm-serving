# Minimal CPU inference service

A FastAPI service with a deterministic echo backend. It repeats whitespace-split
prompt tokens up to `max_tokens`. No model weights, GPU, or external API calls.

## Install

Requires Python 3.11+ with venv support.

Linux / macOS:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Windows (PowerShell, Python 3.11+ from python.org or `winget install Python.Python.3.12`):

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

No activation is needed; the commands below call the venv's executables directly.
On Windows replace `.venv/bin/` with `.venv\Scripts\`.

## Run

```sh
.venv/bin/uvicorn llm_serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

Windows:

```powershell
.venv\Scripts\uvicorn llm_serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

Configuration comes from environment variables at app creation
(PowerShell: `$env:LLM_MAX_CONCURRENCY = "2"` before starting the server):

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

Numeric settings must be positive integers, except `LLM_REQUEST_TIMEOUT_S`,
which accepts any positive number of seconds (for example `2.5`).
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

PowerShell equivalents (`curl` is an alias for `Invoke-WebRequest` in Windows
PowerShell 5, so use `curl.exe` or `Invoke-RestMethod`):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/v1/completions -Method Post -ContentType 'application/json' `
  -Body '{"model":"echo-1","prompt":"hello world","max_tokens":5}'
```

## Failure behavior

| Situation | Response |
| --- | --- |
| Body larger than `LLM_MAX_REQUEST_BYTES` | 413, rejected before parsing |
| All `LLM_MAX_CONCURRENCY` slots busy | 503 immediately (no queueing) |
| Generation exceeds `LLM_REQUEST_TIMEOUT_S` | 504, backend task cancelled |
| Client disconnects mid-generation | backend task cancelled, slot freed (logged as 499) |
| Backend raises | 500 with a generic message; details and request ID in the server log |

A slot is released only after the backend task has actually stopped, so a
backend that is slow to honour cancellation cannot be oversubscribed.

## Test

```sh
.venv/bin/pytest -q            # Linux / macOS
.venv\Scripts\pytest -q        # Windows
```

CI runs the suite on Ubuntu and Windows with Python 3.11 and 3.12.

Tests use in-process clients and fake backends, with no network calls or weights.
Concurrency, disconnect and cancellation tests coordinate with events rather than sleeps. The suite should finish in under 10 seconds.

## Benchmark

With the service running, use its configured model ID (`echo-1` by default):

Linux:

```sh
.venv/bin/python benchmarks/bench.py --model echo-1 --requests 20 --concurrency 2 --label "echo" --output benchmark.json
```

Windows (PowerShell):

```powershell
.venv\Scripts\python benchmarks/bench.py --model echo-1 --requests 20 --concurrency 2 --label "echo" --output benchmark.json
```

Use `--url` for another service, `--prompt` or `--prompt-file` (UTF-8) for input,
`--max-tokens` and `--temperature` for generation, and `--timeout` for the per-request
HTTP timeout. `--warmup` sequential requests run first (default 1, 0 disables them);
they are excluded from counts and measured wall time. Workers send the next request
as soon as the previous one finishes. Failures are counted without retries.

A short summary always prints; `--output` optionally saves JSON with configuration
(prompt length only), environment, successful-request latency percentiles, aggregate
token throughput and mean per-request token speed. RAM is reported in GiB rounded
to one decimal. With no successes, latency and mean per-request token speed are
null and the exit code is 1; otherwise it is 0.
