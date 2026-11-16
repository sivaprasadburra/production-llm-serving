# CPU LLM inference service

A FastAPI service that exposes an OpenAI-style `/v1/completions` API with
production guard rails (size limits, fail-fast concurrency, deadlines,
cancellation). Two backends:

- `echo` (default): deterministic, repeats whitespace-split prompt tokens up to
  `max_tokens`. No model weights, GPU, or external calls. Used by the tests.
- `llama_server`: real CPU inference through a separately running llama.cpp
  `llama-server`. See [Real model with llama-server](#real-model-with-llama-server)
  and [the design record](docs/adr/0001-llama-server-adapter.md).

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
| `LLM_LLAMA_SERVER_URL` | `http://127.0.0.1:8080` |
| `LLM_LLAMA_SERVER_API_KEY` | empty (set it if llama-server runs with `--api-key`) |
| `LLM_LLAMA_SERVER_CONNECT_TIMEOUT_S` | `2` (also the `/readyz` probe timeout) |

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

## Real model with llama-server

Install the adapter's one extra dependency (`httpx`):

```sh
.venv/bin/pip install -e ".[dev,llama]"          # Linux / macOS
.venv\Scripts\python -m pip install -e ".[dev,llama]"   # Windows
```

**1. Get llama-server.** Download a CPU build from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases): on Windows
`llama-<tag>-bin-win-cpu-x64.zip` (about 25 MB; unzip anywhere), on Linux
`llama-<tag>-bin-ubuntu-x64.tar.gz` (needs glibc 2.38+, otherwise build from
source with CMake).

**2. Get a GGUF model.** For a smoke test, the 19 MB test model used in
`benchmarks/RESULTS.md`:
<https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf>
(it only writes toy stories). For useful output on a 16 GB machine without a
GPU, pick a Q4_K_M quantization of a 0.5–4B-parameter instruct model; such files
are roughly 0.4–2.5 GB, so check the size on the model page before downloading.
See the sizing notes in the design record.

**3. Start llama-server**, bound to localhost, with an explicit slot count:

```powershell
# Windows
.\llama-server.exe -m C:\models\stories15M-q4_0.gguf --host 127.0.0.1 --port 8080 --alias local-model -np 1 -t 4
```

```sh
# Linux / macOS
./llama-server -m ~/models/stories15M-q4_0.gguf --host 127.0.0.1 --port 8080 --alias local-model -np 1 -t 4
```

`-np` is the number of parallel slots, `-t` the number of threads (use your
physical core count), `-c` the context size if you want to cap it.

**4. Start the service** pointing at it. `LLM_MAX_CONCURRENCY` must equal `-np`:

```powershell
# Windows
$env:LLM_BACKEND = "llama_server"
$env:LLM_MODEL_ID = "local-model"
$env:LLM_MAX_CONCURRENCY = "1"
$env:LLM_REQUEST_TIMEOUT_S = "120"
.venv\Scripts\uvicorn llm_serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

```sh
# Linux / macOS
LLM_BACKEND=llama_server LLM_MODEL_ID=local-model LLM_MAX_CONCURRENCY=1 LLM_REQUEST_TIMEOUT_S=120 \
  .venv/bin/uvicorn llm_serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

**5. Check it.** `/readyz` turns 200 once llama-server has loaded the model:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/readyz
Invoke-RestMethod http://127.0.0.1:8000/v1/completions -Method Post -ContentType 'application/json' `
  -Body '{"model":"local-model","prompt":"Once upon a time","max_tokens":32}'
```

The two processes can start in either order, and llama-server can be restarted
without restarting the service. With this backend, additional failures apply:

| Situation | Response |
| --- | --- |
| llama-server down or still loading | `/readyz` 503; completions 503 `model backend unavailable` |
| llama-server crashes mid-generation, or returns something unusable | 502 `model backend error` |
| Prompt does not fit the model's context | 400 with llama-server's message |
| Deadline or client disconnect | the upstream connection is closed, so llama-server stops generating |

Do not expose llama-server itself to the network: it has none of this service's limits.

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

Measured results, with hardware and workload details, are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

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

Use `--unique-prompts` to prepend a distinct request number to every prompt,
including warmup, when measuring prompt processing without llama-server's prompt-prefix cache reuse.

A short summary always prints; `--output` optionally saves JSON with configuration
(prompt length only), environment, successful-request latency percentiles, aggregate
token throughput and mean per-request token speed. RAM is reported in GiB rounded
to one decimal. With no successes, latency and mean per-request token speed are
null and the exit code is 1; otherwise it is 0.
