# ADR 0001: Real inference through a separately running `llama-server`

Status: accepted (milestone 2) · Date: 2026-11-16

## Context

Milestone 1 defined the serving contract with an echo backend. Milestone 2 needs
real inference on the target machine: **Windows, 16 GB RAM, no dedicated GPU**.
That means GGUF-quantized models on llama.cpp. The question is how the service
talks to llama.cpp.

## Options

| | A. `llama-cpp-python` in-process | B. Service spawns and supervises `llama-server` | C. Adapter to a separately run `llama-server` (chosen) |
|---|---|---|---|
| Windows install | Needs a C++ toolchain or a matching prebuilt wheel; wheels lag upstream | Official prebuilt zip | Official prebuilt zip (`llama-<tag>-bin-win-cpu-x64.zip`, ~25 MB) |
| Blast radius of a native crash or OOM | Takes the API process down | API survives, must restart child | API survives; operator or service manager restarts llama-server |
| Event loop | Inference is a blocking C call: needs a worker thread | Not blocked | Not blocked (plain async HTTP) |
| Cancelling a running generation | Only cooperatively between tokens; a thread cannot be killed | Close the connection | Close the connection |
| Concurrency | Hand-built (one context, locks or batching) | llama-server slots and batching | llama-server slots and batching |
| Extra code we own | Threading, model lifecycle, batching | Process supervision (Windows job objects, orphan cleanup, restart policy) | One HTTP client (~100 lines) |
| Later move to a GPU box or remote host | Rewrite | Rewrite of the supervisor | Change one URL |
| Cost | Tightest coupling | Most moving parts in our code | Two processes to start; one extra localhost hop |

Ollama was also considered. It wraps llama.cpp and is easy to install, but it adds
its own model store and scheduler between us and the slots we need to reason
about, for no benefit here.

## Decision

**Option C.** `LLM_BACKEND=llama_server` selects an adapter that calls a
separately started `llama-server` over its OpenAI-compatible HTTP API
(`POST /v1/completions`, `GET /health`, `GET /props`). `httpx` is an optional
dependency (`pip install "llm_serving[llama]"`); the echo backend still needs
nothing extra.

The service keeps what llama-server does not provide by itself: a fail-fast
concurrency limit, a hard per-request deadline, request-size limits, strict
validation, a stable error envelope and request IDs. llama-server should bind to
`127.0.0.1` and not be exposed directly.

## Semantics

**Startup.** The two processes may start in any order. The service never fails
at startup because llama-server is absent or still loading; it reports that
through readiness instead. Invalid configuration (for example a URL without
`http://`) does fail at startup.

**Liveness and readiness.** `/healthz` is 200 whenever the service process runs.
`/readyz` is 200 only if llama-server's `/health` answers 200 within
`LLM_LLAMA_SERVER_CONNECT_TIMEOUT_S`; otherwise 503 (down, loading the model,
frozen, erroring). Each probe asks llama-server; nothing is cached.

**Concurrency.** `LLM_MAX_CONCURRENCY` generations run at once; the next request
gets 503 immediately. It must not exceed llama-server's `--parallel`: the excess
would queue inside llama-server, invisibly to the client, while its deadline
runs. Set `--parallel` explicitly (its default is "auto"). When llama-server
becomes ready the adapter reads `total_slots` from `/props` and logs a warning
on a mismatch.

**Deadline and cancellation.** `LLM_REQUEST_TIMEOUT_S` bounds the whole upstream
call (there is deliberately no separate read timeout). On deadline (504) or
client disconnect (499 in logs) the upstream request is cancelled, which closes
the connection; llama-server then cancels the task and frees its slot. The
service's own slot is released only after the upstream call has unwound.

**Failure mapping.**

| Upstream condition | Client sees | Notes |
|---|---|---|
| Connection refused / connect timeout | 503 `model backend unavailable` | retryable |
| 503 (model loading) | 503 `model backend unavailable` | retryable |
| 400 (bad parameter, prompt larger than context) | 400 with llama-server's message, capped at 200 chars | client's fault |
| Connection dropped mid-generation, protocol error | 502 `model backend error` | llama-server crashed or was killed |
| 401/403, other statuses, unparseable or incomplete 200 body | 502 `model backend error` | our misconfiguration or an upstream bug; logged, never echoed |
| No answer before the deadline | 504 `generation timed out` | includes a frozen llama-server |

Credentials (`LLM_LLAMA_SERVER_API_KEY`) are sent as a bearer token, excluded
from `repr(Settings)`, and never logged or returned. Proxy environment variables
are ignored so a local upstream is never routed through a corporate proxy.

**API contract.** Unchanged from milestone 1. `usage` and `finish_reason` come
from llama-server. `model` in the request must still equal `LLM_MODEL_ID`; it is
not forwarded (llama-server serves one model).

## Verified against a real llama-server

Build `b12443` (commit 301db63, built from source on Linux arm64) with the 19 MB
`stories15M-q4_0.gguf` test model, through this service:

- `/health`: 503 `Loading model`, then 200. `/readyz` followed it.
- Service started first with llama-server down: `/healthz` 200, `/readyz` 503,
  completions 503; all recovered when llama-server came up, without a restart.
- 2 s deadline on a generation that would take over a minute: 504 at 2.00 s;
  llama-server logged `cancel task`; the next request was served immediately.
- Client disconnect after 1 s: same cancellation, next request served immediately.
- llama-server killed (SIGKILL) mid-generation: 502 at once, `/readyz` 503,
  recovery after restarting llama-server only.
- llama-server frozen (SIGSTOP) mid-generation: 504 at the deadline, `/readyz`
  503; recovery after SIGCONT.
- Prompt larger than the context: 400 with llama-server's message.
- Slot mismatch (`LLM_MAX_CONCURRENCY=2`, `--parallel 1`, two 4000-token
  requests): the second took 6.3 s instead of 3.3 s. Matched (`1`/`1`): the
  second got 503 in 1 ms.

Not yet verified: the Windows build of llama-server (same code path, not run
here), and behaviour under memory pressure on a 16 GB machine.

## Sizing for 16 GB RAM, CPU only

Rough rule: resident memory ≈ GGUF file size + context memory, and a Q4_K_M
file is very roughly 0.6–0.7 GB per billion parameters. Leave 5–6 GB for
Windows and other applications. That makes 0.5–4B-parameter models comfortable
and 7–8B models possible but tight. Start with `--parallel 1`, `-t` equal to the
number of physical cores, and a modest `-c` (context); check the real per-slot
context in `/props`. Speed depends heavily on the CPU: measure it with
`benchmarks/bench.py` instead of trusting published numbers.

## Remaining production risks (reviewed 2026-11-16)

1. **No authentication or per-client rate limiting** on the service. Fine on
   localhost; anything wider needs a reverse proxy or an auth layer first.
2. **Slow request bodies are not bounded in time.** The deadline covers
   generation, not upload; uvicorn has no body-read timeout. Size is capped, so
   the cost is a held connection, but put a proxy with read timeouts in front,
   or set uvicorn's `--limit-concurrency`.
3. **The concurrency limit is per process.** Run one uvicorn worker, or divide
   `--parallel` between workers; N workers admit N x `LLM_MAX_CONCURRENCY`.
4. **Memory pressure is untested.** On 16 GB a model that is too large makes
   Windows page instead of failing, which looks like a very slow, healthy
   server. Deadlines catch it per request; nothing detects it globally.
5. **No metrics or structured logs.** Failures carry a request ID in plain-text
   logs only; there is no latency, queue or token telemetry.
6. **Readiness probes are not cached** and can take up to two probe timeouts
   (connect, then read) against a frozen llama-server.
7. **Character-based prompt limit.** `LLM_MAX_PROMPT_CHARS` is a coarse guard;
   the real limit is the token context, enforced by llama-server's 400.
8. **Defaults are tuned for the echo backend.** A 30 s deadline and 512-token
   cap are too tight for many CPU models; set them per deployment.
9. **Dependencies are unpinned** (lower bounds only) and llama-server's HTTP API
   is not versioned by us: pin both before depending on this.
10. **Windows llama-server not exercised** by tests or CI (mocked tests do run
    on Windows).

## Implementation plan

1. Done: backend protocol (`async is_ready`, `aclose`, `BackendError`), adapter,
   settings, failure mapping, slot-mismatch warning, mocked integration tests
   including one over real TCP.
2. Done: benchmark harness and measured baseline (see `benchmarks/`).
3. Next: streaming (SSE) end to end, which also gives time-to-first-token.
4. Next: `/v1/chat/completions` (needs the model's chat template, which
   llama-server applies).
5. Next: metrics endpoint; forward `X-Request-ID` upstream; optional
   `Retry-After` on 503.
