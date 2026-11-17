# Measured benchmark results

Two sets of results: [a realistic model](#milestone-3-realistic-model-qwen25-15b-instruct)
(milestone 3) and [the toy model](#environment-2026-11-16) used to build the harness (milestone 2).

## Milestone 3: realistic model (Qwen2.5-1.5B-Instruct)

**No Windows laptop has been measured yet.** Cloud agents cannot run commands
on a Windows machine, so the laptop row below is empty until someone runs
`scripts/validate_llama.py` there (see [docs/windows-validation.md](../docs/windows-validation.md)).
The two measured rows are cloud machines and say so. Do not read them as laptop
numbers: per-core speed, core count and memory bandwidth all differ.

Each row is one run of `python scripts/validate_llama.py --preset qwen2.5-1.5b`.
The full reports (hardware, commands, every check) and raw harness output are in
`benchmarks/results/qwen2.5-1.5b/<machine>/`.

Model: `qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117.3 MB, SHA-256 `6a1a2eb6…9407e`,
from `Qwen/Qwen2.5-1.5B-Instruct-GGUF` at revision `91cad511…ead9`.
llama.cpp `b12443` (commit 301db63). One slot, 4096-token context, 64 completion
tokens per request, `temperature=0`, 20 requests after 2 warm-up, concurrency 1.

| Machine | Windows laptop, 16 GB | GitHub-hosted Windows runner | Cloud sandbox VM |
|---|---|---|---|
| Measured | **not yet** | 2026-11-17 | 2026-11-16 |
| OS | | Windows Server 2025 (10.0.26100), x64 | Linux 7.0 (Debian 12), arm64 |
| CPU | | AMD EPYC 9V45, 2 cores / 4 threads | Neoverse-V2, 4 vCPUs |
| RAM | | 16 GiB | 8 GiB limit |
| llama-server | | official `win-cpu-x64` build | built from source |
| Threads (`-t`) | | 2 | 4 |
| Model load, first launch (s) | | 0.98 | 1.19 |
| Model load, second launch (s) | | 0.95 | 1.18 |
| llama-server memory after load, resident / private (MB) | | 1,711 / 897 | 1,992 / 1,103 |
| llama-server peak resident memory (MB) | | 1,761 | 2,137 |
| llama-server private memory after benchmarks (MB) | | 1,036 | 1,247 |
| Gateway peak resident memory (MB) | | 54 | 53 |
| Direct, same prompt: p50 / p95 / p99 (s) | | 1.823 / 1.930 / 1.953 | 1.276 / 1.294 / 1.304 |
| Gateway, same prompt: p50 / p95 / p99 (s) | | 1.820 / 1.952 / 1.988 | 1.281 / 1.330 / 1.347 |
| Direct, different prompts: p50 / p95 / p99 (s) | | 2.048 / 2.141 / 2.237 | 1.488 / 1.518 / 1.522 |
| Gateway, different prompts: p50 / p95 / p99 (s) | | 2.011 / 2.105 / 2.111 | 1.489 / 1.502 / 1.559 |
| Throughput, same prompt, direct / gateway (tokens/s) | | 34.9 / 34.8 | 50.1 / 49.6 |
| Throughput, different prompts, direct / gateway (tokens/s) | | 31.1 / 31.7 | 42.9 / 42.9 |
| Throughput, same prompt, direct / gateway (requests/s) | | 0.545 / 0.544 | 0.783 / 0.775 |
| Client disconnect: slot free again after (s) | | 0.27 (abandoned request needed about 15 s) | 1.17 (needed about 10 s) |
| llama-server stopped: `/readyz` and completions fail after (s) | | 2.00 | 0.002 |
| All checks | | pass | pass |

Notes on reading this:

- Both model-load figures on each machine were taken with the file already in
  the operating system's cache (it had just been downloaded or read), so they do
  not show a cold load from disk. On a laptop the first launch after a reboot
  will be slower.
- "Resident" memory includes the parts of the memory-mapped model file that have
  been touched; "private" does not. Private memory grows during the run because
  llama-server caches earlier prompts in RAM.
- With 20 requests, p99 is nearly the slowest request.
- The gateway's cost is within run-to-run noise for a model of this size: its
  p50 differs from the direct p50 by between -37 ms and +6 ms on latencies of
  1.3–2.0 s.
- Time to first token is not measured (the API is non-streaming).

### Findings from this validation

1. **Thread count.** Left to its default in the 4-CPU sandbox, llama-server
   started 64 threads (the host's core count) and the toy model ran at
   3.5 tokens/s instead of about 2,700. Always pass `-t`; the script does.
2. **Prompt cache across runs.** llama-server keeps earlier prompts in RAM. A
   second benchmark run that reused the first run's "different" prompts
   processed 1 prompt token per request instead of 37–40. `bench.py
   --unique-prompts` now adds a per-run id.
3. **Memory growth.** Private memory rose by about 3 MB per distinct ~110-token
   prompt (1,177 MB after 22, 1,247 MB after 44, sandbox). The cache may grow to
   8 GB by default (`--cache-ram`).
4. **Outage detection on Windows** takes the connect timeout (2 s by default);
   on Linux it is immediate.
5. **Client disconnect** frees the slot on Windows as well as Linux.

Every number here was measured with `benchmarks/bench.py`; the raw JSON for each
row is in `benchmarks/results/`. Nothing is estimated or extrapolated.

**Read this first.** The model is llama.cpp's 15M-parameter `stories15M` test
model (19 MB). It is hundreds of times smaller than a usable model, so these
tokens-per-second figures say nothing about real models or about a Windows
laptop. They show that the harness works and what the serving layer costs.
Re-run the same commands on the target machine with the target model.

## Environment (2026-11-16)

- Cloud sandbox VM, aarch64, CPU Neoverse-V2, limited to 4 vCPUs and 8 GiB by cgroup
  (the harness records the host's 247.2 GiB in `ram_total_gb`; the cgroup limit is the real one)
- Linux-7.0.0-1017-aws-aarch64-with-glibc2.36, Python 3.11.2
- llama.cpp `b12443` (commit 301db63), built from source, CPU only, `-DCMAKE_BUILD_TYPE=Release`
- Model: `ggml-org/models` `tinyllamas/stories15M-q4_0.gguf`
- Service: one uvicorn worker, `--no-access-log`; load generator on the same machine
- Shared, virtualised hardware: expect run-to-run noise. Three runs are shown where it matters.

## Workload

Closed loop (each worker sends its next request as soon as the last one
returns), non-streaming `POST /v1/completions`, prompt `Once upon a time`
(5 tokens), `max_tokens=64`, `temperature=0`. Warm-up requests are excluded.
The prompt is identical every time, so llama-server's prompt cache is warm.

## Results

| Run | Concurrency | OK | Requests/s | p50 ms | p95 ms | p99 ms | Completion tokens/s (total) | Tokens/s per request (mean) |
|---|---|---|---|---|---|---|---|---|
| `echo-c1` | 1 | 2000/2000 | 1044.4 | 0.9 | 1.1 | 1.4 | 66844 | 68728 |
| `echo-c4` | 4 | 2000/2000 | 1202.7 | 3.3 | 3.6 | 4.0 | 76972 | 19436 |
| `llama-direct-run1` | 1 | 200/200 | 43.1 | 22.9 | 24.1 | 26.9 | 2760 | 2773 |
| `llama-direct-run2` | 1 | 200/200 | 42.6 | 23.4 | 24.7 | 26.0 | 2726 | 2732 |
| `llama-direct-run3` | 1 | 200/200 | 42.4 | 23.2 | 24.8 | 33.3 | 2713 | 2727 |
| `llama-service-run1` | 1 | 200/200 | 39.5 | 25.1 | 26.4 | 28.8 | 2528 | 2538 |
| `llama-service-run2` | 1 | 200/200 | 39.2 | 25.2 | 26.4 | 35.0 | 2508 | 2523 |
| `llama-service-run3` | 1 | 200/200 | 40.4 | 24.6 | 25.5 | 27.3 | 2587 | 2595 |
| `llama-service-np4-c4-run1` | 4 | 400/400 | 81.4 | 48.7 | 51.5 | 53.0 | 5210 | 1309 |
| `llama-service-np4-c4-run2` | 4 | 400/400 | 80.0 | 49.6 | 51.4 | 68.2 | 5118 | 1285 |
| `llama-service-np4-c4-run3` | 4 | 400/400 | 80.1 | 49.3 | 52.3 | 65.2 | 5129 | 1289 |

Rows: `echo-*` is the echo backend (pure service overhead, 2000 requests).
`llama-direct-*` calls llama-server itself (`-np 1 -t 4`, 200 requests).
`llama-service-*` goes through this service with `LLM_MAX_CONCURRENCY=1`
(200 requests). `llama-service-np4-c4-*` uses `-np 4 -t 4 -c 512` and
`LLM_MAX_CONCURRENCY=4` (400 requests).

## What the measurements support

- **Service overhead.** With the echo backend a request takes about 1 ms (p50)
  at concurrency 1. Through the adapter, p50 was 24.6–25.2 ms against
  22.9–23.4 ms direct, so the service plus the extra HTTP hop added roughly
  1.5–2.5 ms per request in these runs. For a real model that generates for
  seconds, that is negligible; for this toy model it is about 8%.
- **Parallel slots.** Four slots on four threads gave about 1.9–2.0x the total
  token throughput of one slot (5.1–5.2k against 2.5–2.6k tokens/s) while each
  request got about twice as slow (p50 ≈ 49 ms against ≈ 25 ms). Slots share the
  same cores: more slots trade per-request speed for total throughput.
- No request failed in any run.

## What they do not show

- Time to first token (the API is non-streaming so far).
- Behaviour with a realistic model, long prompts, a cold prompt cache, or memory pressure.
- Open-loop behaviour (arrivals independent of completions), where queueing and 503s appear.
- Windows performance.

## Reproduce

```sh
# 1. llama-server (see README for where to get it)
llama-server -m stories15M-q4_0.gguf --host 127.0.0.1 --port 8090 -c 128 -np 1 -t 4 --alias stories15m
# 2. service
LLM_BACKEND=llama_server LLM_MODEL_ID=stories15m LLM_LLAMA_SERVER_URL=http://127.0.0.1:8090 \
  LLM_MAX_CONCURRENCY=1 .venv/bin/uvicorn llm_serving.app:create_app --factory --port 8000 --no-access-log
# 3. measure: direct, then through the service
.venv/bin/python benchmarks/bench.py --url http://127.0.0.1:8090 --model stories15m --requests 200 --warmup 5 --max-tokens 64 --output direct.json
.venv/bin/python benchmarks/bench.py --url http://127.0.0.1:8000 --model stories15m --requests 200 --warmup 5 --max-tokens 64 --output service.json
```
