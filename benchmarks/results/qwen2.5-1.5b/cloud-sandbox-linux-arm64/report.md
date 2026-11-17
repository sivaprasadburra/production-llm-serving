# llama-server validation report

**Measured on: cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop)**
Date (UTC): 2026-11-16T23:30:27+00:00  
Result: PASS

Every number below was measured by `scripts/validate_llama.py` during this run.

## Machine

| | |
|---|---|
| OS | Linux-7.0.0-1017-aws-aarch64-with-glibc2.36 |
| CPU | Neoverse-V2 (aarch64) |
| Cores | unknown physical, 4 logical |
| RAM | 247.2 GiB (this process is limited to 8.0 GiB) |
| Python | 3.11.2 |

## Model

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct Q4_K_M |
| File | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1117.3 MB) |
| SHA-256 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Source | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf |
| License | apache-2.0 |

## llama-server

| | |
|---|---|
| Version | version: 1.0.0-dev (build 1, commit 301db63); built with GNU 12.2.0 for Linux aarch64 |
| Build info | b1-301db63 |
| Slots | 1 |
| Context per slot | 4096 tokens |
| Threads | 4 (`-t 4`) |
| Command | `llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 --alias qwen2.5-1.5b -np 1 -c 4096 -t 4` |

Gateway settings: `LLM_BACKEND=llama_server`, `LLM_MODEL_ID=qwen2.5-1.5b`, `LLM_LLAMA_SERVER_URL=http://127.0.0.1:8080`, `LLM_MAX_CONCURRENCY=1`, `LLM_REQUEST_TIMEOUT_S=300`, `LLM_MAX_TOKENS_LIMIT=512`

## Checks

| Check | Result | Detail |
|---|---|---|
| gateway lists the model | pass |  |
| llama-server generates text directly | pass |  |
| gateway generates text through llama-server | pass | 0.162 s |
| unknown model is rejected with 404 | pass |  |
| client disconnect frees the slot | pass | next request answered 200 after 1.174 s; the abandoned request would have run for about 10 s |
| benchmark direct-same-prompt | pass | 20/20 succeeded in 25.553s; 0.78 requests/s; 50.09 completion tokens/s; statuses: {'200': 20} |
| benchmark gateway-same-prompt | pass | 20/20 succeeded in 25.819s; 0.77 requests/s; 49.58 completion tokens/s; statuses: {'200': 20} |
| benchmark direct-unique-prompt | pass | 20/20 succeeded in 29.860s; 0.67 requests/s; 42.87 completion tokens/s; statuses: {'200': 20} |
| benchmark gateway-unique-prompt | pass | 20/20 succeeded in 29.853s; 0.67 requests/s; 42.88 completion tokens/s; statuses: {'200': 20} |
| readiness turns 503 when llama-server is down | pass | 0.002 s |
| completions return 503 when llama-server is down | pass | 0.002 s |
| gateway stays alive without llama-server | pass |  |

## Model load time

Ready = process start until `/health` returns 200. Loaded = llama-server's own log timestamp.
The first launch may read the model from disk; later launches use the OS file cache.

| Launch | Ready (s) | Loaded per log (s) | Memory after load (MB) |
|---|---|---|---|
| 1 | 1.193 | 1.174 | 1992.0 |
| 2 | 1.182 | 1.165 | 1992.0 |

## Memory

Resident = working set (includes the memory-mapped model). Private = not backed by the model file.

| Process | Resident now (MB) | Peak resident (MB) | Private (MB) |
|---|---|---|---|
| llama-server | 2136.5 | 2136.5 | 1247.0 |
| gateway | 52.7 | 52.7 | 37.8 |

## Latency and throughput

Closed loop, concurrency 1, 20 requests after 2 warm-up, `max_tokens=64`, `temperature=0`, non-streaming.
"Same prompt" lets llama-server reuse its prompt cache; "different prompt" forces it to process the whole prompt every time, which is closer to real traffic.

| Path | Prompts | OK | Requests/s | p50 (s) | p95 (s) | p99 (s) | Tokens/s | Prompt tokens |
|---|---|---|---|---|---|---|---|---|
| llama-server direct | same | 20/20 | 0.783 | 1.276 | 1.294 | 1.304 | 50.1 | 34 |
| through gateway | same | 20/20 | 0.775 | 1.281 | 1.330 | 1.347 | 49.6 | 34 |
| llama-server direct | different | 20/20 | 0.670 | 1.488 | 1.518 | 1.522 | 42.9 | 48 |
| through gateway | different | 20/20 | 0.670 | 1.489 | 1.502 | 1.559 | 42.9 | 48 |

Tokens/s is completion tokens divided by wall time, so it includes prompt processing.

- Gateway overhead at p50, same prompts: +5.7 ms (+0.4%).
- Gateway overhead at p50, different prompts: +0.9 ms (+0.1%).

With 20 requests, p99 is close to the slowest request; use `--requests 100` or more for meaningful tails. Differences of a few percent between single runs are within noise.

## Outage behaviour

With llama-server stopped, `/readyz` answered in 0.002 s and a completion request in 0.002 s.

## Sample generation through the gateway

Question: What is the capital of France? Answer in one short sentence.

```
The capital of France is Paris.
```

8 tokens in 0.162 s, finish reason `stop`.

## Reproduce

```
python scripts/validate_llama.py --preset qwen2.5-1.5b --llama-server llama-server --label "cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop)" --out val-qwen2
```

Benchmark commands run by the script:

```
python benchmarks/bench.py --url http://127.0.0.1:8080 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop) | llama-server direct" --output direct-same-prompt.json
python benchmarks/bench.py --url http://127.0.0.1:8000 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop) | through gateway" --output gateway-same-prompt.json
python benchmarks/bench.py --url http://127.0.0.1:8080 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop) | llama-server direct" --output direct-unique-prompt.json --unique-prompts
python benchmarks/bench.py --url http://127.0.0.1:8000 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "cloud sandbox VM, 4 vCPU Neoverse-V2, Linux (NOT a Windows laptop) | through gateway" --output gateway-unique-prompt.json --unique-prompts
```
