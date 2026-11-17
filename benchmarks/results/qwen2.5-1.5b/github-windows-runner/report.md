# llama-server validation report

**Measured on: GitHub Actions windows-latest runner** (GitHub Actions hosted runner: cloud hardware, not a user machine)
Date (UTC): 2026-11-17T02:29:08+00:00  
Result: PASS

Every number below was measured by `scripts/validate_llama.py` during this run.

## Machine

| | |
|---|---|
| OS | Windows-2025Server-10.0.26100-SP0 |
| CPU | AMD EPYC 9V45 96-Core Processor (AMD64) |
| Cores | 2 physical, 4 logical |
| RAM | 16.0 GiB |
| Python | 3.12.10 |

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
| Version | version: 1.0.0-dev (build 12443, commit 301db6337); built with Clang 20.1.8 for Windows x86_64 |
| Build info | b12443-301db6337 |
| Slots | 1 |
| Context per slot | 4096 tokens |
| Threads | 2 (`-t 2`) |
| Command | `llama-server.exe -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 --alias qwen2.5-1.5b -np 1 -c 4096 -t 2` |

Gateway settings: `LLM_BACKEND=llama_server`, `LLM_MODEL_ID=qwen2.5-1.5b`, `LLM_LLAMA_SERVER_URL=http://127.0.0.1:8080`, `LLM_MAX_CONCURRENCY=1`, `LLM_REQUEST_TIMEOUT_S=300`, `LLM_MAX_TOKENS_LIMIT=512`

## Checks

| Check | Result | Detail |
|---|---|---|
| gateway lists the model | pass |  |
| llama-server generates text directly | pass |  |
| gateway generates text through llama-server | pass | 0.233 s |
| unknown model is rejected with 404 | pass |  |
| client disconnect frees the slot | pass | next request answered 200 after 0.27 s; the abandoned request would have run for about 15 s |
| benchmark direct-same-prompt | pass | 20/20 succeeded in 36.713s; 0.54 requests/s; 34.87 completion tokens/s; statuses: {'200': 20} |
| benchmark gateway-same-prompt | pass | 20/20 succeeded in 36.764s; 0.54 requests/s; 34.82 completion tokens/s; statuses: {'200': 20} |
| benchmark direct-unique-prompt | pass | 20/20 succeeded in 41.137s; 0.49 requests/s; 31.12 completion tokens/s; statuses: {'200': 20} |
| benchmark gateway-unique-prompt | pass | 20/20 succeeded in 40.363s; 0.50 requests/s; 31.71 completion tokens/s; statuses: {'200': 20} |
| readiness turns 503 when llama-server is down | pass | 2.001 s |
| completions return 503 when llama-server is down | pass | 2.002 s |
| gateway stays alive without llama-server | pass |  |

## Model load time

Ready = process start until `/health` returns 200. Loaded = llama-server's own log timestamp.
The first launch may read the model from disk; later launches use the OS file cache.

| Launch | Ready (s) | Loaded per log (s) | Memory after load (MB) |
|---|---|---|---|
| 1 | 0.977 | 0.946 | 1711.0 |
| 2 | 0.952 | 0.931 | 1710.7 |

## Memory

Resident = working set (includes the memory-mapped model). Private = not backed by the model file.

| Process | Resident now (MB) | Peak resident (MB) | Private (MB) |
|---|---|---|---|
| llama-server | 1209.4 | 1760.9 | 1035.7 |
| gateway | 29.9 | 53.8 | 42.0 |

## Latency and throughput

Closed loop, concurrency 1, 20 requests after 2 warm-up, `max_tokens=64`, `temperature=0`, non-streaming.
"Same prompt" lets llama-server reuse its prompt cache; "different prompt" forces it to process the whole prompt every time, which is closer to real traffic.

| Path | Prompts | OK | Requests/s | p50 (s) | p95 (s) | p99 (s) | Tokens/s | Prompt tokens |
|---|---|---|---|---|---|---|---|---|
| llama-server direct | same | 20/20 | 0.545 | 1.823 | 1.930 | 1.953 | 34.9 | 34 |
| through gateway | same | 20/20 | 0.544 | 1.820 | 1.952 | 1.988 | 34.8 | 34 |
| llama-server direct | different | 20/20 | 0.486 | 2.048 | 2.141 | 2.237 | 31.1 | 49 |
| through gateway | different | 20/20 | 0.496 | 2.011 | 2.105 | 2.111 | 31.7 | 48 |

Tokens/s is completion tokens divided by wall time, so it includes prompt processing.

- Gateway overhead at p50, same prompts: -3.2 ms (-0.2%).
- Gateway overhead at p50, different prompts: -36.7 ms (-1.8%).

With 20 requests, p99 is close to the slowest request; use `--requests 100` or more for meaningful tails. Differences of a few percent between single runs are within noise.

## Outage behaviour

With llama-server stopped, `/readyz` answered in 2.001 s and a completion request in 2.002 s.

## Sample generation through the gateway

Question: What is the capital of France? Answer in one short sentence.

```
The capital of France is Paris.
```

8 tokens in 0.233 s, finish reason `stop`.

## Reproduce

```
python scripts/validate_llama.py --preset qwen2.5-1.5b --download --label "GitHub Actions windows-latest runner" --out validation-report
```

Benchmark commands run by the script:

```
python benchmarks/bench.py --url http://127.0.0.1:8080 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "GitHub Actions windows-latest runner | llama-server direct" --output direct-same-prompt.json
python benchmarks/bench.py --url http://127.0.0.1:8000 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "GitHub Actions windows-latest runner | through gateway" --output gateway-same-prompt.json
python benchmarks/bench.py --url http://127.0.0.1:8080 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "GitHub Actions windows-latest runner | llama-server direct" --output direct-unique-prompt.json --unique-prompts
python benchmarks/bench.py --url http://127.0.0.1:8000 --model qwen2.5-1.5b --requests 20 --concurrency 1 --warmup 2 --max-tokens 64 --temperature 0 --timeout 300 --prompt-file bench-prompt.txt --label "GitHub Actions windows-latest runner | through gateway" --output gateway-unique-prompt.json --unique-prompts
```
