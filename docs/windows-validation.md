# Validating on a Windows laptop (16 GB RAM, no GPU)

One script does the whole validation on the machine it runs on:
`scripts/validate_llama.py`. It starts a real `llama-server`, puts the FastAPI
gateway in front of it, checks that text is generated through both, benchmarks
llama-server directly and through the gateway, and measures model load time and
memory. It then writes a report containing only what it measured.

Results from other machines (the cloud sandbox, GitHub's Windows runner) are in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md). They are not a substitute
for running this on your own laptop: CPU speed differs a lot between machines.

## Recommended model

**Qwen2.5-1.5B-Instruct, Q4_K_M quantization** (preset `qwen2.5-1.5b`, the default).

| | |
|---|---|
| File | `qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117 MB |
| Source | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` at revision `91cad51170dc346986eccefdc2dd33a9da36ead9` |
| SHA-256 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| License | Apache-2.0, no login or licence click-through |

Why this one: it is at the top of the 0.5B–1.5B range, so it gives the best
answers the range allows; the GGUF is published by the model's authors; the
licence is permissive; and at about 1.1 GB on disk it leaves most of a 16 GB
machine free. Q4_K_M is the usual balance of size and quality for CPU use.

Alternatives: `--preset qwen2.5-0.5b` (491 MB, faster, noticeably weaker
answers) and `--preset stories15m` (19 MB toy model, only proves the plumbing).
Any other GGUF works with `--model C:\path\to\file.gguf`.

The gateway exposes raw text completion, not chat. Qwen expects its chat
format, so the script wraps its prompts in it. If you call the API yourself,
do the same:

```
<|im_start|>user
Your question<|im_end|>
<|im_start|>assistant
```

and pass `"stop": ["<|im_end|>"]`.

## What to run on the laptop

In PowerShell, from the repository folder. Python 3.11 or newer is required
(`winget install Python.Python.3.12`).

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,llama]"
.venv\Scripts\python scripts\validate_llama.py --download --label "Windows laptop, <your CPU>, 16 GB"
```

`--download` allows two downloads into `.cache\` (ignored by Git): the pinned
llama.cpp CPU build for Windows (25 MB) and the model (1,117 MB). Both are
checked against pinned SHA-256 values, and an interrupted download resumes.
Without `--download` the script only tells you what it would fetch.

For numbers you can trust:

- Plug the laptop in and set Windows power mode to "Best performance".
- Close browsers and other heavy applications.
- Do not use the machine while it runs. Expect roughly 10 to 30 minutes,
  depending on the CPU.
- For tighter p95/p99 figures add `--requests 100` (takes about five times longer).

The script prints PASS or FAIL for each check and writes
`benchmarks\results\local\<timestamp>\report.md` and `report.json`.

## Sending the results back

The report contains hardware, model and timing data. It does not contain your
user name, machine name or folder paths. To add it to the repository:

```powershell
git checkout -b windows-laptop-results
mkdir benchmarks\results\qwen2.5-1.5b\windows-laptop
copy benchmarks\results\local\<timestamp>\*.json benchmarks\results\qwen2.5-1.5b\windows-laptop\
copy benchmarks\results\local\<timestamp>\report.md benchmarks\results\qwen2.5-1.5b\windows-laptop\
git add benchmarks\results\qwen2.5-1.5b\windows-laptop
git commit -m "Add Windows laptop validation results"
git push -u origin windows-laptop-results
```

Or paste `report.md` into the pull request.

## Useful options

| Option | Purpose |
|---|---|
| `--threads N` | Threads for llama-server. Default: physical cores. On Intel CPUs with performance and efficiency cores, also try the number of performance cores and compare. |
| `--ctx N` | Context tokens per slot (default 4096). Larger uses more memory. |
| `--parallel N` | Slots in llama-server and matching gateway concurrency (default 1). |
| `--load-runs N` | How many times llama-server is launched to time loading (default 2: first launch, then with a warm file cache). |
| `--llama-port`, `--gateway-port` | Change if 8080 or 8000 are taken. |
| `--llama-server PATH` | Use your own `llama-server.exe` instead of the pinned build. |

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| "Something already answers at http://127.0.0.1:8080" | Another program uses the port. Stop it or pass `--llama-port 8090`. |
| `llama-server exited with code ...` before becoming ready | Open `llama-server.log` in the report folder. A missing `VCRUNTIME140.dll` or `MSVCP140.dll` means the Microsoft Visual C++ Redistributable (x64) is not installed. |
| Windows Security or SmartScreen blocks `llama-server.exe` | The file comes from the official llama.cpp release and its checksum was verified. Allow it, or download it yourself and pass `--llama-server`. |
| A firewall prompt appears | Both servers listen on `127.0.0.1` only. You can deny network access. |
| The first launch is much slower than the second | Normal: the first launch reads 1.1 GB from disk and antivirus may scan it. Both launches are reported. |
| Very low tokens per second | Check "Threads" in the report. Try `--threads` with the physical or performance core count. Make sure the laptop is plugged in. |
| Checksum mismatch | The download was corrupted or altered. The partial file is deleted; run again. |

## What CI covers on Windows

- The normal test suite runs on Windows, including an end-to-end run of this
  script against a fake `llama-server` executable and tests of the Windows
  memory counters.
- The "Windows llama-server validation" workflow runs this script with the real
  Windows `llama-server.exe` on a GitHub-hosted runner: with the toy model on
  pull requests, and with a real model when the pull request has the
  `real-model` label or the workflow is started manually. Those reports say
  "GitHub Actions hosted runner" at the top.
