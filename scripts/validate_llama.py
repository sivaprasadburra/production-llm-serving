"""Validate llama-server and the gateway on THIS machine, and measure them.

Starts a real llama-server, starts the FastAPI gateway in front of it, checks
that text is generated through both, runs benchmarks/bench.py directly and
through the gateway, and records model load time and memory use. Everything
written to the report was measured by this run on this machine.

    python scripts/validate_llama.py --download            # recommended model
    python scripts/validate_llama.py --preset stories15m --download   # 19 MB smoke test

Windows, Linux and macOS. Needs: pip install -e ".[dev,llama]"
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "bench.py"

LLAMA_TAG = "b12443"
LLAMA_ASSETS = {  # (system, machine) -> (asset name, sha256)
    ("Windows", "AMD64"): ("llama-b12443-bin-win-cpu-x64.zip",
                           "e4e0b4fb23cc8f2e6ac22934da0389f685295a9dc02b909cc6dd2377887808ad"),
    ("Windows", "ARM64"): ("llama-b12443-bin-win-cpu-arm64.zip",
                           "c9e59c10795a61c63db325dc0d1817a205fe981fac00e20d50710183e544a485"),
    ("Linux", "x86_64"): ("llama-b12443-bin-ubuntu-x64.tar.gz",
                          "0d2aae32fd7a1fdb22f31ae376445f9cfcfeb3f09fe6dba226f5282e31e4c5a0"),
    ("Linux", "aarch64"): ("llama-b12443-bin-ubuntu-arm64.tar.gz",
                           "60ebd79559d275e5a7989740030438bf46b684b8f187297eeed2061ac6cae4e1"),
    ("Darwin", "arm64"): ("llama-b12443-bin-macos-arm64.tar.gz",
                          "26969784c53c85878444e27bc58dc8ff7036a645a32b73e78bb53306db8eb154"),
}

CHATML = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
PRESETS = {  # exact, pinned model files
    "qwen2.5-1.5b": dict(
        name="Qwen2.5-1.5B-Instruct Q4_K_M", repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
        file="qwen2.5-1.5b-instruct-q4_k_m.gguf", size=1117320736, license="apache-2.0",
        sha256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        template=CHATML, stop=["<|im_end|>"]),
    "qwen2.5-0.5b": dict(
        name="Qwen2.5-0.5B-Instruct Q4_K_M", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision="9217f5db79a29953eb74d5343926648285ec7e67",
        file="qwen2.5-0.5b-instruct-q4_k_m.gguf", size=491400032, license="apache-2.0",
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        template=CHATML, stop=["<|im_end|>"]),
    "stories15m": dict(  # llama.cpp's toy test model: plumbing check only
        name="stories15M Q4_0 (toy test model)", repo="ggml-org/models",
        revision="499bc8821c6b12b4e53c5bffcb21ec206f212d81",
        file="tinyllamas/stories15M-q4_0.gguf", size=19077344, license="mit",
        sha256="66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
        template="{question}", stop=None),
}
QUESTION = "Explain in detail how a CPU executes a program, step by step."
SAMPLE_QUESTION = "What is the capital of France? Answer in one short sentence."


def say(message):
    print(message, flush=True)


# ---------------------------------------------------------------- downloads

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest, sha256, size=None):
    """Download url to dest (resuming a partial file) and verify its SHA-256."""
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    # trust_env stays on here: reaching the internet may need the system proxy.
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as response:
        if response.status_code == 416:  # nothing left to fetch
            pass
        else:
            response.raise_for_status()
            resumed = response.status_code == 206
            total = (size or 0) or int(response.headers.get("content-length", 0)) + (have if resumed else 0)
            done, shown = (have if resumed else 0), -1
            with open(part, "ab" if resumed else "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    percent = done * 100 // total if total else 0
                    if percent // 10 != shown:
                        shown = percent // 10
                        say(f"  {dest.name}: {done / 1e6:.0f} MB" + (f" ({percent}%)" if total else ""))
    actual = sha256_of(part)
    if actual != sha256:
        part.unlink()
        raise SystemExit(f"Checksum mismatch for {dest.name}: expected {sha256}, got {actual}. "
                         "The partial file was deleted; run again.")
    part.replace(dest)
    return dest


def find_server(folder):
    for name in ("llama-server.exe", "llama-server"):
        for path in sorted(Path(folder).rglob(name)):
            if path.is_file():
                return path
    return None


def get_llama_server(cache, allow_download):
    folder = Path(cache) / f"llama-{LLAMA_TAG}"
    found = find_server(folder) if folder.exists() else None
    if found:
        return found
    key = (platform.system(), platform.machine())
    if key not in LLAMA_ASSETS:
        raise SystemExit(f"No pinned llama.cpp build for {key}. Pass --llama-server PATH.")
    asset, digest = LLAMA_ASSETS[key]
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}"
    if not allow_download:
        raise SystemExit(f"llama-server not found. Re-run with --download to fetch {url} "
                         "(about 25 MB), or pass --llama-server PATH.")
    say(f"Downloading llama.cpp {LLAMA_TAG}: {asset}")
    archive = download(url, Path(cache) / asset, digest)
    folder.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(folder)
    else:
        with tarfile.open(archive) as bundle:
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(folder, filter="data")
            else:  # older Python: the archive's checksum was verified above
                bundle.extractall(folder)
    found = find_server(folder)
    if not found:
        raise SystemExit(f"llama-server not found inside {asset}")
    return found


def get_model(preset, cache, allow_download):
    path = Path(cache) / "models" / Path(preset["file"]).name
    if path.exists() and sha256_of(path) == preset["sha256"]:
        return path
    url = f"https://huggingface.co/{preset['repo']}/resolve/{preset['revision']}/{preset['file']}"
    if not allow_download:
        raise SystemExit(f"Model not found. Re-run with --download to fetch {url} "
                         f"({preset['size'] / 1e6:.0f} MB), or pass --model PATH.")
    say(f"Downloading model {preset['name']} ({preset['size'] / 1e6:.0f} MB)")
    return download(url, path, preset["sha256"], preset["size"])


# ---------------------------------------------------------------- measuring

def process_memory(pid):
    """Resident memory of a process in MB: current, peak, and private (not file-backed).

    llama.cpp memory-maps the model file, so `rss_mb` includes the model pages
    touched so far while `private_mb` does not.
    """
    mb = 1024 * 1024
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    )
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE  # 64-bit handle, not a C int
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED | VM_READ
            if not handle:
                return None
            try:
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
            finally:
                kernel32.CloseHandle(handle)
            return {"rss_mb": round(counters.WorkingSetSize / mb, 1),
                    "peak_rss_mb": round(counters.PeakWorkingSetSize / mb, 1),
                    "private_mb": round(counters.PagefileUsage / mb, 1)}
        if sys.platform.startswith("linux"):
            fields = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if value.strip().endswith("kB"):
                    fields[key] = int(value.split()[0]) / 1024
            return {"rss_mb": round(fields["VmRSS"], 1), "peak_rss_mb": round(fields["VmHWM"], 1),
                    "private_mb": round(fields["RssAnon"], 1) if "RssAnon" in fields else None}
        rss_kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True))
        return {"rss_mb": round(rss_kb / 1024, 1), "peak_rss_mb": None, "private_mb": None}
    except Exception:
        return None


def log_seconds(log_text, marker):
    """Seconds since llama-server start for the first log line containing marker.

    llama-server prefixes lines with `M.SS.mmm.uuu` (minutes, seconds, ms, us).
    """
    for line in log_text.splitlines():
        match = re.match(r"(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ", line)
        if match and marker in line:
            minutes, seconds, millis, micros = map(int, match.groups())
            return round(minutes * 60 + seconds + millis / 1e3 + micros / 1e6, 3)
    return None


def log_int(log_text, name):
    match = re.search(rf"\b{name} = (\d+)", log_text)
    return int(match.group(1)) if match else None


def wait_for(client, url, process, timeout, what):
    """Poll url until it answers 200. Returns seconds waited."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise SystemExit(f"{what} exited with code {process.returncode} before becoming ready")
        try:
            if client.get(url, timeout=0.25).status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.02)
    raise SystemExit(f"{what} was not ready after {timeout:.0f} s")


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)


def port_is_free(client, url):
    try:
        client.get(url, timeout=0.5)
    except httpx.HTTPError:
        return True
    return False


def timed(call):
    started = time.perf_counter()
    try:
        result = call()
    except httpx.HTTPError as exc:
        result = exc
    return result, round(time.perf_counter() - started, 3)


# ---------------------------------------------------------------- the run

def default_threads(env):
    """Threads for llama-server: physical cores, never more than the CPUs we may use.

    llama-server's own default counts the host's cores. Inside a container or a
    CPU-limited VM that oversubscribes badly (measured: 64 threads on a 4-CPU
    quota ran about 780 times slower), so the count is always passed explicitly.
    """
    limits = [env.get("cpu_physical_cores"), env.get("cpu_count")]
    try:  # Linux cgroup v2 CPU quota, e.g. "400000 100000" = 4 CPUs
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return min([n for n in limits if n] or [1])


def llama_command(server, model, args, alias):
    return [*server, "-m", str(model), "--host", "127.0.0.1", "--port", str(args.llama_port),
            "--alias", alias, "-np", str(args.parallel), "-c", str(args.ctx * args.parallel),
            "-t", str(args.threads)]


def gateway_env(args, alias):
    return {"LLM_BACKEND": "llama_server", "LLM_MODEL_ID": alias,
            "LLM_LLAMA_SERVER_URL": f"http://127.0.0.1:{args.llama_port}",
            "LLM_MAX_CONCURRENCY": str(args.parallel),
            "LLM_REQUEST_TIMEOUT_S": str(args.timeout),
            "LLM_MAX_TOKENS_LIMIT": str(max(512, args.max_tokens))}


def run(args, server, model, model_info, out):
    alias = args.preset if not args.model else "local-model"
    template, stop_words = model_info["template"], model_info["stop"]
    llama_url = f"http://127.0.0.1:{args.llama_port}"
    gateway_url = f"http://127.0.0.1:{args.gateway_port}"
    client = httpx.Client(trust_env=False)
    checks, launches, bench, llama, gateway = [], [], [], None, None
    result = {"checks": checks, "launches": launches, "benchmarks": bench}

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    for url in (llama_url + "/health", gateway_url + "/healthz"):
        if not port_is_free(client, url):
            raise SystemExit(f"Something already answers at {url}. Stop it or choose other ports.")

    try:
        version = subprocess.run([*server, "--version"], capture_output=True, text=True, timeout=60)
        result["llama_server_version"] = (version.stderr + version.stdout).strip().splitlines()[:2]
        command = llama_command(server, model, args, alias)
        result["llama_command"] = command
        log_path = out / "llama-server.log"

        for number in range(1, args.load_runs + 1):
            say(f"Starting llama-server (launch {number} of {args.load_runs})")
            with open(log_path, "w", encoding="utf-8") as log:
                started = time.perf_counter()
                llama = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                wait_for(client, llama_url + "/health", llama, args.load_timeout, "llama-server")
                ready = time.perf_counter() - started
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            launches.append({"launch": number, "ready_s": round(ready, 3),
                             "log_model_loaded_s": log_seconds(log_text, "model loaded"),
                             "memory_after_load": process_memory(llama.pid)})
            say(f"  ready after {ready:.2f} s")
            if number < args.load_runs:
                stop(llama)
        result["n_threads"] = log_int(log_text, "n_threads")
        result["n_ctx_slot"] = log_int(log_text, "n_ctx_slot")
        props = client.get(llama_url + "/props", timeout=10).json()
        result["llama_props"] = {key: props.get(key) for key in ("build_info", "total_slots")}

        say("Starting the gateway")
        env = gateway_env(args, alias)
        result["gateway_env"] = env
        with open(out / "gateway.log", "w", encoding="utf-8") as log:
            gateway = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "llm_serving.app:create_app", "--factory",
                 "--host", "127.0.0.1", "--port", str(args.gateway_port), "--no-access-log"],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}, cwd=ROOT)
            wait_for(client, gateway_url + "/readyz", gateway, 60, "gateway")

        say("Checks")
        models = client.get(gateway_url + "/v1/models", timeout=10).json()
        check("gateway lists the model", [m["id"] for m in models["data"]] == [alias])
        body = {"model": alias, "prompt": template.format(question=SAMPLE_QUESTION),
                "max_tokens": 48, "temperature": 0}
        if stop_words:
            body["stop"] = stop_words
        direct, _ = timed(lambda: client.post(llama_url + "/v1/completions", json=body, timeout=args.timeout))
        ok = isinstance(direct, httpx.Response) and direct.status_code == 200
        check("llama-server generates text directly", ok and direct.json()["choices"][0]["text"].strip())
        through, seconds = timed(lambda: client.post(gateway_url + "/v1/completions", json=body, timeout=args.timeout))
        ok = isinstance(through, httpx.Response) and through.status_code == 200
        text = through.json()["choices"][0]["text"] if ok else ""
        check("gateway generates text through llama-server", ok and text.strip(), f"{seconds} s")
        if ok:
            usage = through.json()["usage"]
            result["sample"] = {"question": SAMPLE_QUESTION, "text": text, "seconds": seconds,
                                "tokens_per_s": round(usage["completion_tokens"] / seconds, 1) if seconds else None,
                                "usage": usage,
                                "finish_reason": through.json()["choices"][0]["finish_reason"]}
            say("  model said: " + text.strip().encode("ascii", "replace").decode()[:200])
        wrong, _ = timed(lambda: client.post(gateway_url + "/v1/completions", timeout=30,
                                             json={**body, "model": "no-such-model"}))
        check("unknown model is rejected with 404", getattr(wrong, "status_code", None) == 404)

        # Client disconnect: abandon a long generation after 1 s, then time how long the
        # (single) slot takes to accept work again. Without cancellation it would stay
        # busy, answering 503, until the abandoned generation finished on its own.
        long_tokens = 512
        long_body = {**body, "prompt": template.format(question=QUESTION), "max_tokens": long_tokens}
        long_body.pop("stop", None)
        speed = result.get("sample", {}).get("tokens_per_s") or 0
        expected = long_tokens / speed if speed else 0  # seconds the long request needs
        abandoned, _ = timed(lambda: client.post(gateway_url + "/v1/completions", json=long_body, timeout=1.0))
        started = time.perf_counter()
        while True:
            after, _ = timed(lambda: client.post(gateway_url + "/v1/completions", json=body, timeout=args.timeout))
            freed = round(time.perf_counter() - started, 3)
            if getattr(after, "status_code", None) != 503 or freed > 10:
                break
            time.sleep(0.05)
        status = getattr(after, "status_code", type(after).__name__)
        if isinstance(abandoned, httpx.Response) or expected < 5:
            check("client disconnect frees the slot", status == 200,
                  "inconclusive: this model finishes the long request too quickly to tell")
        else:
            result["disconnect"] = {"slot_free_after_s": freed, "abandoned_request_needed_s": round(expected, 1)}
            check("client disconnect frees the slot", status == 200 and freed < expected / 2,
                  f"next request answered {status} after {freed} s; the abandoned request "
                  f"would have run for about {expected:.0f} s")

        say("Benchmarks (benchmarks/bench.py)")
        prompt_file = out / "bench-prompt.txt"
        prompt_file.write_text(template.format(question=QUESTION), encoding="utf-8")
        for unique in (False, True):
            for target, url in (("llama-server direct", llama_url), ("through gateway", gateway_url)):
                name = f"{'direct' if url == llama_url else 'gateway'}-{'unique' if unique else 'same'}-prompt"
                bench_command = [
                    sys.executable, str(BENCH), "--url", url, "--model", alias,
                    "--requests", str(args.requests), "--concurrency", str(args.parallel),
                    "--warmup", str(args.warmup), "--max-tokens", str(args.max_tokens),
                    "--temperature", "0", "--timeout", str(args.timeout),
                    # Relative names (cwd is the report folder): no local paths in the results.
                    "--prompt-file", prompt_file.name, "--label", f"{args.label} | {target}",
                    "--output", f"{name}.json", *(["--unique-prompts"] if unique else [])]
                say(f"  {target}, {'a different prompt each request' if unique else 'same prompt each request'}")
                finished = subprocess.run(bench_command, capture_output=True, text=True, cwd=out)
                output_file = out / f"{name}.json"
                data = json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists() else None
                bench.append({"name": name, "target": target, "unique_prompts": unique,
                              "command": bench_command, "exit_code": finished.returncode, "result": data})
                check(f"benchmark {name}", finished.returncode == 0 and data and data["ok"] == data["requests"],
                      (finished.stdout.strip().splitlines() or [finished.stderr.strip()[-200:]])[0])

        result["llama_memory_after_benchmarks"] = process_memory(llama.pid)
        result["gateway_memory_after_benchmarks"] = process_memory(gateway.pid)

        say("Stopping llama-server to check how the gateway reports an outage")
        stop(llama)
        ready, ready_s = timed(lambda: client.get(gateway_url + "/readyz", timeout=30))
        down, down_s = timed(lambda: client.post(gateway_url + "/v1/completions", json=body, timeout=30))
        result["outage"] = {"readyz_seconds": ready_s, "completion_seconds": down_s}
        check("readiness turns 503 when llama-server is down",
              getattr(ready, "status_code", None) == 503, f"{ready_s} s")
        check("completions return 503 when llama-server is down",
              getattr(down, "status_code", None) == 503, f"{down_s} s")
        check("gateway stays alive without llama-server",
              client.get(gateway_url + "/healthz", timeout=10).status_code == 200)
    finally:
        stop(gateway)
        stop(llama)
        client.close()
    return result


# ---------------------------------------------------------------- report

def environment():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.environment()


def show(command):
    return " ".join(f'"{part}"' if " " in part or "\n" in part else part for part in map(str, command))


def render(report):
    env, model, result = report["environment"], report["model"], report["result"]
    lines = [
        "# llama-server validation report", "",
        f"**Measured on: {report['label']}**"
        + (" (GitHub Actions hosted runner: cloud hardware, not a user machine)" if report["ci"] else ""),
        f"Date (UTC): {report['timestamp_utc']}  ", f"Result: {'PASS' if report['passed'] else 'FAIL'}", "",
        "Every number below was measured by `scripts/validate_llama.py` during this run.", "",
        "## Machine", "", "| | |", "|---|---|",
        f"| OS | {env['platform']} |", f"| CPU | {env['cpu_model']} ({env['machine']}) |",
        f"| Cores | {env.get('cpu_physical_cores') or 'unknown'} physical, {env['cpu_count']} logical |",
        f"| RAM | {env['ram_total_gb']} GiB"
        + (f" (this process is limited to {report['memory_limit_gb']} GiB)" if report.get("memory_limit_gb") else "") + " |",
        f"| Python | {env['python']} |", "",
        "## Model", "", "| | |", "|---|---|",
        f"| Model | {model['name']} |", f"| File | `{model['file']}` ({model['size'] / 1e6:.1f} MB) |",
        f"| SHA-256 | `{model['sha256']}` |",
        f"| Source | {model.get('source', 'local file supplied with --model')} |",
        f"| License | {model.get('license', 'unknown')} |", "",
        "## llama-server", "", "| | |", "|---|---|",
        f"| Version | {'; '.join(result.get('llama_server_version') or ['unknown'])} |",
        f"| Build info | {(result.get('llama_props') or {}).get('build_info')} |",
        f"| Slots | {(result.get('llama_props') or {}).get('total_slots')} |",
        f"| Context per slot | {result.get('n_ctx_slot')} tokens |",
        f"| Threads | {result.get('n_threads')} (`-t {report['config']['threads']}`) |",
        f"| Command | `{show(report['llama_command_display'])}` |", "",
        "Gateway settings: " + ", ".join(f"`{k}={v}`" for k, v in (result.get("gateway_env") or {}).items()), "",
        "## Checks", "", "| Check | Result | Detail |", "|---|---|---|",
        *[f"| {c['name']} | {'pass' if c['ok'] else '**FAIL**'} | {c['detail']} |" for c in result["checks"]], "",
        "## Model load time", "",
        "Ready = process start until `/health` returns 200 (polled; up to 0.25 s coarse on Windows). "
        "Loaded = llama-server's own log timestamp.",
        "The first launch may read the model from disk; later launches use the OS file cache.", "",
        "| Launch | Ready (s) | Loaded per log (s) | Memory after load (MB) |", "|---|---|---|---|",
        *[f"| {l['launch']} | {l['ready_s']} | {l['log_model_loaded_s']} | {(l['memory_after_load'] or {}).get('rss_mb')} |"
          for l in result["launches"]], "",
        "## Memory", "",
        "Resident = working set (includes the memory-mapped model). Private = not backed by the model file.", "",
        "| Process | Resident now (MB) | Peak resident (MB) | Private (MB) |", "|---|---|---|---|",
    ]
    for name, key in (("llama-server", "llama_memory_after_benchmarks"), ("gateway", "gateway_memory_after_benchmarks")):
        memory = result.get(key) or {}
        lines.append(f"| {name} | {memory.get('rss_mb')} | {memory.get('peak_rss_mb')} | {memory.get('private_mb')} |")
    config = report["config"]
    lines += ["", "## Latency and throughput", "",
              f"Closed loop, concurrency {config['parallel']}, {config['requests']} requests after "
              f"{config['warmup']} warm-up, `max_tokens={config['max_tokens']}`, `temperature=0`, non-streaming.",
              "\"Same prompt\" lets llama-server reuse its prompt cache; \"different prompt\" forces it to "
              "process the whole prompt every time, which is closer to real traffic.", "",
              "| Path | Prompts | OK | Requests/s | p50 (s) | p95 (s) | p99 (s) | Tokens/s | Prompt tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
    p50 = {}
    for item in result["benchmarks"]:
        data = item["result"]
        if not data or not data["latency_s"]:
            lines.append(f"| {item['target']} | {'different' if item['unique_prompts'] else 'same'} | failed | | | | | | |")
            continue
        latency = data["latency_s"]
        p50[item["name"]] = latency["p50"]
        lines.append(
            f"| {item['target']} | {'different' if item['unique_prompts'] else 'same'} | {data['ok']}/{data['requests']} "
            f"| {data['requests_per_s']:.3f} | {latency['p50']:.3f} | {latency['p95']:.3f} | {latency['p99']:.3f} "
            f"| {data['completion_tokens_per_s']:.1f} | {data['prompt_tokens_total'] // max(data['ok'], 1)} |")
    lines += ["", "Tokens/s is completion tokens divided by wall time, so it includes prompt processing.", ""]
    for kind in ("same", "unique"):
        if f"direct-{kind}-prompt" in p50 and f"gateway-{kind}-prompt" in p50:
            direct, through = p50[f"direct-{kind}-prompt"], p50[f"gateway-{kind}-prompt"]
            lines.append(f"- Gateway overhead at p50, {'same' if kind == 'same' else 'different'} prompts: "
                         f"{(through - direct) * 1000:+.1f} ms ({(through / direct - 1) * 100:+.1f}%).")
    lines += ["", f"With {config['requests']} requests, p99 is close to the slowest request; "
              "use `--requests 100` or more for meaningful tails. Differences of a few percent between "
              "single runs are within noise.", ""]
    if result.get("outage"):
        lines += ["## Outage behaviour", "",
                  f"With llama-server stopped, `/readyz` answered in {result['outage']['readyz_seconds']} s and a "
                  f"completion request in {result['outage']['completion_seconds']} s. Linux refuses the "
                  "connection at once; Windows only fails when `LLM_LLAMA_SERVER_CONNECT_TIMEOUT_S` "
                  "(default 2 s) expires.", ""]
    if result.get("sample"):
        sample = result["sample"]
        lines += ["## Sample generation through the gateway", "", f"Question: {sample['question']}", "",
                  "```", sample["text"].strip(), "```", "",
                  f"{sample['usage']['completion_tokens']} tokens in {sample['seconds']} s, "
                  f"finish reason `{sample['finish_reason']}`.", ""]
    lines += ["## Reproduce", "", "```", show(report["command_display"]), "```", "",
              "Benchmark commands run by the script:", "", "```",
              *[show(item["display"]) for item in report["bench_commands_display"]], "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen2.5-1.5b")
    parser.add_argument("--model", help="path to a local GGUF file (instead of --preset)")
    parser.add_argument("--llama-server", help="path to llama-server(.exe) (default: pinned download)")
    parser.add_argument("--download", action="store_true", help="allow downloading llama.cpp and the model")
    parser.add_argument("--cache", default=str(ROOT / ".cache"), help="where downloads are kept")
    parser.add_argument("--threads", type=int,
                        help="llama-server -t (default: physical cores, capped at usable CPUs)")
    parser.add_argument("--ctx", type=int, default=4096, help="context tokens per slot")
    parser.add_argument("--parallel", type=int, default=1, help="slots and gateway concurrency")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--load-runs", type=int, default=2, help="how many times to launch llama-server")
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=300, help="per-request timeout in seconds")
    parser.add_argument("--llama-port", type=int, default=8080)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--label", default="unlabelled machine", help='e.g. "my Windows laptop"')
    parser.add_argument("--out", help="report folder (default: benchmarks/results/local/<timestamp>)")
    args = parser.parse_args(argv)
    if min(args.parallel, args.requests, args.max_tokens, args.load_runs, args.ctx) < 1 or args.warmup < 0:
        parser.error("counts must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    for name in ("timeout", "load_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if not all(1 <= port <= 65535 for port in (args.llama_port, args.gateway_port)):
        parser.error("ports must be between 1 and 65535")
    if args.llama_port == args.gateway_port:
        parser.error("llama-port and gateway-port must differ")
    for stream in (sys.stdout, sys.stderr):  # Windows consoles are often cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    stamp = datetime.now(timezone.utc)
    out = Path(args.out) if args.out else ROOT / "benchmarks" / "results" / "local" / stamp.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    if args.model:
        model = Path(args.model)
        if not model.is_file():
            parser.error(f"model file not found: {model}")
        info = {"name": model.name, "file": model.name, "size": model.stat().st_size,
                "sha256": sha256_of(model), "template": "{question}", "stop": None}
    else:
        preset = PRESETS[args.preset]
        model = get_model(preset, args.cache, args.download)
        info = {**preset, "source": f"https://huggingface.co/{preset['repo']}/resolve/{preset['revision']}/{preset['file']}"}
    server_path = Path(args.llama_server) if args.llama_server else get_llama_server(args.cache, args.download)
    if not server_path.is_file():
        parser.error(f"llama-server not found: {server_path}")
    server = [sys.executable, str(server_path)] if server_path.suffix == ".py" else [str(server_path)]

    env = environment()
    if not args.threads:
        args.threads = default_threads(env)
    say(f"Model: {info['name']}  |  llama-server: {server_path.name}  |  threads: {args.threads}  |  report: {out}")
    result = run(args, server, model, info, out)
    passed = bool(result["checks"]) and all(c["ok"] for c in result["checks"])

    def display(command):  # file names only: no user names or local folders in the report
        return [str(part) if "://" in str(part) or not (os.sep in str(part) or "/" in str(part))
                else Path(str(part)).name for part in command]

    try:  # Linux cgroup v2 memory limit (containers, CI sandboxes)
        limit = round(int(Path("/sys/fs/cgroup/memory.max").read_text()) / 1024**3, 1)
    except (OSError, ValueError):
        limit = None
    report = {
        "label": args.label, "memory_limit_gb": limit, "ci": bool(os.environ.get("GITHUB_ACTIONS")),
        "timestamp_utc": stamp.isoformat(timespec="seconds"), "passed": passed,
        "environment": env,
        "model": {k: v for k, v in info.items() if k not in ("template", "stop")},
        "config": {k: v for k, v in vars(args).items() if k not in ("model", "llama_server", "cache", "out")},
        "llama_command_display": display(result.pop("llama_command", [])),
        "command_display": ["python", "scripts/validate_llama.py", *display(argv if argv is not None else sys.argv[1:])],
        "bench_commands_display": [{"name": b["name"], "display": ["python", "benchmarks/bench.py", *display(b.pop("command")[2:])]}
                                   for b in result["benchmarks"]],
        "result": result,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    say(f"\n{'PASS' if passed else 'FAIL'}: report written to {out / 'report.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
