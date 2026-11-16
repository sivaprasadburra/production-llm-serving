"""Closed-loop completion benchmark; uses only httpx and the standard library."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import httpx


def percentile(sorted_values, q):
    """Linearly interpolate a percentile (q in [0, 100]) of sorted values."""
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    if not sorted_values:
        return None
    rank = (len(sorted_values) - 1) * q / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


async def run_benchmark(client: httpx.AsyncClient, *, model, requests, concurrency,
                        warmup, prompt, max_tokens, temperature) -> list[dict]:
    payload = dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    async def send():
        started = time.perf_counter()
        try:
            response = await client.post("v1/completions", json=payload)
        except httpx.RequestError as exc:
            return {"latency_s": time.perf_counter() - started,
                    "status": f"error:{type(exc).__name__}"}
        record = {"latency_s": time.perf_counter() - started, "status": response.status_code}
        if response.status_code == 200:
            usage = response.json()["usage"]
            record.update(prompt_tokens=usage["prompt_tokens"],
                          completion_tokens=usage["completion_tokens"])
        return record

    for _ in range(warmup):
        await send()
    records = [None] * requests
    next_request = 0

    async def worker():
        nonlocal next_request
        while next_request < requests:
            number = next_request
            next_request += 1  # No await: taking a number is atomic within the event loop.
            records[number] = await send()

    await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
    return records


def summarize(records, wall_time_s) -> dict:
    successful = [r for r in records if r["status"] == 200]
    latencies = sorted(r["latency_s"] for r in successful)
    completion_tokens = sum(r["completion_tokens"] for r in successful)
    return {
        "wall_time_s": wall_time_s, "requests": len(records), "ok": len(successful),
        "status_counts": dict(Counter(str(r["status"]) for r in records)),
        "requests_per_s": len(successful) / wall_time_s,
        "latency_s": {
            "mean": statistics.mean(latencies), "min": latencies[0],
            "p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99), "max": latencies[-1],
        } if latencies else None,
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in successful),
        "completion_tokens_total": completion_tokens,
        "completion_tokens_per_s": completion_tokens / wall_time_s,
        "completion_tokens_per_s_per_request_mean": statistics.mean(
            r["completion_tokens"] / r["latency_s"] for r in successful
        ) if successful else None,
    }


def _field(text, key):
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip() == key and value.strip():
            return value.strip()
    return None


def _cpu_model(system):
    try:
        if system == "Linux":
            try:
                cpuinfo = Path("/proc/cpuinfo").read_text()
            except OSError:
                cpuinfo = ""
            name = _field(cpuinfo, "model name")  # x86
            if not name:  # ARM: /proc/cpuinfo only has a numeric "CPU part"
                try:
                    lscpu = subprocess.check_output(
                        ["lscpu"], text=True, timeout=2, env={**os.environ, "LC_ALL": "C"}
                    )
                except (OSError, subprocess.SubprocessError):
                    lscpu = ""
                name = _field(lscpu, "Model name") or _field(cpuinfo, "CPU part")
            return name or "unknown"
        if system == "Windows":
            return os.environ.get("PROCESSOR_IDENTIFIER") or "unknown"
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip() or "unknown"
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def environment() -> dict:
    system = platform.system()
    ram = None
    try:
        if system == "Linux":
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        elif system == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in (
                        "total_phys", "avail_phys", "total_page", "avail_page",
                        "total_virtual", "avail_virtual", "avail_extended",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.total_phys / 1024**3
    except Exception:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": _cpu_model(system), "cpu_count": os.cpu_count(),
            "ram_total_gb": round(ram, 1) if ram is not None else None,
            "python": platform.python_version()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default="Once upon a time")
    prompts.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--label")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if min(args.requests, args.concurrency, args.max_tokens) < 1 or args.warmup < 0:
        parser.error("requests, concurrency and max-tokens must be positive; warmup must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    except OSError as exc:
        parser.error(str(exc))
    config = {key: value for key, value in vars(args).items() if key != "prompt"}
    config["prompt_chars"] = len(prompt)

    async def measure():
        async with httpx.AsyncClient(
            base_url=args.url.rstrip("/") + "/", trust_env=False, timeout=args.timeout,
            limits=httpx.Limits(max_connections=args.concurrency,
                               max_keepalive_connections=args.concurrency),
        ) as client:
            common = dict(model=args.model, concurrency=args.concurrency, prompt=prompt,
                          max_tokens=args.max_tokens, temperature=args.temperature)
            await run_benchmark(client, requests=0, warmup=args.warmup, **common)
            started = time.perf_counter()
            records = await run_benchmark(client, requests=args.requests, warmup=0, **common)
            return summarize(records, time.perf_counter() - started)

    result = {"label": args.label, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "environment": environment(), **asyncio.run(measure())}
    print(f"{result['ok']}/{result['requests']} succeeded in {result['wall_time_s']:.3f}s; "
          f"{result['requests_per_s']:.2f} requests/s; "
          f"{result['completion_tokens_per_s']:.2f} completion tokens/s; "
          f"statuses: {result['status_counts']}")
    if result["latency_s"]:
        print("latency ms: " + ", ".join(
            f"{name} {result['latency_s'][name] * 1000:.1f}" for name in ("p50", "p95", "p99", "max")
        ))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
