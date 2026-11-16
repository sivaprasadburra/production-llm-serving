import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from llm_serving.app import create_app
from llm_serving.backends.echo import EchoBackend
from llm_serving.config import Settings

spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).resolve().parents[1] / "benchmarks" / "bench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("values,q,expected", [
    ([7], 99, 7), ([1, 2, 3, 4], 0, 1), ([1, 2, 3, 4], 100, 4),
    ([1, 2, 3, 4], 50, 2.5), ([0, 10], 95, 9.5), ([], 50, None),
])
def test_percentile(values, q, expected):
    assert bench.percentile(values, q) == expected


def test_summarize():
    records = [
        dict(status=200, latency_s=1, prompt_tokens=2, completion_tokens=4),
        dict(status=503, latency_s=100),
        dict(status="error:ReadTimeout", latency_s=200),
        dict(status=200, latency_s=3, prompt_tokens=3, completion_tokens=6),
    ]
    result = bench.summarize(records, 5)
    assert result == {
        "wall_time_s": 5, "requests": 4, "ok": 2,
        "status_counts": {"200": 2, "503": 1, "error:ReadTimeout": 1},
        "requests_per_s": 0.4,
        "latency_s": {"mean": 2, "min": 1, "p50": 2, "p95": 2.9, "p99": 2.98, "max": 3},
        "prompt_tokens_total": 5, "completion_tokens_total": 10,
        "completion_tokens_per_s": 2, "completion_tokens_per_s_per_request_mean": 3,
    }
    failed = bench.summarize(records[1:3], 5)
    assert failed["latency_s"] is None
    assert failed["completion_tokens_per_s_per_request_mean"] is None
    assert failed["requests_per_s"] == failed["completion_tokens_per_s"] == 0
    assert failed["prompt_tokens_total"] == failed["completion_tokens_total"] == 0


OPTIONS = dict(model="echo-1", requests=6, concurrency=2, warmup=2,
               prompt="one two", max_tokens=5, temperature=0.0)


def test_echo():
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(Settings())), base_url="http://test"
        ) as client:
            records = await bench.run_benchmark(client, **OPTIONS)
        assert len(records) == 6
        assert all(r["status"] == 200 and r["completion_tokens"] == 5 for r in records)
        assert all(r["prompt_tokens"] == 2 and r["latency_s"] > 0 for r in records)
    asyncio.run(scenario())


@pytest.mark.parametrize("concurrency", [1, 2, 3])
def test_concurrency_and_warmup(concurrency):
    async def scenario():
        class CountingBackend(EchoBackend):
            calls = active = peak = 0

            async def generate(self, *args):
                self.calls += 1
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.calls <= 2:
                    assert self.active == 1  # Sequential warmup.
                else:
                    if self.active == concurrency:
                        full.set()
                    await release.wait()
                try:
                    return await super().generate(*args)
                finally:
                    self.active -= 1

        full, release = asyncio.Event(), asyncio.Event()
        backend = CountingBackend()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(Settings(), backend)), base_url="http://test"
        ) as client:
            task = asyncio.create_task(bench.run_benchmark(
                client, **{**OPTIONS, "concurrency": concurrency}
            ))
            await asyncio.wait_for(full.wait(), 1)
            assert backend.active == concurrency
            release.set()
            records = await asyncio.wait_for(task, 1)
        assert backend.calls == 8
        assert backend.peak == concurrency
        assert len(records) == 6
        assert all(r["status"] == 200 for r in records)
    asyncio.run(scenario())


def test_capacity():
    async def scenario():
        entered, rejected, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class BlockingBackend(EchoBackend):
            async def generate(self, *args):
                entered.set()
                await release.wait()
                return await super().generate(*args)

        async def observe(response):
            if response.status_code == 503:
                rejected.set()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_app(Settings(max_concurrency=1), BlockingBackend())),
            base_url="http://test", event_hooks={"response": [observe]},
        ) as client:
            task = asyncio.create_task(bench.run_benchmark(
                client, **{**OPTIONS, "requests": 2, "warmup": 0}
            ))
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(rejected.wait(), 1)
            release.set()
            records = await asyncio.wait_for(task, 1)
        assert bench.summarize(records, 1)["status_counts"] == {"200": 1, "503": 1}
    asyncio.run(scenario())


def test_transport_errors_are_not_retried():
    async def scenario():
        calls = 0

        def fail(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(fail), base_url="http://test") as client:
            records = await bench.run_benchmark(client, **OPTIONS)
        assert calls == 8
        assert len(records) == 6
        assert all(r["status"] == "error:ReadTimeout" for r in records)
    asyncio.run(scenario())


def test_environment():
    result = bench.environment()
    assert set(result) == {"platform", "machine", "cpu_model", "cpu_count", "ram_total_gb", "python"}
    assert result["cpu_model"]


def test_cpu_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(bench.Path, "read_text", fail)
    monkeypatch.setattr(bench.subprocess, "check_output", fail)
    assert bench._cpu_model("Linux") == "unknown"
    assert bench._cpu_model("Darwin") == "unknown"
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "test CPU")
    assert bench._cpu_model("Windows") == "test CPU"


def test_main_failure(capsys):
    assert bench.main(["--url", "http://127.0.0.1:1", "--model", "echo-1",
                       "--requests", "1", "--warmup", "0", "--timeout", "1"]) == 1
    assert "0/1 succeeded" in capsys.readouterr().out


def test_main_config_and_timing(monkeypatch, tmp_path, capsys):
    # Capture JSON in memory; do not write benchmark results to disk.
    captured = {}
    calls = []
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(bench, "environment", lambda: {})
    monkeypatch.setattr(bench.Path, "read_text", lambda *a, **k: "private prompt")
    monkeypatch.setattr(bench.Path, "write_text", lambda self, text, **k: captured.update(json.loads(text)))

    async def fake_run(client, **kwargs):
        calls.append(kwargs)
        assert str(client.base_url) == "http://test/prefix/"
        return [dict(status=200, latency_s=1, prompt_tokens=2, completion_tokens=4)] * kwargs["requests"]

    monkeypatch.setattr(bench, "run_benchmark", fake_run)
    assert bench.main(["--url", "http://test/prefix", "--model", "echo-1", "--requests", "2",
                       "--prompt-file", "prompt.txt", "--output", str(tmp_path / "unused.json")]) == 0
    assert [(c["requests"], c["warmup"]) for c in calls] == [(0, 1), (2, 0)]
    assert captured["wall_time_s"] == 2
    assert captured["config"]["prompt_chars"] == 14
    assert "private prompt" not in json.dumps(captured)
    assert "prompt" not in captured["config"]
    assert "2/2 succeeded" in capsys.readouterr().out
