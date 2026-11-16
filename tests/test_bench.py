import asyncio
import importlib.util
import json
import os
import sys
from types import SimpleNamespace
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
    assert set(result) == {"platform", "machine", "cpu_model", "cpu_count", "ram_total_gb", "python",
                           "cpu_physical_cores"}
    assert result["cpu_model"]


def test_environment_inconsistent_core_counts(monkeypatch):
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(bench, "_cpu_physical_cores", lambda system: 64)
    assert bench.environment()["cpu_physical_cores"] is None


def test_cpu_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(bench.Path, "read_text", fail)
    monkeypatch.setattr(bench.subprocess, "check_output", fail)
    assert bench._cpu_model("Linux") == "unknown"
    assert bench._cpu_model("Darwin") == "unknown"
    monkeypatch.setitem(sys.modules, "winreg", SimpleNamespace(OpenKey=fail, HKEY_LOCAL_MACHINE=0))
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


@pytest.mark.parametrize("unique_prompts", [False, True])
@pytest.mark.parametrize("through_main", [False, True])
def test_unique_prompts(monkeypatch, unique_prompts, through_main):
    received = []

    class RecordingBackend(EchoBackend):
        async def generate(self, prompt, *args):
            received.append(prompt)
            return await super().generate(prompt, *args)

    app = create_app(Settings(), RecordingBackend())
    client_class = httpx.AsyncClient

    def client_factory(**kwargs):
        return client_class(transport=httpx.ASGITransport(app), **kwargs)

    if through_main:
        captured = {}
        monkeypatch.setattr(bench.httpx, "AsyncClient", client_factory)
        monkeypatch.setattr(bench, "environment", lambda: {})
        monkeypatch.setattr(bench.Path, "write_text",
                            lambda self, text, **k: captured.update(json.loads(text)))
        args = ["--model", "echo-1", "--requests", "6", "--warmup", "2",
                "--concurrency", "2", "--prompt", OPTIONS["prompt"], "--output", "unused.json"]
        assert bench.main(args + (["--unique-prompts"] if unique_prompts else [])) == 0
        assert captured["config"]["unique_prompts"] is unique_prompts
    else:
        async def scenario():
            async with client_factory(base_url="http://test") as client:
                records = await bench.run_benchmark(client, **OPTIONS, unique_prompts=unique_prompts)
            assert len(records) == 6
            assert all(r["status"] == 200 for r in records)
        asyncio.run(scenario())
    assert len(received) == 8
    assert all(prompt.endswith(OPTIONS["prompt"]) for prompt in received)
    if unique_prompts and through_main:
        # main() adds a per-run id so a second run never repeats the first run's prompts.
        prefixes = {prompt[: -len(OPTIONS["prompt"])] for prompt in received}
        run_ids = {prefix.split("-")[0] for prefix in prefixes}
        assert len(prefixes) == 8 and len(run_ids) == 1
        first_run = set(received)
        received.clear()
        assert bench.main(args + ["--unique-prompts"]) == 0
        assert not first_run & set(received)
    elif unique_prompts:
        assert set(received) == {f"[request {n}] {OPTIONS['prompt']}" for n in range(8)}
    else:
        assert received == [OPTIONS["prompt"]] * 8


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hardware")
def test_windows_environment():
    result = bench.environment()
    assert result["cpu_model"] != "unknown"
    assert isinstance(result["cpu_physical_cores"], int)
    assert 0 < result["cpu_physical_cores"] <= os.cpu_count()
    assert result["ram_total_gb"] > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux hardware")
def test_linux_physical_cores():
    cores = bench.environment()["cpu_physical_cores"]
    assert cores is None or (isinstance(cores, int) and 0 < cores <= os.cpu_count())


def test_windows_cpu_registry(monkeypatch):
    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def open_key(root, path):
        assert root == 0
        assert path == r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        return Key()

    def query_value(key, name):
        assert name == "ProcessorNameString"
        return "  CPU name  ", 1

    monkeypatch.setitem(sys.modules, "winreg", SimpleNamespace(
        HKEY_LOCAL_MACHINE=0, OpenKey=open_key, QueryValueEx=query_value))
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "family string")
    assert bench._cpu_model("Windows") == "CPU name"


def test_linux_physical_core_pairs(monkeypatch):
    cpuinfo = "\n\n".join(
        f"processor: {n}\nphysical id: {socket}\ncore id: {core}"
        for n, (socket, core) in enumerate([(0, 0), (0, 0), (0, 1), (1, 0)])
    )
    monkeypatch.setattr(bench.Path, "read_text", lambda self: cpuinfo)
    assert bench._cpu_physical_cores("Linux") == 3


@pytest.mark.parametrize("output,expected", [
    ("Core(s) per socket: 4\nSocket(s): 2", 8),
    ("Socket(s): -\nCore(s) per cluster: 4\nCluster(s): 2", 8),
    ("", None), ("Core(s) per socket: 0\nSocket(s): 2", None),
])
def test_linux_physical_core_fallback(monkeypatch, output, expected):
    monkeypatch.setattr(bench.Path, "read_text", lambda self: "CPU part: 0xd03")

    def lscpu(command, **kwargs):
        assert command == ["lscpu"]
        assert kwargs["env"]["LC_ALL"] == "C"
        return output

    monkeypatch.setattr(bench.subprocess, "check_output", lscpu)
    assert bench._cpu_physical_cores("Linux") == expected


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
@pytest.mark.parametrize("output,expected", [(" 8\n", 8), ("", None), ("bad", None), ("0", None)])
def test_physical_core_commands(monkeypatch, system, output, expected):
    def command(args, **kwargs):
        if system == "Windows":
            assert args == ["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]
            assert kwargs["timeout"] == 15
        else:
            assert args == ["sysctl", "-n", "hw.physicalcpu"]
        return output

    monkeypatch.setattr(bench.subprocess, "check_output", command)
    assert bench._cpu_physical_cores(system) == expected


@pytest.mark.parametrize("system", ["Linux", "Windows", "Darwin", "other"])
def test_physical_cores_failure(monkeypatch, system):
    def fail(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(bench.Path, "read_text", fail)
    monkeypatch.setattr(bench.subprocess, "check_output", fail)
    assert bench._cpu_physical_cores(system) is None
