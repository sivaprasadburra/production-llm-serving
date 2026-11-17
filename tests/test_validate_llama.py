"""Tests for scripts/validate_llama.py. They run on Linux and Windows in CI."""
import hashlib
import http.server
import importlib.util
import json
import os
import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validate_llama", ROOT / "scripts" / "validate_llama.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

FAKE_SERVER = Path(__file__).with_name("fake_llama_server.py")
measures_memory = pytest.mark.skipif(not sys.platform.startswith(("linux", "win32")),
                                     reason="peak/private memory is read on Linux and Windows")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@measures_memory
def test_process_memory_of_this_process():
    memory = validate.process_memory(os.getpid())
    assert memory["rss_mb"] > 1
    assert memory["peak_rss_mb"] >= memory["rss_mb"] * 0.99
    assert memory["private_mb"] > 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set counters")
def test_windows_working_set_grows_with_allocation():
    before = validate.process_memory(os.getpid())
    block = bytearray(64 * 1024 * 1024)  # touched (zero-filled) => committed and resident
    after = validate.process_memory(os.getpid())
    assert after["private_mb"] - before["private_mb"] > 48
    assert after["peak_rss_mb"] >= before["peak_rss_mb"]
    del block


def test_process_memory_of_missing_process_is_none():
    assert validate.process_memory(2**22 + 12345) is None


def test_llama_log_parsing():
    log = ("0.00.073.964 I cmn          init: llama threadpool init, n_threads = 8\n"
           "0.00.604.431 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'\n"
           "1.02.091.338 I srv  server_run_i: model loaded\n")
    assert validate.log_seconds(log, "model loaded") == 62.091
    assert validate.log_seconds(log, "never printed") is None
    assert validate.log_int(log, "n_threads") == 8
    assert validate.log_int(log, "n_ctx_slot") == 4096
    assert validate.log_int("", "n_threads") is None


@pytest.mark.parametrize("env,expected", [
    ({"cpu_physical_cores": 4, "cpu_count": 8}, 4),     # hyper-threaded laptop
    ({"cpu_physical_cores": None, "cpu_count": 4}, 4),  # unknown topology
    ({"cpu_physical_cores": None, "cpu_count": None}, 1),
])
def test_default_threads_never_exceeds_usable_cpus(env, expected, monkeypatch):
    monkeypatch.setattr(validate.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert validate.default_threads(env) == expected


def test_default_threads_respects_cgroup_quota(monkeypatch):
    monkeypatch.setattr(validate.Path, "read_text", lambda *a, **k: "200000 100000\n")
    assert validate.default_threads({"cpu_physical_cores": 64, "cpu_count": 64}) == 2
    monkeypatch.setattr(validate.Path, "read_text", lambda *a, **k: "max 100000\n")
    assert validate.default_threads({"cpu_physical_cores": 6, "cpu_count": 12}) == 6


def test_commands_and_gateway_settings():
    args = validate.argparse.Namespace(llama_port=8081, parallel=2, ctx=2048, threads=6,
                                       timeout=300.0, max_tokens=64)
    command = validate.llama_command(["llama-server.exe"], Path("C:/models/my model.gguf"), args, "m")
    assert command[:3] == ["llama-server.exe", "-m", str(Path("C:/models/my model.gguf"))]
    assert command[command.index("-np") + 1] == "2"
    assert command[command.index("-c") + 1] == "4096"  # total context = per-slot context x slots
    assert command[command.index("-t") + 1] == "6"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    env = validate.gateway_env(args, "m")
    assert env["LLM_MAX_CONCURRENCY"] == "2"  # must equal -np
    assert env["LLM_LLAMA_SERVER_URL"] == "http://127.0.0.1:8081"
    assert validate.show(command).count('"') == 2  # only the path with a space is quoted


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    ranges = []

    def do_GET(self):
        start = 0
        header = self.headers.get("Range")
        type(self).ranges.append(header)
        if header:
            start = int(header.split("=")[1].rstrip("-"))
        body = self.payload[start:]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def file_server(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    RangeHandler.payload = os.urandom(300_000)
    RangeHandler.ranges = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/file.bin"
    server.shutdown()
    thread.join(5)


def test_download_verifies_resumes_and_caches(file_server, tmp_path):
    payload = RangeHandler.payload
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "sub" / "file.bin"

    assert validate.download(file_server, dest, digest).read_bytes() == payload
    assert RangeHandler.ranges == [None]
    validate.download(file_server, dest, digest)  # already present and correct: no request
    assert RangeHandler.ranges == [None]

    dest.unlink()
    (tmp_path / "sub" / "file.bin.part").write_bytes(payload[:100_000])  # interrupted earlier
    assert validate.download(file_server, dest, digest).read_bytes() == payload
    assert RangeHandler.ranges[-1] == "bytes=100000-"


def test_download_rejects_wrong_checksum(file_server, tmp_path):
    dest = tmp_path / "file.bin"
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        validate.download(file_server, dest, "0" * 64)
    assert not dest.exists()
    assert not (tmp_path / "file.bin.part").exists()


def test_nothing_is_downloaded_without_permission(tmp_path):
    with pytest.raises(SystemExit, match="--download"):
        validate.get_llama_server(tmp_path, allow_download=False)
    with pytest.raises(SystemExit, match="--download"):
        validate.get_model(validate.PRESETS["qwen2.5-1.5b"], tmp_path, allow_download=False)


def test_presets_are_pinned():
    for preset in validate.PRESETS.values():
        assert len(preset["revision"]) == 40 and len(preset["sha256"]) == 64 and preset["size"] > 0
    for asset, digest in validate.LLAMA_ASSETS.values():
        assert validate.LLAMA_TAG in asset and len(digest) == 64
    assert ("Windows", "AMD64") in validate.LLAMA_ASSETS


@pytest.mark.parametrize("relative_output", [False, True])
def test_end_to_end_with_a_fake_llama_server(tmp_path, monkeypatch, capsys, relative_output):
    """Full run: real processes, real HTTP, real benchmark harness; only the model is fake."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    model = tmp_path / "private folder" / "fake model.gguf"
    model.parent.mkdir()
    model.write_bytes(b"GGUF not really")
    out = tmp_path / "report"
    if relative_output:
        monkeypatch.chdir(tmp_path)
        import shutil
        shutil.copy(FAKE_SERVER, tmp_path / "fake_llama_server.py")
    code = validate.main([
        "--model", str(model), "--llama-server", "fake_llama_server.py" if relative_output else str(FAKE_SERVER), "--load-runs", "2",
        "--requests", "3", "--warmup", "1", "--max-tokens", "8", "--threads", "2",
        "--llama-port", str(free_port()), "--gateway-port", str(free_port()),
        "--label", "unit test", "--out", "report" if relative_output else str(out),
    ])
    printed = capsys.readouterr().out
    assert code == 0, printed

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["ci"]
    assert all(check["ok"] for check in report["result"]["checks"])
    assert report["model"]["sha256"] == hashlib.sha256(b"GGUF not really").hexdigest()
    launches = report["result"]["launches"]
    assert [launch["launch"] for launch in launches] == [1, 2]
    assert all(0.25 < launch["ready_s"] < 30 for launch in launches)  # the fake "loads" for 0.3 s
    assert launches[0]["log_model_loaded_s"] == 0.3
    assert report["result"]["n_threads"] == 2
    benchmarks = report["result"]["benchmarks"]
    assert [b["name"] for b in benchmarks] == ["direct-same-prompt", "gateway-same-prompt",
                                               "direct-unique-prompt", "gateway-unique-prompt"]
    assert all(b["result"]["ok"] == 3 and b["result"]["completion_tokens_total"] == 24 for b in benchmarks)
    assert benchmarks[2]["result"]["config"]["unique_prompts"] is True
    if sys.platform.startswith(("linux", "win32")):
        assert launches[0]["memory_after_load"]["rss_mb"] > 1
        assert report["result"]["llama_memory_after_benchmarks"]["peak_rss_mb"] > 1
        assert report["result"]["gateway_memory_after_benchmarks"]["rss_mb"] > 1

    text = (out / "report.md").read_text(encoding="utf-8")
    assert "Measured on: unit test" in text
    assert "GitHub Actions hosted runner" in text  # CI results can never pass as a user machine
    assert "Gateway overhead at p50" in text
    for private in (str(tmp_path), "private folder"):  # no local folders or user names leak
        assert private not in text
        assert private not in json.dumps(report)

@pytest.mark.parametrize("option,value", [
    ("--threads", "-1"), ("--threads", "0"),
    ("--timeout", "nan"), ("--timeout", "0"),
    ("--load-timeout", "inf"), ("--load-timeout", "-1"),
    ("--llama-port", "0"), ("--gateway-port", "65536"),
])
def test_invalid_runtime_options_fail_before_loading(option, value, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        validate.main([option, value, "--cache", str(tmp_path)])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_ports_must_differ(capsys):
    with pytest.raises(SystemExit) as exc:
        validate.main(["--llama-port", "8000", "--gateway-port", "8000"])
    assert exc.value.code == 2
    assert "must differ" in capsys.readouterr().err
