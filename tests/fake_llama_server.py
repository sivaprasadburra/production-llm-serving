"""Stand-in for the llama-server executable: same flags, log lines and HTTP surface.

Lets tests exercise scripts/validate_llama.py end to end (process start-up,
readiness polling, memory reading, benchmarks, report) without model weights.
"""
import argparse
import json
import sys
import time

import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--version", action="store_true")
parser.add_argument("-m")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--alias", default="fake")
parser.add_argument("-np", type=int, default=1)
parser.add_argument("-c", type=int, default=128)
parser.add_argument("-t", type=int, default=1)
args = parser.parse_args()
if args.version:
    print("version: fake-1 (test double)", file=sys.stderr)
    raise SystemExit(0)

LOAD_SECONDS = 0.3
started = time.monotonic()
print(f"0.00.010.000 I cmn          init: llama threadpool init, n_threads = {args.t}", flush=True)
print(f"0.00.020.000 I srv    load_model: initializing, n_slots = {args.np}, n_ctx_slot = {args.c // args.np}", flush=True)
print("0.00.300.000 I srv  server_run_i: model loaded", flush=True)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    loading = time.monotonic() - started < LOAD_SECONDS
    if scope["path"] == "/health":
        status, payload = (503, {"error": {"message": "Loading model"}}) if loading else (200, {"status": "ok"})
    elif scope["path"] == "/props":
        status, payload = 200, {"total_slots": args.np, "build_info": "fake-1"}
    elif scope["path"] == "/v1/completions" and not loading:
        request = json.loads(body)
        tokens = request["max_tokens"]
        status, payload = 200, {
            "choices": [{"index": 0, "text": " word" * tokens, "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(request["prompt"].split()), "completion_tokens": tokens},
        }
    else:
        status, payload = 503, {"error": {"message": "unavailable"}}
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


uvicorn.run(app, host=args.host, port=args.port, log_level="error", lifespan="off")
