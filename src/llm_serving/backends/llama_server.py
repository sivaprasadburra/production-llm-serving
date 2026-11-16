"""Adapter for a separately running llama.cpp `llama-server` (OpenAI-compatible HTTP API).

Design: docs/adr/0001-llama-server-adapter.md
"""
import logging

import httpx

from ..config import Settings
from . import BackendError, GenerationResult

logger = logging.getLogger(__name__)

UNAVAILABLE = "model backend unavailable"
BAD_GATEWAY = "model backend error"


class LlamaServerBackend:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        key = settings.llama_server_api_key
        self._probe_timeout = settings.llama_server_connect_timeout_s
        self._max_concurrency = settings.max_concurrency
        self._slots_checked = False
        self._client = httpx.AsyncClient(
            base_url=settings.llama_server_url,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            # No read timeout: the service-level deadline cancels this request,
            # which closes the connection and makes llama-server stop generating.
            timeout=httpx.Timeout(None, connect=settings.llama_server_connect_timeout_s),
            transport=transport,
            trust_env=False,  # never route a local upstream through HTTP(S)_PROXY
        )

    async def is_ready(self) -> bool:
        """True once llama-server has finished loading the model."""
        try:
            response = await self._client.get("/health", timeout=self._probe_timeout)
        except httpx.HTTPError:
            response = None
        if response is None or response.status_code != 200:
            self._slots_checked = False  # re-check after llama-server comes back
            return False
        if not self._slots_checked:
            await self._check_slots()
        return True

    async def _check_slots(self) -> None:
        """Warn once if we admit more requests than llama-server has slots.

        The excess would queue inside llama-server (invisible to clients, eating
        their timeout budget) instead of failing fast with a 503 here.
        """
        try:
            response = await self._client.get("/props", timeout=self._probe_timeout)
            slots = response.json()["total_slots"]
        except (httpx.HTTPError, ValueError, LookupError, TypeError):
            return  # /props is optional; try again on the next readiness probe
        self._slots_checked = True
        if type(slots) is int and self._max_concurrency > slots:
            logger.warning(
                "LLM_MAX_CONCURRENCY=%s exceeds llama-server slots (--parallel %s): "
                "excess requests will queue upstream instead of getting a 503",
                self._max_concurrency, slots,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> GenerationResult:
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                   "stream": False}
        if stop:
            payload["stop"] = stop
        try:
            response = await self._client.post("/v1/completions", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise BackendError(503, UNAVAILABLE) from exc
        except httpx.HTTPError as exc:  # connection dropped mid-generation, protocol error, ...
            raise BackendError(502, BAD_GATEWAY) from exc

        if response.status_code == 200:
            return _parse_completion(response)
        if response.status_code == 503:  # model still loading
            raise BackendError(503, UNAVAILABLE)
        if response.status_code == 400:  # e.g. prompt does not fit the context window
            raise BackendError(400, _upstream_message(response))
        # 401/403 mean *our* credentials are wrong; never surface that as a client error.
        logger.error("llama-server returned unexpected status %s", response.status_code)
        raise BackendError(502, BAD_GATEWAY)


def _parse_completion(response: httpx.Response) -> GenerationResult:
    try:
        body = response.json()
        choice, usage = body["choices"][0], body["usage"]
        text, prompt_tokens, completion_tokens = (
            choice["text"], usage["prompt_tokens"], usage["completion_tokens"]
        )
        finish_reason = "length" if choice.get("finish_reason") == "length" else "stop"
    except (ValueError, LookupError, TypeError) as exc:
        raise BackendError(502, BAD_GATEWAY) from exc
    if not (isinstance(text, str) and type(prompt_tokens) is int and type(completion_tokens) is int):
        raise BackendError(502, BAD_GATEWAY)
    return GenerationResult(text, prompt_tokens, completion_tokens, finish_reason)


def _upstream_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, LookupError, TypeError):
        message = None
    if not isinstance(message, str) or not message:
        return "request rejected by model backend"
    return message[:200]
