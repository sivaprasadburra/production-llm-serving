from dataclasses import dataclass
from typing import Literal, Protocol

from ..config import Settings


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


class BackendError(Exception):
    """A failure the client should see with this status and (safe) message.

    Anything else a backend raises is treated as a bug and reported as a 500.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Backend(Protocol):
    async def is_ready(self) -> bool: ...

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> GenerationResult: ...

    async def aclose(self) -> None: ...


def get_backend(settings: Settings) -> Backend:
    if settings.backend == "echo":
        from .echo import EchoBackend

        return EchoBackend()
    if settings.backend == "llama_server":
        try:
            from .llama_server import LlamaServerBackend
        except ImportError as exc:
            raise ValueError(
                'LLM_BACKEND=llama_server needs httpx: pip install "llm_serving[llama]"'
            ) from exc
        return LlamaServerBackend(settings)
    raise ValueError(f"Unknown LLM_BACKEND: {settings.backend!r}")
