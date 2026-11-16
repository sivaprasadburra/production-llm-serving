from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


class Backend(Protocol):
    model_id: str

    def is_ready(self) -> bool: ...

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> GenerationResult: ...


def get_backend(name: str) -> Backend:
    from .echo import EchoBackend

    if name == "echo":
        return EchoBackend()
    raise ValueError(f"Unknown LLM_BACKEND: {name!r}")
