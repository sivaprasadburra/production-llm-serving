from . import GenerationResult


class EchoBackend:
    model_id = "echo-1"

    def is_ready(self) -> bool:
        return True

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
    ) -> GenerationResult:
        tokens = prompt.split()
        text = " ".join(tokens[i % len(tokens)] for i in range(max_tokens)) if tokens else ""
        reason = "length"
        matches = [text.find(item) for item in stop or [] if item in text]
        if matches:
            text = text[:min(matches)]
            reason = "stop"
        return GenerationResult(text, len(tokens), len(text.split()), reason)
