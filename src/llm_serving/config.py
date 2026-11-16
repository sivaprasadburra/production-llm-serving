import math
import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    backend: str = "echo"
    model_id: str = "echo-1"
    max_request_bytes: int = 1048576
    max_prompt_chars: int = 8192
    max_tokens_limit: int = 512
    max_concurrency: int = 4
    request_timeout_s: float = 30.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and (type(value) is not int or value <= 0):
                raise ValueError(f"LLM_{field.name.upper()} must be a positive integer")
            if field.type is float and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"LLM_{field.name.upper()} must be a positive number")

    @classmethod
    def from_env(cls):
        values = {}
        for field in fields(cls):
            name = f"LLM_{field.name.upper()}"
            if name in os.environ:
                value = os.environ[name]
                if field.type in (int, float):
                    kind = "integer" if field.type is int else "number"
                    try:
                        value = field.type(value)
                    except ValueError:
                        raise ValueError(f"{name} must be a positive {kind}") from None
                values[field.name] = value
        return cls(**values)
