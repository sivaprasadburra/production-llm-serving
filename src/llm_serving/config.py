import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    backend: str = "echo"
    model_id: str = "echo-1"
    max_prompt_chars: int = 8192
    max_tokens_limit: int = 512
    max_concurrency: int = 4
    request_timeout_s: int = 30

    def __post_init__(self):
        for field in fields(self):
            if field.type is int:
                value = getattr(self, field.name)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"LLM_{field.name.upper()} must be a positive integer")

    @classmethod
    def from_env(cls):
        values = {}
        for field in fields(cls):
            name = f"LLM_{field.name.upper()}"
            if name in os.environ:
                value = os.environ[name]
                if field.type is int:
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(f"{name} must be a positive integer") from None
                values[field.name] = value
        return cls(**values)
