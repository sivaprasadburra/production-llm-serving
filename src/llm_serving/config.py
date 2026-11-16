import math
import os
from dataclasses import dataclass, field, fields


@dataclass
class Settings:
    backend: str = "echo"
    model_id: str = "echo-1"
    max_request_bytes: int = 1048576
    max_prompt_chars: int = 8192
    max_tokens_limit: int = 512
    max_concurrency: int = 4
    request_timeout_s: float = 30.0
    llama_server_url: str = "http://127.0.0.1:8080"
    llama_server_api_key: str = field(default="", repr=False)  # keep out of logs
    llama_server_connect_timeout_s: float = 2.0

    def __post_init__(self):
        if not self.llama_server_url.startswith(("http://", "https://")):
            raise ValueError("LLM_LLAMA_SERVER_URL must start with http:// or https://")
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int and (type(value) is not int or value <= 0):
                raise ValueError(f"LLM_{f.name.upper()} must be a positive integer")
            if f.type is float and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"LLM_{f.name.upper()} must be a positive number")

    @classmethod
    def from_env(cls):
        values = {}
        for f in fields(cls):
            name = f"LLM_{f.name.upper()}"
            if name in os.environ:
                value = os.environ[name]
                if f.type in (int, float):
                    kind = "integer" if f.type is int else "number"
                    try:
                        value = f.type(value)
                    except ValueError:
                        raise ValueError(f"{name} must be a positive {kind}") from None
                values[f.name] = value
        return cls(**values)
