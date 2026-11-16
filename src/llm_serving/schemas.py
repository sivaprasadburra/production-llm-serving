from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0, le=2)
    stop: str | list[str] | None = None
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must not be empty after stripping whitespace")
        return value

    @field_validator("stop")
    @classmethod
    def valid_stop(cls, value):
        items = [value] if isinstance(value, str) else value
        if items is not None and (len(items) > 4 or any(not item for item in items)):
            raise ValueError("stop must contain at most 4 non-empty strings")
        return items


class Choice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: Literal["local"] = "local"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
