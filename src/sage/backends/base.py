"""Common backend contract and per-call generation evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    backend_type: str
    model_identifier: str
    generation_parameters: dict[str, Any]
    latency: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    call_status: str = "success"
    error_message: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "backend_type": self.backend_type,
            "model_identifier": self.model_identifier,
            "generation_parameters": self.generation_parameters,
            "latency": self.latency,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "call_status": self.call_status,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
        }


class BaseModelBackend(Protocol):
    backend_type: str
    model_identifier: str

    def generate(self, messages: list[dict[str, str]], config: dict[str, Any] | None = None) -> GenerationResult:
        ...
