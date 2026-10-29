"""Assistant role: delegates each action to the configured actual model backend."""
from __future__ import annotations

from typing import Any

from sage.backends.base import BaseModelBackend, GenerationResult


class Assistant:
    def __init__(self, backend: BaseModelBackend, generation_config: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.generation_config = generation_config or {"temperature": 0.0, "max_tokens": 256}

    def act(self, messages: list[dict[str, str]]) -> GenerationResult:
        return self.backend.generate(messages, self.generation_config)
