"""Deterministic test backend; its outputs are always labeled as mock."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from .base import GenerationResult


class MockBackend:
    backend_type = "mock"

    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Callable[[list[dict[str, str]], int], str] | None = None,
        model_identifier: str = "deterministic-mock-v1",
    ) -> None:
        self.responses = deque(responses or [])
        self.responder = responder
        self.model_identifier = model_identifier
        self.calls: list[dict[str, Any]] = []
        self._call_count = 0

    def generate(self, messages: list[dict[str, str]], config: dict[str, Any] | None = None) -> GenerationResult:
        started = time.perf_counter()
        if self.responder is not None:
            text = self.responder(messages, self._call_count)
        elif self.responses:
            text = self.responses.popleft()
        else:
            raise RuntimeError("MockBackend has no queued response")
        self._call_count += 1
        self.calls.append({"messages": messages, "output": text, "config": config or {}})
        return GenerationResult(
            text=text, backend_type=self.backend_type, model_identifier=self.model_identifier,
            generation_parameters=dict(config or {}), latency=time.perf_counter() - started,
            input_tokens=sum(len(m.get("content", "").split()) for m in messages),
            output_tokens=len(text.split()), raw_response={"mock": True},
        )
