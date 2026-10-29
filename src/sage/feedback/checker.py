"""Role wrapper that records every environment verification."""
from __future__ import annotations

from typing import Any, Protocol

from sage.core.types import Feedback, TaskCase


class CheckableEnvironment(Protocol):
    def check(self, task: TaskCase, candidate: Any) -> Feedback: ...


class Checker:
    def __init__(self, environment: CheckableEnvironment) -> None:
        self.environment = environment
        self.calls = 0

    def check(self, task: TaskCase, candidate: Any) -> Feedback:
        self.calls += 1
        return self.environment.check(task, candidate)
