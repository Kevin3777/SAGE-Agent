"""User role packages a task request and its explicit constraints."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from sage.core.types import TaskCase


@dataclass
class UserObservation:
    task_id: str
    request: str
    constraints: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class User:
    def submit(self, task: TaskCase) -> UserObservation:
        return UserObservation(
            task_id=task.task_id, request=task.user_task,
            constraints=list(task.constraints),
        )
