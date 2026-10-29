"""Serializable data structures shared by agents, checkers, and memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class MemoryState(str, Enum):
    STM = "STM"
    LTM = "LTM"
    FORGOTTEN = "FORGOTTEN"


@dataclass
class Feedback:
    feedback_id: str
    success: bool
    score: float
    message: str
    errors: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    reward: float = 0.0

    @classmethod
    def create(
        cls, success: bool, score: float, message: str,
        errors: list[str] | None = None, evidence_refs: list[str] | None = None,
    ) -> "Feedback":
        return cls(
            feedback_id=new_id("fb"), success=success, score=float(score),
            message=message, errors=errors or [], evidence_refs=evidence_refs or [],
            reward=1.0 if success else -1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskCase:
    task_id: str
    task_type: str
    user_task: str
    instructions: str
    payload: dict[str, Any]
    constraints: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    task_id: str
    observation: str
    action: str
    assistant_output: str
    checker_feedback: dict[str, Any]
    reward: float
    timestamp: float
    reflection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionGoal:
    memory_goal: str
    strategy_goal: str
    failure_to_address: str
    proposed_changes: list[str]
    evidence_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyRevision:
    version: int
    previous_version: int | None
    revised_instructions: str
    selected_memories: list[str]
    corrective_plan: list[str]
    originating_feedback_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    memory_id: str
    task_type: str
    original_information: str
    optimized_information: str
    reflection: dict[str, Any]
    evidence_refs: list[str]
    initial_strength: float
    optimized_strength: float
    created_at: float
    last_accessed_at: float
    memory_state: MemoryState = MemoryState.STM
    retention_rate: float = 1.0
    reinforcement_count: int = 0
    state_since: float | None = None
    optimization_stats: dict[str, Any] = field(default_factory=dict)
    strength_unit: str = "seconds"

    @property
    def current_strength(self) -> float:
        # Engineering interpretation of Eq. 7: optimized strength is used in STM;
        # the base information strength is used after promotion to LTM.
        if self.memory_state == MemoryState.STM:
            return self.optimized_strength
        return self.initial_strength

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["memory_state"] = self.memory_state.value
        result["current_strength"] = self.current_strength
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        values = dict(data)
        values.pop("current_strength", None)
        values.setdefault("strength_unit", "seconds")
        values["memory_state"] = MemoryState(values.get("memory_state", "LTM"))
        return cls(**values)
