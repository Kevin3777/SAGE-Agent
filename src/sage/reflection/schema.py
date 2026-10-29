"""Typed, provenance-carrying reflective experience."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Reflection:
    task_id: str
    summary: str
    error_description: str
    root_cause: str
    failed_decision: str
    correction: str
    next_action: str
    reusable_experience: str
    task_categories: list[str]
    evidence_refs: list[str]
    confidence: float
    source_feedback_id: str
    method: str = "evidence-grounded-template-v1"
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reflection":
        return cls(**data)
