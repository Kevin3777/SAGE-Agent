"""Generate a concise experience from checker-verified trajectory evidence.

The default generator is deterministic so tests and local experiments remain
reproducible. It derives lessons from structured checker errors and does not
present model speculation as fact.
"""
from __future__ import annotations

from typing import Any

from sage.core.types import Feedback, TaskCase, Trajectory

from .schema import Reflection
from .validator import ReflectionValidator


def _short(text: str, limit: int = 320) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ReflectionGenerator:
    def __init__(self, validator: ReflectionValidator | None = None) -> None:
        self.validator = validator or ReflectionValidator()

    def generate(
        self, task: TaskCase, outputs: list[str], feedbacks: list[Feedback],
        rewards: list[float], recent_trajectory: list[Trajectory],
        available_experiences: list[dict[str, Any]] | None = None,
    ) -> Reflection:
        if not feedbacks:
            raise ValueError("reflection requires at least one checker feedback")
        failed = next((item for item in reversed(feedbacks) if not item.success), None)
        if failed is None:
            raise ValueError("no failed checker evidence to reflect on")
        output = outputs[-1] if outputs else "(no parsed assistant output)"
        errors = failed.errors or [failed.message]
        error_summary = "; ".join(_short(item, 160) for item in errors)
        prior_errors = list(dict.fromkeys(
            error for item in feedbacks[:-1] if not item.success for error in item.errors
        ))
        repeated_decision = len(outputs) > 1 and outputs[-1].strip() == outputs[-2].strip()
        related = (available_experiences or [])
        related_text = _short(str(related[0].get("text", "")), 120) if related else ""
        if task.task_type == "multi_hop_qa":
            next_action = (
                "Resolve the answer through the passage-to-entity chain, use only supported "
                "facts, and include the required passage IDs in the structured answer."
            )
            if any("missing required supporting passages" in item.casefold() for item in errors):
                root_cause = "The answer omitted one or more checker-required passage citations."
            else:
                root_cause = "The chosen answer was not grounded in the complete passage-to-entity link."
            category = "multi-hop question answering"
        elif task.task_type == "sqlite_sql":
            next_action = (
                "Check the provided SQLite schema, build a single read-only SELECT with the "
                "required joins and filters, and compare its columns and rows with the request."
            )
            if "active" in task.user_task.casefold() and "active" not in output.casefold():
                root_cause = "The generated query omitted the active-employee filter stated in the task."
            elif any("no such column" in item.casefold() for item in errors):
                root_cause = "The query referenced a column the SQLite checker could not resolve in the supplied schema."
            else:
                root_cause = "The query's joins, filters, or selected columns did not produce the checker-required result."
            category = "read-only SQL reasoning"
        else:
            next_action = "Use the checker evidence to revise the failed decision before retrying."
            root_cause = "The decision did not satisfy the concrete condition reported by the checker."
            category = task.task_type
        if repeated_decision:
            next_action += " Change the rejected action instead of repeating the same output."
        if prior_errors:
            error_summary = "Earlier attempt(s): " + "; ".join(_short(item, 100) for item in prior_errors) + ". Latest: " + error_summary
        attempt_count = len(outputs)
        reward_sum = sum(float(value) for value in rewards)
        summary = (
            f"Failure lesson for {category} after {attempt_count} attempt(s) "
            f"(cumulative reward {reward_sum:g}): {error_summary}"
        )
        if related_text:
            summary += f" Related stored experience was considered: {related_text}"
        reflection = Reflection(
            task_id=task.task_id,
            summary=summary,
            error_description=error_summary,
            root_cause=root_cause,
            failed_decision=_short(output),
            correction=_short(failed.message),
            next_action=next_action,
            reusable_experience=(
                f"For {category}, treat checker evidence as a constraint on the next action: "
                f"{_short(error_summary, 180)}"
            ),
            task_categories=[task.task_type],
            evidence_refs=list(dict.fromkeys(failed.evidence_refs)),
            confidence=0.8 if failed.evidence_refs and failed.errors else 0.55,
            source_feedback_id=failed.feedback_id,
        )
        return self.validator.validate(
            reflection,
            source_feedback_id=failed.feedback_id,
            evidence_refs=failed.evidence_refs,
            checker_errors=failed.errors or [failed.message],
        )
