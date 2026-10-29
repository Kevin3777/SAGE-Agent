"""Reject malformed, low-evidence, or checker-contradictory reflections."""
from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import Reflection


class ReflectionValidationError(ValueError):
    pass


class ReflectionValidator:
    required_text = (
        "summary", "error_description", "root_cause", "failed_decision",
        "correction", "next_action", "reusable_experience",
    )

    def validate(
        self, reflection: Reflection, *, source_feedback_id: str,
        evidence_refs: Iterable[str], checker_errors: Iterable[str],
    ) -> Reflection:
        if not isinstance(reflection, Reflection):
            raise ReflectionValidationError("reflection must use the Reflection schema")
        missing = [name for name in self.required_text if not getattr(reflection, name).strip()]
        if missing:
            raise ReflectionValidationError(f"missing required reflection fields: {missing}")
        if not math.isfinite(reflection.confidence) or not 0.0 <= reflection.confidence <= 1.0:
            raise ReflectionValidationError("confidence must be finite and in [0, 1]")
        if reflection.source_feedback_id != source_feedback_id:
            raise ReflectionValidationError("reflection feedback provenance does not match")
        trusted = set(evidence_refs)
        cited = set(reflection.evidence_refs)
        if not cited or not cited.issubset(trusted):
            raise ReflectionValidationError("reflection must cite only available checker evidence")
        error_list = list(checker_errors)
        if not error_list:
            raise ReflectionValidationError("failure reflection requires checker error evidence")
        reflected_error = " ".join(reflection.error_description.casefold().split())
        for error in error_list:
            required_evidence = " ".join(str(error).casefold().split())[:60]
            if required_evidence and required_evidence not in reflected_error:
                raise ReflectionValidationError("reflection omits or contradicts checker error evidence")
        if reflection.correction.casefold() == reflection.failed_decision.casefold():
            raise ReflectionValidationError("correction repeats the failed decision")
        if reflection.confidence > 0.7 and not trusted:
            raise ReflectionValidationError("high-confidence reflection has no supporting evidence")
        reflection.validation_notes = [
            "schema_valid", "evidence_refs_verified", "checker_failure_grounded",
        ]
        return reflection
