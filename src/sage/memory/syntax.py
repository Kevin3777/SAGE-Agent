"""MemorySyntax: extractive linguistic optimization plus explicit strength estimates."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_WORD = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+|[^\w\s]", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+|;\s*")
_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*){0,3}\b|\"([^\"]+)\""
)
_IMPORTANT = re.compile(
    r"\b(must|only|never|exactly|required|constraint|because|cause|error|failed|"
    r"correct|success|should|filter|join|evidence|answer|prevent|ensure)\b",
    re.IGNORECASE,
)
_STRUCTURE = re.compile(
    r"\b(task constraints|failed assistant output|assistant output|checker feedback|checker errors)\s*:",
    re.IGNORECASE,
)
_SECONDS_PER_UNIT = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}


def lexical_token_count(text: str) -> int:
    """Portable token-count proxy; not claimed to match an LLM tokenizer."""
    return len(_WORD.findall(text))


def _sentences(text: str) -> list[str]:
    return [part.strip(" \t\r\n-•") for part in _SENTENCE.split(text) if part.strip(" \t\r\n-•")]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = re.sub(r"\W+", " ", item.casefold()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item.strip())
    return result


@dataclass
class OptimizationResult:
    original_information: str
    optimized_information: str
    important_facts: list[str]
    key_entities: list[str]
    constraints: list[str]
    optimization_valid: bool
    statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_information": self.original_information,
            "optimized_information": self.optimized_information,
            "important_facts": self.important_facts,
            "key_entities": self.key_entities,
            "constraints": self.constraints,
            "optimization_valid": self.optimization_valid,
            "statistics": self.statistics,
        }


class StrengthEstimator:
    """Normalized engineering estimate; the paper does not specify this formula."""

    def __init__(
        self, minimum_seconds: float = 60.0, maximum_seconds: float = 3600.0,
        time_unit: str = "seconds",
    ) -> None:
        if minimum_seconds <= 0 or maximum_seconds <= minimum_seconds:
            raise ValueError("strength range must satisfy 0 < minimum < maximum")
        if time_unit not in _SECONDS_PER_UNIT:
            raise ValueError(f"time_unit must be one of {sorted(_SECONDS_PER_UNIT)}")
        self.minimum_seconds = float(minimum_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.time_unit = time_unit
        self._seconds_per_unit = _SECONDS_PER_UNIT[time_unit]

    def score(
        self, *, checker_quality: float, key_constraint: bool,
        task_importance: float, reusability: float, evidence_count: int,
        reuse_count: int = 0,
    ) -> float:
        values = (checker_quality, task_importance, reusability)
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("strength components must be in [0, 1]")
        if evidence_count < 0 or reuse_count < 0:
            raise ValueError("evidence_count and reuse_count must be non-negative")
        normalized = (
            0.30 * float(checker_quality)
            + 0.15 * float(bool(key_constraint))
            + 0.20 * float(task_importance)
            + 0.20 * float(reusability)
            + 0.10 * min(int(evidence_count) / 3.0, 1.0)
            + 0.05 * min(int(reuse_count) / 5.0, 1.0)
        )
        return min(1.0, max(0.0, normalized))

    def estimate(
        self, *, checker_quality: float, key_constraint: bool,
        task_importance: float, reusability: float, evidence_count: int,
        optimization_valid: bool, compression_ratio: float,
        reuse_count: int = 0,
    ) -> tuple[float, float, float]:
        score = self.score(
            checker_quality=checker_quality, key_constraint=key_constraint,
            task_importance=task_importance, reusability=reusability,
            evidence_count=evidence_count, reuse_count=reuse_count,
        )
        base_seconds = self.minimum_seconds + score * (self.maximum_seconds - self.minimum_seconds)
        base = base_seconds / self._seconds_per_unit
        if optimization_valid and compression_ratio < 1.0 and score > 0:
            boost = min(0.50, 0.08 + 0.35 * (1.0 - compression_ratio) + 0.07 * score)
            optimized = base * (1.0 + boost)
        else:
            optimized = base
        minimum = self.minimum_seconds / self._seconds_per_unit
        maximum = self.maximum_seconds / self._seconds_per_unit
        if not (minimum <= base <= maximum):
            raise ValueError("estimated base strength outside configured range")
        if not (base <= optimized <= maximum * 1.5):
            raise ValueError("estimated optimized strength is invalid")
        return base, optimized, score


class MemorySyntax:
    """Extract key facts, entities, constraints, causes, and actions from a reflection."""

    constraint_markers = ("must", "only", "never", "exactly", "required", "constraint", "should")

    def optimize(
        self, original_information: str, reflection: dict[str, Any] | Any,
        task_constraints: list[str] | None = None, evidence_refs: list[str] | None = None,
    ) -> OptimizationResult:
        reflection_dict = reflection.to_dict() if hasattr(reflection, "to_dict") else dict(reflection or {})
        task_constraints = task_constraints or []
        evidence_refs = evidence_refs or reflection_dict.get("evidence_refs", [])
        source_sentences = _unique(_sentences(original_information))
        eligible_sentences = [
            s for s in source_sentences
            if not _STRUCTURE.search(s) and "{" not in s and "}" not in s
            and not s.lstrip().startswith('"')
        ]
        important_facts = [s for s in eligible_sentences if _IMPORTANT.search(s)]
        if not important_facts and eligible_sentences:
            important_facts = [max(eligible_sentences, key=lexical_token_count)]
        all_source = " ".join([original_information, *task_constraints])
        entities: list[str] = []
        leading_stopwords = {
            "for", "must", "when", "the", "a", "an", "answer", "question", "task",
            "failed", "checker", "we", "i", "this", "what", "who", "which", "cite",
            "ids", "evidence", "assistant",
        }
        for match in _ENTITY.finditer(all_source):
            value = next((group for group in match.groups() if group), None) or match.group(0)
            value = value.strip()
            parts = value.split()
            while parts and parts[0].casefold() in leading_stopwords:
                parts.pop(0)
            value = " ".join(parts)
            if len(value) > 1 and value.casefold() not in {"assistant", "checker", "reflection", "the"}:
                entities.append(value)
        entities = _unique(entities)[:16]
        constraints = _unique(
            list(task_constraints)
            + [s for s in source_sentences if any(marker in s.casefold() for marker in self.constraint_markers)]
        )
        fields: list[tuple[str, str]] = []
        for label, key in (
            ("Error", "error_description"), ("Cause", "root_cause"),
            ("Fix", "correction"), ("Next", "next_action"),
            ("Lesson", "reusable_experience"),
        ):
            value = str(reflection_dict.get(key, "")).strip()
            if value:
                fields.append((label, value))
        if important_facts:
            fields.append(("Facts", " | ".join(important_facts)))
        if entities:
            fields.append(("Entities", ", ".join(entities)))
        if constraints:
            fields.append(("Constraints", " | ".join(constraints)))
        if evidence_refs:
            fields.append(("Evidence", ", ".join(map(str, evidence_refs))))
        optimized = "\n".join(f"{label}: {value}" for label, value in fields)
        if not optimized:
            optimized = " ".join(source_sentences) or original_information.strip()

        lower_optimized = optimized.casefold()
        constraints_preserved = all(item.casefold() in lower_optimized for item in constraints)
        entities_preserved = all(item.casefold() in lower_optimized for item in entities)
        facts_preserved = all(item.casefold() in lower_optimized for item in important_facts)
        valid = constraints_preserved and entities_preserved and facts_preserved
        reflection_source = " ".join(
            str(reflection_dict.get(key, ""))
            for key in (
                "error_description", "root_cause", "failed_decision",
                "correction", "next_action", "reusable_experience", "summary",
            )
        )
        source_bundle = " ".join([
            original_information, reflection_source,
            " ".join(task_constraints), " ".join(map(str, evidence_refs)),
        ])
        before = lexical_token_count(source_bundle)
        after = lexical_token_count(optimized)
        ratio = after / max(before, 1)
        stats = {
            "token_measure": "lexical_proxy",
            "tokens_before_scope": "original information + reflection fields + constraints + evidence refs",
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": max(0, before - after),
            "compression_ratio": ratio,
            "important_facts_preserved": facts_preserved,
            "constraints_preserved": constraints_preserved,
            "key_entities_preserved": entities_preserved,
            "optimization_valid": valid,
        }
        return OptimizationResult(
            original_information=original_information,
            optimized_information=optimized,
            important_facts=important_facts,
            key_entities=entities,
            constraints=constraints,
            optimization_valid=valid,
            statistics=stats,
        )
