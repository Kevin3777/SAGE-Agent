"""Threshold-driven STM/LTM lifecycle and relevance-aware retrieval."""
from __future__ import annotations

import re
from typing import Any

from sage.core.clock import Clock, SystemClock
from sage.core.types import MemoryRecord, MemoryState, new_id

from .forgetting import ForgettingCurve
from .store import LongTermMemory, ShortTermMemory
from .syntax import MemorySyntax, StrengthEstimator


class ThresholdPolicy:
    """Implements the paper's prose interpretation of Eq. 8."""

    def __init__(self, theta1: float = 0.7, theta2: float = 0.3) -> None:
        if not (0.0 <= theta2 < theta1 <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= theta2 < theta1 <= 1")
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)

    def classify(self, retention: float) -> MemoryState:
        if not 0.0 <= float(retention) <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        if retention >= self.theta1:
            return MemoryState.STM
        if retention >= self.theta2:
            return MemoryState.LTM
        return MemoryState.FORGOTTEN


class MemoryManager:
    def __init__(
        self, stm: ShortTermMemory, ltm: LongTermMemory, *,
        clock: Clock | None = None, curve: ForgettingCurve | None = None,
        thresholds: ThresholdPolicy | None = None,
    ) -> None:
        self.stm = stm
        self.ltm = ltm
        self.clock = clock or SystemClock()
        self.curve = curve or ForgettingCurve()
        self.strength_estimator = StrengthEstimator(time_unit=self.curve.time_unit)
        self.thresholds = thresholds or ThresholdPolicy()
        self.events: list[dict[str, Any]] = []
        self.counters = {
            "memory_migration_count": 0,
            "memory_forget_count": 0,
            "memory_write_count": 0,
            "memory_retrieval_count": 0,
        }
        # STM is still short-lived working state, but persisting its timestamped
        # rows lets the next process resume decay and promote them to LTM.
        for memory in self.ltm.store.list(states=[MemoryState.STM]):
            evicted = self.stm.add_memory(memory)
            if evicted is not None:
                self._promote(evicted, reason="restore_capacity")

    def add(self, memory: MemoryRecord) -> None:
        memory.memory_state = MemoryState.STM
        memory.state_since = self.clock.now()
        stored, inserted = self.ltm.save_active(memory)
        if not inserted:
            self.events.append({
                "action": "deduplicate_memory", "memory_id": stored.memory_id,
                "state": stored.memory_state.value, "at": self.clock.now(),
            })
            self.reinforce(stored)
            return
        evicted = self.stm.add_memory(memory)
        self.events.append({"action": "write_stm", "memory_id": memory.memory_id, "at": self.clock.now()})
        self.counters["memory_write_count"] += 1
        if evicted is not None:
            self._promote(evicted, reason="stm_capacity")

    def _promote(self, memory: MemoryRecord, reason: str, retention: float | None = None) -> None:
        previous = memory.memory_state
        memory.memory_state = MemoryState.LTM
        # Engineering assumption: promotion starts an LTM decay epoch. The raw
        # Eq. 6 strength changes from S* (STM) to S (LTM), as Eq. 7 indicates.
        memory.state_since = self.clock.now()
        memory.last_accessed_at = self.clock.now()
        if retention is not None:
            memory.retention_rate = float(retention)
        existing = self.ltm.get(memory.memory_id)
        if existing is not None:
            self.ltm.update(memory)
            stored, inserted = memory, True
        else:
            stored, inserted = self.ltm.insert(memory)
        self.counters["memory_migration_count"] += 1
        self.events.append({
            "action": "migrate_to_ltm", "memory_id": stored.memory_id,
            "from": previous.value, "to": MemoryState.LTM.value,
            "reason": reason, "retention_at_transition": retention,
            "deduplicated": not inserted, "at": self.clock.now(),
        })

    def tick(self) -> list[dict[str, Any]]:
        now = self.clock.now()
        emitted: list[dict[str, Any]] = []
        preexisting_ltm = {memory.memory_id: memory for memory in self.ltm.all()}
        for memory in list(self.stm.memories()):
            started = memory.state_since if memory.state_since is not None else memory.created_at
            curve = self.curve if memory.strength_unit == self.curve.time_unit else ForgettingCurve(memory.strength_unit)
            memory.retention_rate = curve.since(started, now, memory.optimized_strength)
            state = self.thresholds.classify(memory.retention_rate)
            if state == MemoryState.LTM:
                self.stm.remove_memory(memory.memory_id)
                self._promote(memory, reason="retention_threshold", retention=memory.retention_rate)
                emitted.append(self.events[-1])
            elif state == MemoryState.FORGOTTEN:
                self.stm.remove_memory(memory.memory_id)
                memory.memory_state = MemoryState.FORGOTTEN
                self.counters["memory_forget_count"] += 1
                event = {
                    "action": "forget", "memory_id": memory.memory_id,
                    "from": MemoryState.STM.value, "to": MemoryState.FORGOTTEN.value,
                    "retention": memory.retention_rate, "at": now,
                }
                self.events.append(event)
                emitted.append(event)
            else:
                self.ltm.update(memory)
                emitted.append({
                    "action": "retain_stm", "memory_id": memory.memory_id,
                    "retention": memory.retention_rate, "at": now,
                })
        for memory in preexisting_ltm.values():
            started = memory.state_since if memory.state_since is not None else memory.last_accessed_at
            curve = self.curve if memory.strength_unit == self.curve.time_unit else ForgettingCurve(memory.strength_unit)
            memory.retention_rate = curve.since(started, now, memory.initial_strength)
            state = self.thresholds.classify(memory.retention_rate)
            if state == MemoryState.FORGOTTEN:
                self.ltm.delete(memory.memory_id)
                memory.memory_state = MemoryState.FORGOTTEN
                self.counters["memory_forget_count"] += 1
                event = {
                    "action": "delete_ltm", "memory_id": memory.memory_id,
                    "from": MemoryState.LTM.value, "to": MemoryState.FORGOTTEN.value,
                    "retention": memory.retention_rate, "at": now,
                }
                self.events.append(event)
                emitted.append(event)
            else:
                self.ltm.update(memory)
                emitted.append({
                    "action": "retain_ltm", "memory_id": memory.memory_id,
                    "retention": memory.retention_rate, "at": now,
                })
        return emitted

    def reinforce(self, memory: MemoryRecord) -> None:
        now = self.clock.now()
        memory.reinforcement_count += 1
        memory.last_accessed_at = now
        memory.state_since = now
        memory.retention_rate = 1.0
        if memory.memory_state == MemoryState.STM:
            memory.optimized_strength *= 1.05
            self.ltm.update(memory)
        else:
            memory.initial_strength *= 1.05
            memory.optimized_strength = max(memory.optimized_strength, memory.initial_strength)
            self.ltm.update(memory)
        self.events.append({
            "action": "reinforce", "memory_id": memory.memory_id,
            "reinforcement_count": memory.reinforcement_count, "at": now,
        })


def _terms(text: str) -> set[str]:
    return {
        token.casefold() for token in re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", text)
        if len(token) > 1
    }


class MemoryRetriever:
    def __init__(self, manager: MemoryManager) -> None:
        self.manager = manager

    def retrieve(self, query: str, task_type: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        self.manager.tick()
        candidates = self.manager.stm.memories() + self.manager.ltm.all()
        query_terms = _terms(query)
        ranked: list[dict[str, Any]] = []
        for memory in candidates:
            memory_terms = _terms(memory.optimized_information)
            overlap = len(query_terms & memory_terms)
            relevance = overlap / max(1, len(query_terms | memory_terms))
            if task_type and memory.task_type == task_type:
                relevance = min(1.0, relevance + 0.25)
            if relevance <= 0:
                continue
            combined = 0.7 * relevance + 0.3 * memory.retention_rate
            ranked.append({
                "record": memory, "relevance": relevance,
                "retention": memory.retention_rate, "rank_score": combined,
            })
        ranked.sort(key=lambda item: (item["rank_score"], item["relevance"], item["retention"]), reverse=True)
        selected = ranked[:max(0, int(limit))]
        for item in selected:
            self.manager.reinforce(item["record"])
        self.manager.counters["memory_retrieval_count"] += 1
        return selected


def create_memory_record(
    *, task_type: str, original_information: str, reflection: dict[str, Any],
    checker_quality: float, task_constraints: list[str], evidence_refs: list[str],
    clock: Clock | None = None, syntax: MemorySyntax | None = None,
    estimator: StrengthEstimator | None = None,
) -> MemoryRecord:
    clock = clock or SystemClock()
    syntax = syntax or MemorySyntax()
    estimator = estimator or StrengthEstimator()
    optimized = syntax.optimize(
        original_information, reflection, task_constraints=task_constraints,
        evidence_refs=evidence_refs,
    )
    confidence = float(reflection.get("confidence", 0.5))
    base, enhanced, strength_score = estimator.estimate(
        checker_quality=max(0.0, min(1.0, checker_quality)),
        key_constraint=bool(task_constraints),
        task_importance=0.75 if task_constraints else 0.55,
        reusability=min(1.0, max(0.0, confidence)),
        evidence_count=len(evidence_refs),
        optimization_valid=optimized.optimization_valid,
        compression_ratio=optimized.statistics["compression_ratio"],
    )
    now = clock.now()
    stats = optimized.statistics | {
        "strength_score": strength_score,
        "strength_unit": estimator.time_unit,
        "strength_formula": "0.30 checker + 0.15 constraint + 0.20 importance + "
        "0.20 reuse + 0.10 evidence + 0.05 past reuse; engineering assumption",
        "base_strength": base,
        "optimized_strength": enhanced,
    }
    return MemoryRecord(
        memory_id=new_id("mem"), task_type=task_type,
        original_information=original_information,
        optimized_information=optimized.optimized_information,
        reflection=reflection, evidence_refs=list(evidence_refs),
        initial_strength=base, optimized_strength=enhanced,
        created_at=now, last_accessed_at=now,
        memory_state=MemoryState.STM, retention_rate=1.0,
        reinforcement_count=0, state_since=now, optimization_stats=stats,
        strength_unit=estimator.time_unit,
    )
