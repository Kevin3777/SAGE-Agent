from __future__ import annotations

import math

import pytest

from sage.core.types import MemoryRecord, MemoryState, new_id
from sage.memory.forgetting import ForgettingCurve
from sage.memory.manager import MemoryManager, ThresholdPolicy
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory


def make_record(clock, initial=5.0, optimized=10.0):
    now = clock.now()
    return MemoryRecord(
        memory_id=new_id("mem"), task_type="unit",
        original_information="original", optimized_information="validated summary",
        reflection={"reusable_experience": "Check source evidence."},
        evidence_refs=["e1"], initial_strength=initial, optimized_strength=optimized,
        created_at=now, last_accessed_at=now, memory_state=MemoryState.STM,
        retention_rate=1.0, reinforcement_count=0, state_since=now,
    )


def test_eq6_numeric_value_and_zero_elapsed():
    assert ForgettingCurve.retention(0, 12) == 1.0
    assert ForgettingCurve.retention(30, 60) == pytest.approx(math.exp(-0.5))


def test_retention_decreases_with_time_and_increases_with_strength():
    values = [ForgettingCurve.retention(t, 10) for t in (0, 1, 3, 10, 30)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert ForgettingCurve.retention(10, 20) > ForgettingCurve.retention(10, 10)


def test_configured_time_unit_converts_system_clock_seconds():
    curve = ForgettingCurve("hours")
    assert curve.since(0, 3600, 1) == pytest.approx(math.exp(-1))


def test_memory_lifecycle_uses_each_record_strength_unit(fake_clock, tmp_path):
    curve = ForgettingCurve("hours")
    manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(MemoryStore(tmp_path / "hours.db", fake_clock)),
        clock=fake_clock, curve=curve,
    )
    now = fake_clock.now()
    memory = MemoryRecord(
        memory_id=new_id("mem"), task_type="unit",
        original_information="source", optimized_information="summary",
        reflection={"reusable_experience": "lesson"}, evidence_refs=["e1"],
        initial_strength=0.5, optimized_strength=1.0,
        created_at=now, last_accessed_at=now, state_since=now,
        strength_unit="hours",
    )
    manager.add(memory)
    fake_clock.advance(3600 * -math.log(0.5))
    events = manager.tick()
    assert next(event for event in events if event["action"] == "migrate_to_ltm")["retention_at_transition"] == pytest.approx(0.5)


@pytest.mark.parametrize(("elapsed", "strength"), [(-1, 2), (math.inf, 2), (1, 0), (1, math.nan)])
def test_invalid_curve_inputs_raise(elapsed, strength):
    with pytest.raises(ValueError):
        ForgettingCurve.retention(elapsed, strength)


@pytest.mark.parametrize(
    ("retention", "state"),
    [
        (0.7, MemoryState.STM), (0.3, MemoryState.LTM),
        (0.700001, MemoryState.STM), (0.299999, MemoryState.FORGOTTEN),
    ],
)
def test_threshold_boundaries(retention, state):
    assert ThresholdPolicy(0.7, 0.3).classify(retention) == state


def test_fake_clock_moves_memory_from_stm_to_ltm(memory_stack, fake_clock):
    memory = make_record(fake_clock)
    assert memory.current_strength == memory.optimized_strength
    memory_stack.add(memory)
    fake_clock.advance(-memory.optimized_strength * math.log(0.5))
    events = memory_stack.tick()
    assert any(event["action"] == "migrate_to_ltm" for event in events)
    saved = memory_stack.ltm.get(memory.memory_id)
    assert saved is not None
    assert saved.memory_state == MemoryState.LTM
    assert saved.current_strength == saved.initial_strength
    assert saved.to_dict()["current_strength"] == saved.initial_strength
    assert memory_stack.stm.get_memory(memory.memory_id) is None


def test_fake_clock_deletes_ltm_below_theta2(memory_stack, fake_clock):
    memory = make_record(fake_clock)
    memory_stack.add(memory)
    fake_clock.advance(-memory.optimized_strength * math.log(0.5))
    memory_stack.tick()
    fake_clock.advance(-memory.initial_strength * math.log(0.2))
    events = memory_stack.tick()
    assert any(event["action"] == "delete_ltm" for event in events)
    assert memory_stack.ltm.get(memory.memory_id) is None
    assert memory_stack.counters["memory_forget_count"] == 1


def test_high_retention_is_kept_in_stm(memory_stack, fake_clock):
    memory = make_record(fake_clock)
    memory_stack.add(memory)
    fake_clock.advance(1.0)
    events = memory_stack.tick()
    assert any(event["action"] == "retain_stm" for event in events)
    assert memory_stack.stm.get_memory(memory.memory_id) is memory


def test_capacity_overflow_promotes_oldest_to_ltm(fake_clock, tmp_path):
    manager = MemoryManager(
        ShortTermMemory(capacity=1), LongTermMemory(MemoryStore(tmp_path / "capacity.db", fake_clock)),
        clock=fake_clock,
    )
    first, second = make_record(fake_clock), make_record(fake_clock)
    second.optimized_information = "a distinct validated summary"
    manager.add(first)
    manager.add(second)
    assert manager.stm.get_memory(first.memory_id) is None
    assert manager.ltm.get(first.memory_id) is not None
    assert manager.stm.get_memory(second.memory_id) is second


def test_unexpired_stm_restores_and_can_migrate_in_new_process(fake_clock, tmp_path):
    path = tmp_path / "stm-restart.db"
    first = MemoryManager(
        ShortTermMemory(capacity=4), LongTermMemory(MemoryStore(path, fake_clock)),
        clock=fake_clock,
    )
    memory = make_record(fake_clock)
    first.add(memory)
    restarted = MemoryManager(
        ShortTermMemory(capacity=4), LongTermMemory(MemoryStore(path, fake_clock)),
        clock=fake_clock,
    )
    assert restarted.stm.get_memory(memory.memory_id) is not None
    fake_clock.advance(-memory.optimized_strength * math.log(0.5))
    events = restarted.tick()
    assert any(event["action"] == "migrate_to_ltm" for event in events)
    assert restarted.ltm.get(memory.memory_id).memory_state == MemoryState.LTM
