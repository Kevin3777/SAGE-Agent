from __future__ import annotations

import math

from sage.agents.agent import SageAgent
from sage.backends.mock import MockBackend
from sage.core.clock import FakeClock
from sage.environments import MultiHopQAEnvironment
from sage.memory.manager import MemoryManager
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory


def test_failed_answer_reflection_ltm_persistence_and_cross_task_reuse(tmp_path):
    clock = FakeClock(50_000.0)
    db_path = tmp_path / "cross-session.sqlite3"
    first_store = MemoryStore(db_path, clock)
    first_manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(first_store), clock=clock,
    )
    environment = MultiHopQAEnvironment()
    first_task = environment.get_task("qa-ember-city")
    first_backend = MockBackend(['{"answer":"Auckland","evidence":["p1","p2"]}'])
    first_result = SageAgent(
        first_backend, environment, memory_manager=first_manager, clock=clock,
    ).run(first_task, mode="full", max_iterations=1)
    assert not first_result["success"]
    assert first_result["reflection_count"] == 1
    assert first_result["metrics"]["memory_write_count"] == 1

    source_memory = first_manager.stm.memories()[0]
    clock.advance(-source_memory.optimized_strength * math.log(0.69))
    lifecycle = first_manager.tick()
    migration = next(event for event in lifecycle if event["action"] == "migrate_to_ltm")
    assert migration["retention_at_transition"] == pytest_approx(0.69)
    assert first_store.get(source_memory.memory_id) is not None

    # A fresh manager/store instance demonstrates that LTM survives a new session.
    reloaded_store = MemoryStore(db_path, clock)
    restored = reloaded_store.get(source_memory.memory_id)
    assert restored is not None
    second_manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(reloaded_store), clock=clock,
    )
    second_task = environment.get_task("qa-atlas-founder")
    second_backend = MockBackend(['{"answer":"Imani Brooks","evidence":["p1","p2"]}'])
    second_result = SageAgent(
        second_backend, environment, memory_manager=second_manager, clock=clock,
    ).run(second_task, mode="full", max_iterations=2)
    assert second_result["success"]
    selected = second_result["iterations"][0]["state_before_action"]["retrieved_memory_ids"]
    assert restored.memory_id in selected
    assert "Retrieved past experience" in second_backend.calls[0]["messages"][-1]["content"]
    assert restored.optimized_information in second_backend.calls[0]["messages"][-1]["content"]


def pytest_approx(value):
    # Avoid an additional import in the test helper list; this remains a numeric assertion.
    from pytest import approx
    return approx(value)
