from __future__ import annotations

import pytest

from sage.core.clock import FakeClock
from sage.environments import MultiHopQAEnvironment
from sage.memory.manager import MemoryManager
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory


@pytest.fixture
def qa_env():
    return MultiHopQAEnvironment()


@pytest.fixture
def fake_clock():
    return FakeClock(10_000.0)


@pytest.fixture
def memory_stack(tmp_path, fake_clock):
    store = MemoryStore(tmp_path / "memories.sqlite3", clock=fake_clock)
    manager = MemoryManager(
        ShortTermMemory(capacity=4, trajectory_capacity=8),
        LongTermMemory(store), clock=fake_clock,
    )
    return manager
