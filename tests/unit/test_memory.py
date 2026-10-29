from __future__ import annotations

from sage.core.types import MemoryRecord, MemoryState, Trajectory, new_id
from sage.memory.manager import MemoryRetriever, create_memory_record
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory
from sage.memory.syntax import MemorySyntax, StrengthEstimator, lexical_token_count


def sample_record(clock, text, task_type="multi_hop_qa"):
    now = clock.now()
    return MemoryRecord(
        memory_id=new_id("mem"), task_type=task_type,
        original_information=text, optimized_information=text,
        reflection={"reusable_experience": text}, evidence_refs=["p1"],
        initial_strength=100.0, optimized_strength=140.0,
        created_at=now, last_accessed_at=now, memory_state=MemoryState.LTM,
        retention_rate=1.0, reinforcement_count=0, state_since=now,
    )


def test_stm_capacity_and_recent_trajectory(memory_stack, fake_clock):
    stm = memory_stack.stm
    for i in range(3):
        stm.add_trajectory(Trajectory(
            task_id=f"t{i}", observation="question", action="answer",
            assistant_output="answer", checker_feedback={"message": "ok"},
            reward=1, timestamp=fake_clock.now(),
        ))
    assert [item.task_id for item in stm.recent(limit=2)] == ["t1", "t2"]
    assert stm.context(limit=1).startswith("- t2:")


def test_ltm_persists_across_store_instances(tmp_path, fake_clock):
    first_store = MemoryStore(tmp_path / "persistent.db", fake_clock)
    memory = sample_record(fake_clock, "Atlas Notes links to Northstar Labs.")
    first_store.insert(memory)
    second_store = MemoryStore(tmp_path / "persistent.db", fake_clock)
    restored = second_store.get(memory.memory_id)
    assert restored is not None
    assert restored.memory_state == MemoryState.LTM
    assert restored.optimized_information == memory.optimized_information


def test_strength_unit_persists_in_sqlite(tmp_path, fake_clock):
    store = MemoryStore(tmp_path / "hours.db", fake_clock)
    memory = sample_record(fake_clock, "A lesson with hour units.")
    memory.strength_unit = "hours"
    store.insert(memory)
    restored = MemoryStore(tmp_path / "hours.db", fake_clock).get(memory.memory_id)
    assert restored.strength_unit == "hours"


def test_duplicate_memory_is_deduplicated(tmp_path, fake_clock):
    store = MemoryStore(tmp_path / "dedupe.db", fake_clock)
    first = sample_record(fake_clock, "The company is Northstar Labs.")
    duplicate = sample_record(fake_clock, "The company is Northstar Labs.")
    stored_first, inserted_first = store.insert(first)
    stored_second, inserted_second = store.insert(duplicate)
    assert inserted_first is True
    assert inserted_second is False
    assert stored_first.memory_id == stored_second.memory_id
    assert len(store.list()) == 1
    assert stored_second.reinforcement_count == 1


def test_retrieval_ranks_relevant_memory_and_reinforces(memory_stack, fake_clock):
    relevant = sample_record(fake_clock, "Atlas Notes was created by Northstar Labs; cite both passages.")
    unrelated = sample_record(fake_clock, "SQLite projects use a read-only join.", task_type="sqlite_sql")
    memory_stack.ltm.insert(relevant)
    memory_stack.ltm.insert(unrelated)
    retrieved = MemoryRetriever(memory_stack).retrieve("Atlas Notes Northstar Labs", "multi_hop_qa", limit=2)
    assert retrieved[0]["record"].memory_id == relevant.memory_id
    touched = memory_stack.ltm.get(relevant.memory_id)
    assert touched.reinforcement_count == 1
    assert touched.last_accessed_at == fake_clock.now()


def test_deleted_memory_cannot_be_retrieved(memory_stack, fake_clock):
    memory = sample_record(fake_clock, "Atlas Notes Northstar Labs.")
    memory_stack.ltm.insert(memory)
    assert memory_stack.ltm.delete(memory.memory_id) is True
    assert memory_stack.ltm.get(memory.memory_id) is None
    assert MemoryRetriever(memory_stack).retrieve("Atlas Notes Northstar Labs", "multi_hop_qa") == []


def test_memory_syntax_extracts_and_preserves_facts_entities_constraints():
    source = (
        "For Atlas Notes, follow the product to Northstar Labs. "
        "The failed decision omitted passage p2. "
    ) * 4
    reflection = {
        "error_description": "The answer missed passage p2.",
        "root_cause": "The company link was not checked.",
        "correction": "Use Northstar Labs to resolve the link.",
        "next_action": "Cite p1 and p2.",
        "reusable_experience": "Preserve both links in multi-hop answers.",
    }
    result = MemorySyntax().optimize(
        source, reflection, ["Must cite both p1 and p2."], ["p1", "p2"],
    )
    assert result.optimization_valid
    assert "Atlas Notes" in result.optimized_information
    assert "Northstar Labs" in result.optimized_information
    assert "p2" in result.optimized_information
    assert "Must cite both p1 and p2." in result.optimized_information
    assert result.statistics["tokens_after"] < result.statistics["tokens_before"]
    assert result.statistics["tokens_saved"] > 0


def test_strength_estimator_normalizes_and_only_boosts_valid_compression():
    estimator = StrengthEstimator()
    score = estimator.score(
        checker_quality=0.8, key_constraint=True, task_importance=0.7,
        reusability=0.9, evidence_count=4, reuse_count=2,
    )
    base, optimized, normalized = estimator.estimate(
        checker_quality=0.8, key_constraint=True, task_importance=0.7,
        reusability=0.9, evidence_count=4, optimization_valid=True,
        compression_ratio=0.5, reuse_count=2,
    )
    assert score == normalized
    assert 0 <= score <= 1
    assert optimized > base
    assert estimator.estimate(
        checker_quality=0.8, key_constraint=True, task_importance=0.7,
        reusability=0.9, evidence_count=4, optimization_valid=False,
        compression_ratio=0.5,
    )[1] == estimator.estimate(
        checker_quality=0.8, key_constraint=True, task_importance=0.7,
        reusability=0.9, evidence_count=4, optimization_valid=False,
        compression_ratio=0.5,
    )[0]


def test_strength_estimator_returns_configured_time_unit():
    estimator = StrengthEstimator(time_unit="hours")
    base, optimized, _ = estimator.estimate(
        checker_quality=0.8, key_constraint=True, task_importance=0.7,
        reusability=0.9, evidence_count=3, optimization_valid=True,
        compression_ratio=0.5,
    )
    assert 0 < base < 1
    assert optimized > base


def test_agent_memory_record_keeps_both_versions(fake_clock):
    reflection = {
        "error_description": "wrong answer", "root_cause": "missing entity link",
        "correction": "check evidence", "next_action": "cite p1 and p2",
        "reusable_experience": "follow both links", "confidence": 0.8,
        "evidence_refs": ["p1", "p2"],
    }
    record = create_memory_record(
        task_type="multi_hop_qa",
        original_information=("Atlas Notes links to Northstar Labs. " * 8),
        reflection=reflection, checker_quality=0.8,
        task_constraints=["Must cite p1 and p2."],
        evidence_refs=["p1", "p2"], clock=fake_clock,
    )
    assert record.original_information != record.optimized_information
    assert record.initial_strength > 0
    assert record.optimized_strength >= record.initial_strength
    assert record.current_strength == record.optimized_strength
    assert record.to_dict()["current_strength"] == record.optimized_strength
    assert record.optimization_stats["tokens_before"] > lexical_token_count(record.original_information)
