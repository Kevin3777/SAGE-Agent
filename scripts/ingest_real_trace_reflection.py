"""Promote a real local-model reflection into persistent LTM for a follow-up run.

This utility only accepts a trace explicitly labeled backend_type=local. It
replays the trace's checked reflection through MemorySyntax and simulates STM
aging with FakeClock before persisting the resulting LTM record.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from sage.core.clock import FakeClock
from sage.memory.manager import MemoryManager, create_memory_record
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else project / "outputs" / "real_qa.json"
    database = Path(sys.argv[2]) if len(sys.argv) > 2 else project / "outputs" / "real_validation_memory.sqlite3"
    output = project / "outputs" / "real_memory_migration.json"
    trace = json.loads(source.read_text(encoding="utf-8"))
    if trace.get("backend", {}).get("backend_type") != "local":
        raise SystemExit("Refusing to ingest a trace that is not labeled backend_type=local.")
    failed = next(
        (
            row for row in trace.get("iterations", [])
            if not row.get("checker_feedback", {}).get("success") and row.get("reflection")
        ),
        None,
    )
    if failed is None:
        raise SystemExit("The real local trace has no checked failure with a validated reflection.")
    feedback = failed["checker_feedback"]
    reflection = failed["reflection"]
    if reflection.get("source_feedback_id") != feedback.get("feedback_id"):
        raise SystemExit("Reflection source feedback ID does not match the real checker record.")

    wall_now = time.time()
    clock = FakeClock(wall_now)
    store = MemoryStore(database, clock=clock)
    manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(store), clock=clock,
    )
    task = trace["task"]
    raw_information = (
        f"Task: {task['user_task']}\n"
        f"Assistant output from real model: {failed['raw_output']}\n"
        f"Checker feedback: {feedback['message']}\n"
        f"Checker errors: {'; '.join(feedback['errors'])}"
    )
    memory = create_memory_record(
        task_type=task["task_type"],
        original_information=raw_information,
        reflection=reflection,
        checker_quality=float(feedback["score"]),
        task_constraints=task.get("constraints", []),
        evidence_refs=feedback.get("evidence_refs", []),
        clock=clock,
    )
    manager.add(memory)
    migration_delay = -memory.optimized_strength * math.log(0.6)
    clock.advance(migration_delay)
    events = manager.tick()
    migration = next((event for event in events if event.get("action") == "migrate_to_ltm"), None)
    if migration is None:
        raise SystemExit("Fake-clock lifecycle did not produce the expected STM-to-LTM transition.")

    # Rebase the persisted LTM decay epoch to wall time after the virtual aging
    # demonstration so a subsequent normal CLI process can load it safely.
    persisted = store.get(memory.memory_id)
    if persisted is None:
        raise SystemExit("Migrated record was not persisted in SQLite.")
    persisted.state_since = time.time()
    persisted.last_accessed_at = persisted.state_since
    persisted.retention_rate = 1.0
    store.update(persisted)
    reloaded = MemoryStore(database).get(memory.memory_id)
    if reloaded is None:
        raise SystemExit("Persistent-memory reload check failed.")
    result = {
        "source_real_trace": str(source.resolve()),
        "source_backend_type": trace["backend"]["backend_type"],
        "source_model_identifier": trace["backend"]["model_identifier"],
        "source_feedback_id": feedback["feedback_id"],
        "source_reflection_method": reflection.get("method"),
        "memory_id": memory.memory_id,
        "sqlite_database": str(database.resolve()),
        "fake_clock_migration_event": migration,
        "reloaded_state": reloaded.memory_state.value,
        "optimized_information": reloaded.optimized_information,
        "optimization_stats": reloaded.optimization_stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"JSON written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
