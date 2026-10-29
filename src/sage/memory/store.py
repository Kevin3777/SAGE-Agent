"""Bounded STM and SQLite-backed persistent long-term memory."""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from sage.core.clock import Clock, SystemClock
from sage.core.types import MemoryRecord, MemoryState, Trajectory


class ShortTermMemory:
    def __init__(self, capacity: int = 64, trajectory_capacity: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("STM capacity must be at least one")
        self.capacity = int(capacity)
        self.trajectory_capacity = int(trajectory_capacity or capacity * 4)
        if self.trajectory_capacity < 1:
            raise ValueError("STM trajectory capacity must be at least one")
        self._memories: deque[MemoryRecord] = deque()
        self._trajectories: deque[Trajectory] = deque(maxlen=self.trajectory_capacity)
        self.trajectory_evictions = 0

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord | None:
        evicted = self._memories.popleft() if len(self._memories) >= self.capacity else None
        self._memories.append(memory)
        return evicted

    def remove_memory(self, memory_id: str) -> MemoryRecord | None:
        for item in list(self._memories):
            if item.memory_id == memory_id:
                self._memories.remove(item)
                return item
        return None

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return next((item for item in self._memories if item.memory_id == memory_id), None)

    def memories(self, task_type: str | None = None) -> list[MemoryRecord]:
        rows = list(self._memories)
        return [row for row in rows if task_type is None or row.task_type == task_type]

    def add_trajectory(self, trajectory: Trajectory) -> None:
        if len(self._trajectories) == self.trajectory_capacity:
            self.trajectory_evictions += 1
        self._trajectories.append(trajectory)

    def recent(self, task_id: str | None = None, limit: int = 8) -> list[Trajectory]:
        rows = list(self._trajectories)
        if task_id is not None:
            rows = [row for row in rows if row.task_id == task_id]
        return rows[-max(0, int(limit)):] if limit else []

    def for_task(self, task_id: str) -> list[Trajectory]:
        return [row for row in self._trajectories if row.task_id == task_id]

    def context(self, task_id: str | None = None, limit: int = 4) -> str:
        return "\n".join(
            f"- {item.task_id}: action={item.action!r}; feedback={item.checker_feedback.get('message', '')}; reward={item.reward:g}"
            for item in self.recent(task_id=task_id, limit=limit)
        )


class MemoryStore:
    """SQLite persistence for active LTM records; deleted records leave no searchable row."""

    def __init__(self, db_path: str | Path, clock: Clock | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    task_type TEXT NOT NULL,
                    original_information TEXT NOT NULL,
                    optimized_information TEXT NOT NULL,
                    reflection_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    initial_strength REAL NOT NULL CHECK(initial_strength > 0),
                    optimized_strength REAL NOT NULL CHECK(optimized_strength > 0),
                    strength_unit TEXT NOT NULL DEFAULT 'seconds',
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    state_since REAL NOT NULL,
                    memory_state TEXT NOT NULL,
                    retention_rate REAL NOT NULL CHECK(retention_rate >= 0 AND retention_rate <= 1),
                    reinforcement_count INTEGER NOT NULL,
                    optimization_stats_json TEXT NOT NULL
                )
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(memories)")}
            if "strength_unit" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN strength_unit TEXT NOT NULL DEFAULT 'seconds'"
                )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _fingerprint(record: MemoryRecord) -> str:
        import hashlib
        import re

        canonical = re.sub(r"\W+", " ", record.optimized_information.casefold()).strip()
        return hashlib.sha256(f"{record.task_type}\n{canonical}".encode("utf-8")).hexdigest()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"], task_type=row["task_type"],
            original_information=row["original_information"],
            optimized_information=row["optimized_information"],
            reflection=json.loads(row["reflection_json"]),
            evidence_refs=json.loads(row["evidence_refs_json"]),
            initial_strength=row["initial_strength"],
            optimized_strength=row["optimized_strength"], created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            memory_state=MemoryState(row["memory_state"]),
            retention_rate=row["retention_rate"],
            reinforcement_count=row["reinforcement_count"], state_since=row["state_since"],
            optimization_stats=json.loads(row["optimization_stats_json"]),
            strength_unit=row["strength_unit"],
        )

    def _write(self, connection: sqlite3.Connection, record: MemoryRecord, fingerprint: str) -> None:
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, fingerprint, task_type, original_information,
                optimized_information, reflection_json, evidence_refs_json,
                initial_strength, optimized_strength, strength_unit, created_at, last_accessed_at,
                state_since, memory_state, retention_rate, reinforcement_count,
                optimization_stats_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                task_type=excluded.task_type,
                original_information=excluded.original_information,
                optimized_information=excluded.optimized_information,
                reflection_json=excluded.reflection_json,
                evidence_refs_json=excluded.evidence_refs_json,
                initial_strength=excluded.initial_strength,
                optimized_strength=excluded.optimized_strength,
                strength_unit=excluded.strength_unit,
                last_accessed_at=excluded.last_accessed_at,
                state_since=excluded.state_since,
                memory_state=excluded.memory_state,
                retention_rate=excluded.retention_rate,
                reinforcement_count=excluded.reinforcement_count,
                optimization_stats_json=excluded.optimization_stats_json
            """,
            (
                record.memory_id, fingerprint, record.task_type,
                record.original_information, record.optimized_information,
                json.dumps(record.reflection, ensure_ascii=False, sort_keys=True),
                json.dumps(record.evidence_refs, ensure_ascii=False),
                record.initial_strength, record.optimized_strength, record.strength_unit,
                record.created_at, record.last_accessed_at,
                record.state_since or record.created_at,
                record.memory_state.value, record.retention_rate,
                record.reinforcement_count,
                json.dumps(record.optimization_stats, ensure_ascii=False, sort_keys=True),
            ),
        )

    def insert(self, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        fingerprint = self._fingerprint(record)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM memories WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existing:
                merged = self._from_row(existing)
                merged.reinforcement_count += 1
                merged.last_accessed_at = self.clock.now()
                merged.state_since = self.clock.now()
                merged.retention_rate = 1.0
                self._write(connection, merged, fingerprint)
                return merged, False
            record.memory_state = MemoryState.LTM
            record.state_since = record.state_since if record.state_since is not None else self.clock.now()
            self._write(connection, record, fingerprint)
            return record, True

    def save_active(self, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        """Persist an STM or LTM state without promoting it implicitly."""
        fingerprint = self._fingerprint(record)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (record.memory_id,)
            ).fetchone()
            if existing:
                self._write(connection, record, fingerprint)
                return record, True
            duplicate = connection.execute(
                "SELECT * FROM memories WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if duplicate:
                return self._from_row(duplicate), False
            self._write(connection, record, fingerprint)
            return record, True

    def update(self, record: MemoryRecord) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT memory_id FROM memories WHERE memory_id=?", (record.memory_id,)).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {record.memory_id}")
            self._write(connection, record, self._fingerprint(record))

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
            return cursor.rowcount > 0

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        return self._from_row(row) if row else None

    def list(self, states: Iterable[MemoryState] | None = None) -> list[MemoryRecord]:
        with self._connect() as connection:
            if states:
                values = [state.value for state in states]
                placeholders = ",".join("?" for _ in values)
                rows = connection.execute(
                    f"SELECT * FROM memories WHERE memory_state IN ({placeholders}) ORDER BY created_at, memory_id",
                    values,
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM memories ORDER BY created_at, memory_id").fetchall()
        return [self._from_row(row) for row in rows]


class LongTermMemory:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def insert(self, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        return self.store.insert(record)

    def save_active(self, record: MemoryRecord) -> tuple[MemoryRecord, bool]:
        return self.store.save_active(record)

    def update(self, record: MemoryRecord) -> None:
        self.store.update(record)

    def delete(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def all(self) -> list[MemoryRecord]:
        return self.store.list(states=[MemoryState.LTM])
