"""Read-only SQLite task environment with a real schema and authorizer."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sage.core.types import Feedback, TaskCase


def default_database_path() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate / "examples" / "databases" / "sage_demo.sqlite3"
    return current / "outputs" / "sage_demo.sqlite3"


def prepare_demo_database(path: str | Path | None = None) -> Path:
    database = Path(path) if path else default_database_path()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS departments (
                department_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS employees (
                employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(department_id),
                active INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS assignments (
                employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                hours REAL NOT NULL,
                PRIMARY KEY(employee_id, project_id)
            );
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO departments(department_id, name) VALUES (?, ?)",
            [(1, "Engineering"), (2, "Data"), (3, "Operations")],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO employees(employee_id, name, department_id, active) VALUES (?, ?, ?, ?)",
            [
                (1, "Mira Chen", 1, 1), (2, "Noah Patel", 2, 1),
                (3, "Lena Ortiz", 3, 1), (4, "Owen Gray", 1, 0),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO projects(project_id, name) VALUES (?, ?)",
            [(1, "Apollo"), (2, "Beacon")],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO assignments(employee_id, project_id, hours) VALUES (?, ?, ?)",
            [(1, 1, 12.5), (2, 1, 8.0), (3, 2, 5.0), (4, 1, 3.0)],
        )
        connection.commit()
    finally:
        connection.close()
    return database.resolve()


class SQLAgentEnvironment:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = prepare_demo_database(db_path)
        self.tasks = {
            "sql-apollo-team": TaskCase(
                task_id="sql-apollo-team", task_type="sqlite_sql",
                user_task=(
                    "List the names and department names of active employees assigned to the "
                    "Apollo project. Return columns name, department, ordered by employee name."
                ),
                instructions=(
                    "Write exactly one SQLite read-only SELECT query. Use the supplied schema. "
                    "Return SQL only; optional code fences are allowed. Never use DDL or DML."
                ),
                payload={
                    "db_path": str(self.db_path),
                    "schema": (
                        "employees(employee_id, name, department_id, active); "
                        "departments(department_id, name); "
                        "projects(project_id, name); "
                        "assignments(employee_id, project_id, hours)"
                    ),
                    "expected_columns": ["name", "department"],
                    "expected_rows": [["Mira Chen", "Engineering"], ["Noah Patel", "Data"]],
                },
                constraints=["Use a single read-only SELECT.", "Filter to active employees on Apollo."],
            )
        }

    def get_task(self, task_id: str = "sql-apollo-team") -> TaskCase:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown SQL task {task_id!r}; available: {', '.join(self.tasks)}") from exc

    @staticmethod
    def format_prompt(task: TaskCase) -> str:
        return f"Request: {task.user_task}\nSQLite schema: {task.payload['schema']}"

    @staticmethod
    def parse_output(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ValueError("empty SQL output")
        match_json = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match_json:
            try:
                value = json.loads(match_json.group(0))
                if isinstance(value, dict) and "sql" in value:
                    text = str(value["sql"])
            except json.JSONDecodeError:
                pass
        fenced = re.search(r"\x60{3}(?:sql)?\s*(.*?)\x60{3}", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        return {"sql": text.strip().rstrip(";")}

    @staticmethod
    def _readonly_authorizer(action: int, arg1: str | None, arg2: str | None, db: str | None, source: str | None) -> int:
        allowed = {
            sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
        }
        if hasattr(sqlite3, "SQLITE_RECURSIVE"):
            allowed.add(sqlite3.SQLITE_RECURSIVE)
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        function_name = (arg2 or arg1 or "").casefold()
        if action == sqlite3.SQLITE_FUNCTION and function_name in {"load_extension", "writefile", "readfile"}:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and db not in (None, "main"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def execute_readonly(self, db_path: str | Path, sql: str) -> tuple[list[str], list[list[Any]]]:
        path = Path(db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"demo SQLite file does not exist: {path}")
        uri = "file:" + quote(path.as_posix(), safe="/:")
        connection = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=2.0)
        try:
            connection.set_authorizer(self._readonly_authorizer)
            cursor = connection.execute(sql)
            columns = [item[0] for item in (cursor.description or [])]
            rows = [list(row) for row in cursor.fetchall()]
            if not columns:
                raise sqlite3.OperationalError("query returned no result columns")
            return columns, rows
        finally:
            connection.close()

    def check(self, task: TaskCase, candidate: Any) -> Feedback:
        if not isinstance(candidate, dict) or not str(candidate.get("sql", "")).strip():
            error = "output could not be parsed into a SQL query"
            return Feedback.create(
                False, 0.0, f"Malformed SQL: {error}. Return one SELECT query.",
                [error], [f"checker:{task.task_id}:format"],
            )
        query = str(candidate["sql"])
        try:
            columns, rows = self.execute_readonly(task.payload["db_path"], query)
        except sqlite3.DatabaseError as exc:
            detail = str(exc)
            if "not authorized" in detail.casefold():
                error = f"unsafe SQL was rejected by the SQLite read-only authorizer: {detail}"
            else:
                error = f"SQLite query error: {detail}"
            return Feedback.create(
                False, 0.0, error, [error], [f"checker:{task.task_id}:sqlite"],
            )
        except (OSError, ValueError) as exc:
            error = f"SQLite environment error: {exc}"
            return Feedback.create(False, 0.0, error, [error], [f"checker:{task.task_id}:environment"])
        wanted_columns = task.payload["expected_columns"]
        wanted_rows = task.payload["expected_rows"]
        if columns != wanted_columns:
            error = f"result columns mismatch: expected {wanted_columns}, got {columns}"
            return Feedback.create(
                False, 0.4 if rows == wanted_rows else 0.0, error, [error],
                [f"checker:{task.task_id}:columns"],
            )
        if rows != wanted_rows:
            error = f"result rows mismatch: expected {wanted_rows}, got {rows}"
            return Feedback.create(
                False, 0.5, error, [error], [f"checker:{task.task_id}:rows"],
            )
        return Feedback.create(
            True, 1.0, "Read-only SQL query returned the expected columns and rows.",
            [], [f"checker:{task.task_id}:result"],
        )
