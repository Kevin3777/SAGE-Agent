from __future__ import annotations

import sqlite3

import pytest

from sage.environments import SQLAgentEnvironment


def test_sql_agent_uses_readonly_connection_and_rejects_delete(tmp_path):
    environment = SQLAgentEnvironment(tmp_path / "safe.sqlite3")
    task = environment.get_task()
    feedback = environment.check(task, {"sql": "DELETE FROM employees"})
    assert feedback.success is False
    assert "unsafe SQL was rejected" in feedback.message or "readonly" in feedback.message.casefold()
    columns, rows = environment.execute_readonly(
        task.payload["db_path"], "SELECT COUNT(*) AS n FROM employees",
    )
    assert columns == ["n"]
    assert rows == [[4]]


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "PRAGMA user_version",
    "ATTACH DATABASE ':memory:' AS other",
    "SELECT 1; DELETE FROM employees",
])
def test_mutating_or_database_control_sql_is_rejected(tmp_path, sql):
    environment = SQLAgentEnvironment(tmp_path / "readonly.sqlite3")
    with pytest.raises(sqlite3.DatabaseError):
        environment.execute_readonly(environment.db_path, sql)


def test_real_join_query_returns_expected_rows(tmp_path):
    environment = SQLAgentEnvironment(tmp_path / "join.sqlite3")
    task = environment.get_task()
    query = (
        "SELECT e.name AS name, d.name AS department FROM employees e "
        "JOIN departments d ON d.department_id=e.department_id "
        "JOIN assignments a ON a.employee_id=e.employee_id "
        "JOIN projects p ON p.project_id=a.project_id "
        "WHERE e.active=1 AND p.name='Apollo' ORDER BY e.name"
    )
    feedback = environment.check(task, {"sql": query})
    assert feedback.success
