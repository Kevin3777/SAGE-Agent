from __future__ import annotations

from sage.agents.agent import SageAgent
from sage.backends.local import LocalModelBackend
from sage.backends.mock import MockBackend
from sage.core.clock import FakeClock
from sage.core.types import Feedback
from sage.environments import SQLAgentEnvironment
from sage.memory.manager import MemoryManager
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory
from sage.reflection.generator import ReflectionGenerator
from sage.reflection.validator import ReflectionValidationError, ReflectionValidator


def test_reflection_is_structured_and_evidence_grounded(qa_env):
    task = qa_env.get_task()
    failed = qa_env.check(task, {"answer": "Auckland", "evidence": ["p1", "p2"]})
    reflection = ReflectionGenerator().generate(
        task, ['{"answer":"Auckland"}'], [failed], [failed.reward], [],
    )
    assert reflection.error_description
    assert reflection.root_cause
    assert reflection.correction
    assert reflection.next_action
    assert reflection.reusable_experience
    assert reflection.evidence_refs == ["p1", "p2"]
    assert reflection.to_dict()["source_feedback_id"] == failed.feedback_id


def test_reflection_validator_rejects_unsupported_evidence(qa_env):
    task = qa_env.get_task()
    failed = qa_env.check(task, {"answer": "Auckland", "evidence": ["p1", "p2"]})
    reflection = ReflectionGenerator().generate(task, ["bad"], [failed], [-1], [])
    reflection.evidence_refs = ["invented-source"]
    try:
        ReflectionValidator().validate(
            reflection, source_feedback_id=failed.feedback_id,
            evidence_refs=failed.evidence_refs, checker_errors=failed.errors,
        )
    except ReflectionValidationError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("unsupported reflection evidence was accepted")


def test_checker_detects_wrong_answer_and_missing_citations(qa_env):
    task = qa_env.get_task()
    answer_only = qa_env.check(task, {"answer": "Wellington", "evidence": ["p1"]})
    wrong_answer = qa_env.check(task, {"answer": "Auckland", "evidence": ["p1", "p2"]})
    assert not answer_only.success and "missing required" in answer_only.message
    assert not wrong_answer.success and "answer mismatch" in wrong_answer.message
    assert answer_only.score > wrong_answer.score


def test_qa_parser_extracts_passage_ids_from_citation_text(qa_env):
    parsed = qa_env.parse_output(
        '{"answer":"Wellington","evidence":["p2: Cedar Labs is based there", "p1"]}'
    )
    assert parsed == {"answer": "Wellington", "evidence": ["p2", "p1"]}


def test_sql_checker_reports_syntax_and_column_errors(tmp_path):
    environment = SQLAgentEnvironment(tmp_path / "demo.sqlite3")
    task = environment.get_task()
    syntax = environment.check(task, {"sql": "SELECT FROM"})
    column = environment.check(task, {"sql": "SELECT missing_column FROM employees"})
    assert not syntax.success and "SQLite query error" in syntax.message
    assert not column.success and "no such column" in column.message


def test_previous_checker_feedback_enters_next_model_input(qa_env, fake_clock, tmp_path):
    backend = MockBackend([
        '{"answer":"Auckland","evidence":["p1","p2"]}',
        '{"answer":"Wellington","evidence":["p1","p2"]}',
    ])
    manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(MemoryStore(tmp_path / "agent.db", fake_clock)),
        clock=fake_clock,
    )
    result = SageAgent(backend, qa_env, memory_manager=manager, clock=fake_clock).run(
        qa_env.get_task(), mode="full", max_iterations=5,
    )
    first_feedback = result["iterations"][0]["checker_feedback"]
    second_input = backend.calls[1]["messages"][-1]["content"]
    assert result["success"] is True
    assert len(result["iterations"]) == 2
    assert first_feedback["feedback_id"] in second_input
    assert first_feedback["message"] in second_input
    assert "Current non-parametric policy revision" in second_input
    assert "Retrieved past experience" in second_input


def test_success_stops_early(qa_env):
    backend = MockBackend(['{"answer":"Wellington","evidence":["p1","p2"]}'])
    result = SageAgent(backend, qa_env).run(qa_env.get_task(), mode="reflection", max_iterations=9)
    assert result["success"] is True
    assert result["status"] == "success"
    assert result["assistant_calls"] == 1
    assert result["checker_calls"] == 1


def test_failed_run_stops_at_max_iterations(qa_env):
    backend = MockBackend(["wrong", "still wrong"])
    result = SageAgent(backend, qa_env).run(qa_env.get_task(), mode="reflection", max_iterations=2)
    assert result["success"] is False
    assert result["status"] == "max_iterations"
    assert len(result["iterations"]) == 2
    assert result["reflection_count"] == 2


def test_output_parse_exception_is_logged_and_can_recover(tmp_path, fake_clock):
    environment = SQLAgentEnvironment(tmp_path / "parse.sqlite3")
    task = environment.get_task()
    correct = (
        "SELECT e.name AS name, d.name AS department FROM employees e "
        "JOIN departments d ON d.department_id=e.department_id "
        "JOIN assignments a ON a.employee_id=e.employee_id "
        "JOIN projects p ON p.project_id=a.project_id "
        "WHERE e.active=1 AND p.name='Apollo' ORDER BY e.name"
    )
    backend = MockBackend(["", correct])
    manager = MemoryManager(
        ShortTermMemory(), LongTermMemory(MemoryStore(tmp_path / "parse-memory.db", fake_clock)),
        clock=fake_clock,
    )
    result = SageAgent(backend, environment, memory_manager=manager, clock=fake_clock).run(
        task, mode="full", max_iterations=3,
    )
    assert result["success"]
    assert result["iterations"][0]["parse_error"].startswith("ValueError:")
    assert result["iterations"][0]["checker_feedback"]["success"] is False


def test_base_is_one_shot_and_does_not_reflect(qa_env):
    backend = MockBackend(['{"answer":"Auckland","evidence":["p1","p2"]}'])
    result = SageAgent(backend, qa_env).run(qa_env.get_task(), mode="base", max_iterations=5)
    assert result["status"] == "baseline_one_shot_failure"
    assert result["assistant_calls"] == 1
    assert result["reflection_count"] == 0
    assert result["policy_revisions"] == []


def test_mock_backend_is_never_labeled_as_local():
    result = MockBackend(["answer"]).generate([{"role": "user", "content": "test"}])
    assert result.backend_type == "mock"
    assert result.raw_response == {"mock": True}
    assert result.model_identifier == "deterministic-mock-v1"
    try:
        LocalModelBackend("missing-model.gguf")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing real model was silently accepted")


def test_generation_exception_is_recorded_and_retried(qa_env):
    class FailOnce:
        backend_type = "local"
        model_identifier = "explicit-failure-test"

        def __init__(self):
            self.calls = 0

        def generate(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("controlled backend failure")
            return MockBackend(['{"answer":"Wellington","evidence":["p1","p2"]}']).generate(messages, config)

    result = SageAgent(FailOnce(), qa_env).run(qa_env.get_task(), mode="reflection", max_iterations=3)
    assert result["success"]
    assert result["iterations"][0]["call_error"] == "RuntimeError: controlled backend failure"
    assert result["iterations"][0]["model_call"]["call_status"] == "error"
