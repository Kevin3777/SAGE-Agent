"""Causal iterative-feedback loop with reflection and non-parametric policy evolution."""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

from sage.backends.base import BaseModelBackend
from sage.core.clock import Clock, SystemClock
from sage.core.types import (
    EvolutionGoal, Feedback, PolicyRevision, TaskCase, Trajectory,
)
from sage.feedback.checker import Checker
from sage.memory.manager import MemoryManager, MemoryRetriever, create_memory_record
from sage.reflection.generator import ReflectionGenerator

from .assistant import Assistant
from .user import User


class TaskEnvironment(Protocol):
    def format_prompt(self, task: TaskCase) -> str: ...
    def parse_output(self, raw: str) -> Any: ...
    def check(self, task: TaskCase, candidate: Any) -> Feedback: ...


class SageAgent:
    MODES = {"base", "reflection", "full"}

    def __init__(
        self, backend: BaseModelBackend, environment: TaskEnvironment, *,
        memory_manager: MemoryManager | None = None,
        reflection_generator: ReflectionGenerator | None = None,
        clock: Clock | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.assistant = Assistant(backend, generation_config)
        self.environment = environment
        self.user = User()
        self.checker = Checker(environment)
        self.memory_manager = memory_manager
        self.memory_retriever = MemoryRetriever(memory_manager) if memory_manager else None
        self.reflection_generator = reflection_generator or ReflectionGenerator()
        self.clock = clock or SystemClock()
        self.generation_config = generation_config or {"temperature": 0.0, "max_tokens": 256}

    def _messages(
        self, task: TaskCase, iteration: int, max_iterations: int,
        feedback_history: list[Feedback], reflections: list[dict[str, Any]],
        revision: PolicyRevision | None, retrieved: list[dict[str, Any]],
        mode: str,
    ) -> list[dict[str, str]]:
        observation = self.user.submit(task)
        system = (
            "You are the Assistant role in a SAGE-style agent. Follow the task constraints, "
            "use only available evidence, and revise your action when prior checker feedback "
            "identifies a concrete error."
        )
        sections = [
            f"Task instructions:\n{task.instructions}",
            self.environment.format_prompt(task),
            f"Task ID: {task.task_id}; iteration {iteration} of {max_iterations}.",
        ]
        if mode == "full" and self.memory_manager:
            context = self.memory_manager.stm.context(task_id=task.task_id, limit=4)
            if context:
                sections.append("Short-term trajectory:\n" + context)
        if retrieved:
            rendered = []
            for item in retrieved:
                memory = item["record"]
                rendered.append(
                    f"[{memory.memory_state.value}; relevance={item['relevance']:.3f}; "
                    f"retention={item['retention']:.3f}; strength={memory.current_strength:g} "
                    f"{memory.strength_unit}] {memory.optimized_information}"
                )
            sections.append("Retrieved past experience:\n" + "\n".join(rendered))
        if feedback_history:
            previous = feedback_history[-1]
            sections.append(
                "Most recent prior Checker feedback (already available before this action):\n"
                + json.dumps(previous.to_dict(), ensure_ascii=False, sort_keys=True)
            )
        if reflections:
            sections.append(
                "Evidence-grounded reflection from earlier iteration:\n"
                + json.dumps(reflections[-1], ensure_ascii=False, sort_keys=True)
            )
        if revision:
            sections.append(
                "Current non-parametric policy revision:\n"
                + json.dumps(revision.to_dict(), ensure_ascii=False, sort_keys=True)
            )
        sections.append(f"User task:\n{observation.request}\nUser constraints: {observation.constraints}")
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(sections)}]

    def run(self, task: TaskCase, mode: str = "full", max_iterations: int = 10) -> dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        started = time.perf_counter()
        checker_calls_before = self.checker.calls
        iterations: list[dict[str, Any]] = []
        feedback_history: list[Feedback] = []
        outputs: list[str] = []
        rewards: list[float] = []
        reflections: list[dict[str, Any]] = []
        revisions: list[dict[str, Any]] = []
        current_revision: PolicyRevision | None = None
        before_counts = dict(self.memory_manager.counters) if self.memory_manager else {}
        before_event_count = len(self.memory_manager.events) if self.memory_manager else 0
        token_input_total = 0
        token_output_total = 0
        token_input_known = True
        token_output_known = True
        status = "max_iterations"
        succeeded = False

        for iteration in range(1, max_iterations + 1):
            policy_before_action = current_revision.to_dict() if current_revision else None
            memory_event_start = len(self.memory_manager.events) if self.memory_manager else 0
            retrieved: list[dict[str, Any]] = []
            if mode == "full" and self.memory_retriever:
                query = task.user_task + " " + " ".join(task.constraints)
                retrieved = self.memory_retriever.retrieve(query, task.task_type, limit=4)
            messages = self._messages(
                task, iteration, max_iterations, feedback_history,
                reflections, current_revision, retrieved, mode,
            )
            prompt_text = "\n".join(message["content"] for message in messages)
            call_record: dict[str, Any] = {
                "backend_type": getattr(self.backend, "backend_type", type(self.backend).__name__),
                "model_identifier": getattr(self.backend, "model_identifier", type(self.backend).__name__),
                "backend_status": getattr(self.backend, "load_status", "ready"),
                "messages": messages,
                "generation_parameters": dict(self.generation_config),
                "call_status": "pending",
            }
            generation_result = None
            call_error = None
            try:
                generation_result = self.assistant.act(messages)
                raw_output = generation_result.text
                call_record = generation_result.to_dict() | {"messages": messages}
                outputs.append(raw_output)
                if generation_result.input_tokens is None:
                    token_input_known = False
                    token_input_total += len(prompt_text.split())
                else:
                    token_input_total += int(generation_result.input_tokens)
                if generation_result.output_tokens is None:
                    token_output_known = False
                    token_output_total += len(raw_output.split())
                else:
                    token_output_total += int(generation_result.output_tokens)
                try:
                    parsed = self.environment.parse_output(raw_output)
                    parse_error = None
                except Exception as exc:
                    parsed = None
                    parse_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                raw_output = ""
                parsed = None
                parse_error = None
                call_error = f"{type(exc).__name__}: {exc}"
                call_record.update({"call_status": "error", "error_message": call_error})
                call_record["latency"] = None
                call_record["text"] = ""

            candidate = parsed
            if parse_error:
                candidate = None
            feedback = self.checker.check(task, candidate)
            rewards.append(feedback.reward)
            trajectory = Trajectory(
                task_id=task.task_id,
                observation=self.environment.format_prompt(task),
                action=raw_output,
                assistant_output=raw_output,
                checker_feedback=feedback.to_dict(),
                reward=feedback.reward,
                timestamp=self.clock.now(),
            )

            reflection_dict = None
            if not feedback.success and mode in {"reflection", "full"}:
                try:
                    reflection = self.reflection_generator.generate(
                        task, outputs, feedback_history + [feedback], rewards,
                        (self.memory_manager.stm.recent(task_id=task.task_id) if self.memory_manager else [])
                        + [trajectory],
                        available_experiences=[
                            {"memory_id": item["record"].memory_id, "text": item["record"].optimized_information}
                            for item in retrieved
                        ],
                    )
                    reflection_dict = reflection.to_dict()
                    reflections.append(reflection_dict)
                    trajectory.reflection = reflection_dict
                    proposed = [
                        "Apply the checker correction before generating again.",
                        reflection.next_action,
                    ]
                    goal = EvolutionGoal(
                        memory_goal="Preserve the validated correction as reusable experience.",
                        strategy_goal="Change the next prompt and plan to address the observed failure.",
                        failure_to_address=feedback.message,
                        proposed_changes=proposed,
                        evidence_refs=feedback.evidence_refs,
                    )
                    current_revision = PolicyRevision(
                        version=len(revisions) + 1,
                        previous_version=len(revisions) or None,
                        revised_instructions=reflection.next_action,
                        selected_memories=[item["record"].memory_id for item in retrieved],
                        corrective_plan=proposed,
                        originating_feedback_id=feedback.feedback_id,
                    )
                    revisions.append({
                        "evolution_goal": goal.to_dict(),
                        "policy_revision": current_revision.to_dict(),
                    })
                    if mode == "full" and self.memory_manager:
                        raw_memory = (
                            f"Task: {task.user_task}\nTask constraints: {'; '.join(task.constraints)}\n"
                            f"Failed assistant output: {raw_output}\nChecker feedback: {feedback.message}\n"
                            f"Checker errors: {'; '.join(feedback.errors)}"
                        )
                        record = create_memory_record(
                            task_type=task.task_type, original_information=raw_memory,
                            reflection=reflection_dict, checker_quality=feedback.score,
                            task_constraints=task.constraints,
                            evidence_refs=feedback.evidence_refs, clock=self.clock,
                            estimator=(
                                self.memory_manager.strength_estimator
                                if self.memory_manager else None
                            ),
                        )
                        self.memory_manager.add(record)
                except Exception as exc:
                    reflection_dict = {
                        "generation_error": f"{type(exc).__name__}: {exc}",
                        "source_feedback_id": feedback.feedback_id,
                    }
                    reflections.append(reflection_dict)

            if mode == "full" and self.memory_manager:
                self.memory_manager.stm.add_trajectory(trajectory)
            log = {
                "iteration": iteration,
                "state_before_action": {
                    "prior_feedback_ids": [item.feedback_id for item in feedback_history],
                    "policy_revision": policy_before_action,
                    "retrieved_memory_ids": [item["record"].memory_id for item in retrieved],
                    "retrieved_memories": [
                        {
                            "memory_id": item["record"].memory_id,
                            "state": item["record"].memory_state.value,
                            "relevance": item["relevance"],
                            "retention": item["retention"],
                            "current_strength": item["record"].current_strength,
                            "strength_unit": item["record"].strength_unit,
                            "optimized_information": item["record"].optimized_information,
                        }
                        for item in retrieved
                    ],
                },
                "model_call": call_record,
                "raw_output": raw_output,
                "parsed_output": parsed,
                "parse_error": parse_error,
                "call_error": call_error,
                "checker_feedback": feedback.to_dict(),
                "reward": feedback.reward,
                "reflection": reflection_dict,
                "memory_events": (
                    self.memory_manager.events[memory_event_start:]
                    if self.memory_manager and mode == "full" else []
                ),
            }
            iterations.append(log)
            feedback_history.append(feedback)
            if feedback.success:
                status = "success"
                succeeded = True
                break
            if mode == "base":
                status = "baseline_one_shot_failure"
                break

        memory_counters = {}
        if self.memory_manager:
            after_counts = self.memory_manager.counters
            memory_counters = {key: after_counts[key] - before_counts.get(key, 0) for key in after_counts}
        reflection_count = sum(
            1 for item in reflections if "generation_error" not in item
        )
        return {
            "task": {
                "task_id": task.task_id, "task_type": task.task_type,
                "user_task": task.user_task, "constraints": task.constraints,
            },
            "roles": ["User", "Assistant", "Checker"],
            "mode": mode,
            "status": status,
            "success": succeeded,
            "iterations": iterations,
            "assistant_calls": len(iterations),
                "checker_calls": self.checker.calls - checker_calls_before,
            "reflection_count": reflection_count,
            "reflections": reflections,
            "policy_revisions": revisions,
            "memory_events": (
                self.memory_manager.events[before_event_count:]
                if self.memory_manager else []
            ),
            "metrics": {
                "success": succeeded,
                "iterations": len(iterations),
                "checker_calls": len(iterations),
                "reflection_count": reflection_count,
                "memory_retrieval_count": memory_counters.get("memory_retrieval_count", 0),
                "memory_write_count": memory_counters.get("memory_write_count", 0),
                "memory_migration_count": memory_counters.get("memory_migration_count", 0),
                "memory_forget_count": memory_counters.get("memory_forget_count", 0),
                "token_usage": {
                    "input_tokens": token_input_total,
                    "output_tokens": token_output_total,
                    "input_exact": token_input_known,
                    "output_exact": token_output_known,
                },
                "runtime_seconds": time.perf_counter() - started,
            },
            "backend": {
                "backend_type": getattr(self.backend, "backend_type", type(self.backend).__name__),
                "model_identifier": getattr(self.backend, "model_identifier", type(self.backend).__name__),
                "load_status": getattr(self.backend, "load_status", "ready"),
            },
        }
