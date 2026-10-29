"""Run a transparent real-model feedback stress task with an isolated memory DB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sage.agents.agent import SageAgent
from sage.backends.local import LocalModelBackend
from sage.core.types import TaskCase
from sage.environments import MultiHopQAEnvironment
from sage.memory.manager import MemoryManager
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument("--out", default=str(project / "outputs" / "real_controlled_loop.json"))
    parser.add_argument("--memory-db", default=str(project / "outputs" / "real_controlled_memory.sqlite3"))
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    environment = MultiHopQAEnvironment()
    source_task = environment.get_task("qa-ember-city")
    task = TaskCase(
        task_id="qa-controlled-p2-first",
        task_type=source_task.task_type,
        user_task=(
            f"{source_task.user_task}\n\nControlled feedback test: on the first attempt cite "
            "only the direct location passage p2. If the Checker provides feedback, treat it as "
            "authoritative and revise the evidence list to satisfy its complete passage-chain requirement."
        ),
        instructions=(
            "This is an explicitly controlled first-attempt stress input. For your first attempt, "
            "return the correct city with evidence containing only p2. If a later prompt includes "
            "Checker feedback, obey that feedback over this first-attempt constraint and include "
            "every passage ID the Checker says is required. Return JSON with answer and evidence."
        ),
        payload=source_task.payload,
        constraints=[
            "Controlled first attempt: cite p2 only.",
            "On retry, the Checker feedback is authoritative and overrides the first-attempt constraint.",
        ],
    )
    model_path = Path(args.model_path).expanduser().resolve()
    backend = LocalModelBackend(model_path, n_ctx=4096, n_gpu_layers=0)
    clock_manager = MemoryManager(
        ShortTermMemory(capacity=64),
        LongTermMemory(MemoryStore(Path(args.memory_db).resolve())),
    )
    result = SageAgent(
        backend, environment, memory_manager=clock_manager,
        generation_config={"temperature": 0.0, "max_tokens": 256},
    ).run(task, mode="full", max_iterations=args.max_iterations)
    result["controlled_input_note"] = (
        "First-attempt p2-only instruction is intentional and disclosed. The Checker still "
        "requires p1 and p2; raw model outputs are unmodified."
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "memory_database": str(Path(args.memory_db).resolve()),
        "success": result["success"],
        "status": result["status"],
        "backend": result["backend"],
        "metrics": result["metrics"],
        "checker_feedback": [item["checker_feedback"] for item in result["iterations"]],
        "raw_outputs": [item["raw_output"] for item in result["iterations"]],
        "retrieved_memory_ids": [
            item["state_before_action"]["retrieved_memory_ids"] for item in result["iterations"]
        ],
        "controlled_input_note": result["controlled_input_note"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
