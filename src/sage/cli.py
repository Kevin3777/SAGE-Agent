"""Command-line interface for environment checks, runs, evaluation, and memory demos."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sage import __version__
from sage.agents.agent import SageAgent
from sage.backends import LocalModelBackend, MockBackend, OpenAICompatibleBackend
from sage.core.clock import FakeClock, SystemClock
from sage.core.types import MemoryRecord, MemoryState, new_id
from sage.environments import MultiHopQAEnvironment, SQLAgentEnvironment
from sage.memory.forgetting import ForgettingCurve
from sage.memory.manager import MemoryManager, ThresholdPolicy, create_memory_record
from sage.memory.store import LongTermMemory, MemoryStore, ShortTermMemory
from sage.memory.syntax import MemorySyntax, StrengthEstimator


def _find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    return current


ROOT = _find_project_root()
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_MODEL = Path("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
BUILTIN_DEFAULTS: dict[str, Any] = {
    "backend": "mock",
    "model_path": str(DEFAULT_MODEL),
    "mode": "full",
    "max_iterations": 10,
    "memory_db": "outputs/sage_memory.sqlite3",
    "time_unit": "seconds",
    "temperature": 0.0,
    "max_tokens": 256,
}


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None and not DEFAULT_CONFIG.is_file():
        return dict(BUILTIN_DEFAULTS)
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_path}")
    values: dict[str, Any] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = _parse_scalar(value)
    return values


def _path_from_config(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else (ROOT / path).resolve()


def _output_path(value: str | None, default_name: str) -> Path | None:
    if value == "-":
        return None
    path = Path(value) if value else ROOT / "outputs" / default_name
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _emit(data: Any, output: Path | None) -> None:
    rendered = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"JSON written to: {output}")
    print(rendered)


def _make_backend(name: str, model_path: str | None, config: dict[str, Any]):
    if name == "mock":
        return MockBackend()
    if name == "local":
        chosen = Path(model_path) if model_path else Path(config.get("model_path", DEFAULT_MODEL))
        if not chosen.is_absolute():
            chosen = (Path.cwd() / chosen).resolve()
        return LocalModelBackend(
            chosen,
            n_ctx=int(config.get("n_ctx", 4096)),
            n_threads=config.get("n_threads"),
            n_gpu_layers=int(config.get("n_gpu_layers", 0)),
        )
    if name == "openai-compatible":
        return OpenAICompatibleBackend(
            base_url=os.getenv("SAGE_API_BASE_URL"),
            model=os.getenv("SAGE_API_MODEL"),
            api_key=os.getenv("SAGE_API_KEY"),
        )
    raise ValueError(f"unsupported backend: {name}")


def _memory_manager(
    db_path: Path, clock=None, capacity: int = 64, time_unit: str = "seconds",
) -> MemoryManager:
    clock = clock or SystemClock()
    store = MemoryStore(db_path, clock=clock)
    return MemoryManager(
        ShortTermMemory(capacity=capacity),
        LongTermMemory(store), clock=clock,
        curve=ForgettingCurve(time_unit),
        thresholds=ThresholdPolicy(0.7, 0.3),
    )


def _environment(task_name: str):
    aliases = {"qa": "qa-ember-city", "sql": "sql-apollo-team"}
    task_name = aliases.get(task_name, task_name)
    if task_name.startswith("qa-"):
        environment = MultiHopQAEnvironment()
    elif task_name.startswith("sql-"):
        environment = SQLAgentEnvironment()
    else:
        raise ValueError("task must be qa, sql, qa-ember-city, qa-atlas-founder, or sql-apollo-team")
    return environment, environment.get_task(task_name)


def _mock_backend_for_task(task_type: str, task=None, *, first_wrong: bool = True) -> MockBackend:
    if task_type == "multi_hop_qa":
        answer = task.payload["answer"] if task else "Wellington"
        correct = json.dumps({"answer": answer, "evidence": ["p1", "p2"]})
        wrong_answer = "Auckland" if answer.casefold() != "auckland" else "Wellington"
        wrong = json.dumps({"answer": wrong_answer, "evidence": ["p1", "p2"]})
    else:
        correct = (
            "SELECT e.name AS name, d.name AS department "
            "FROM employees e JOIN departments d ON d.department_id=e.department_id "
            "JOIN assignments a ON a.employee_id=e.employee_id "
            "JOIN projects p ON p.project_id=a.project_id "
            "WHERE e.active=1 AND p.name='Apollo' ORDER BY e.name"
        )
        wrong = (
            "SELECT e.name AS name, d.name AS department "
            "FROM employees e JOIN departments d ON d.department_id=e.department_id "
            "JOIN assignments a ON a.employee_id=e.employee_id "
            "JOIN projects p ON p.project_id=a.project_id "
            "WHERE p.name='Apollo' ORDER BY e.name"
        )
    return MockBackend([wrong, correct] if first_wrong else [correct])


def command_env(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model_path = Path(args.model_path) if args.model_path else Path(config.get("model_path", DEFAULT_MODEL))
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    try:
        import torch
        torch_report = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_report = {"error": f"{type(exc).__name__}: {exc}"}
    host_report = None
    host_python = Path(sys.base_prefix) / "python.exe"
    if host_python.is_file() and host_python.resolve() != Path(sys.executable).resolve():
        probe = (
            "import importlib.util,json,sys; "
            "spec=importlib.util.find_spec('torch'); "
            "data={'python_executable':sys.executable,'python_version':sys.version,"
            "'torch_installed':spec is not None}; "
            "\nif spec:\n import torch; data.update({'torch_version':torch.__version__,"
            "'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})\n"
            "print(json.dumps(data))"
        )
        try:
            completed = subprocess.run(
                [str(host_python), "-c", probe], capture_output=True, text=True,
                timeout=20, check=False,
            )
            if completed.returncode == 0:
                host_report = json.loads(completed.stdout)
            else:
                host_report = {"error": completed.stderr.strip()}
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            host_report = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import llama_cpp
        llama_report = {"available": True, "version": getattr(llama_cpp, "__version__", "unknown")}
    except ImportError:
        llama_report = {"available": False}
    gpu_report = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if completed.returncode == 0:
            gpu_report = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    data = {
        "project": "SAGE-Reproduction",
        "sage_version": __version__,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch": torch_report,
        "base_python_environment": host_report,
        "nvidia_smi": gpu_report,
        "local_llama_cpp": llama_report,
        "model_path": str(model_path),
        "model_exists": model_path.is_file(),
        "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "backend_default": config.get("backend", "mock"),
    }
    _emit(data, _output_path(args.out, "environment.json"))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    environment, task = _environment(args.task or "qa")
    backend_name = args.backend or config.get("backend", "mock")
    backend = (
        _mock_backend_for_task(task.task_type, task)
        if backend_name == "mock"
        else _make_backend(backend_name, args.model_path, config)
    )
    memory_path = _path_from_config(config.get("memory_db"), ROOT / "outputs" / "sage_memory.sqlite3")
    manager = _memory_manager(memory_path, time_unit=str(config.get("time_unit", "seconds")))
    generation = {
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 256)),
    }
    agent = SageAgent(backend, environment, memory_manager=manager, generation_config=generation)
    result = agent.run(
        task, mode=args.mode or str(config.get("mode", "full")),
        max_iterations=args.max_iterations or int(config.get("max_iterations", 10)),
    )
    result["memory_database"] = str(memory_path)
    _emit(result, _output_path(args.out, "last_run.json"))
    return 0 if result["success"] else 1


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    chosen_backend = args.backend or config.get("backend", "mock")
    chosen_tasks = ["qa", "sql"] if (args.task or "all") == "all" else [args.task]
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sage-evaluation-") as temporary:
        root = Path(temporary)
        for task_name in chosen_tasks:
            environment, task = _environment(task_name)
            for mode in ("base", "reflection", "full"):
                backend = (
                    _make_backend(chosen_backend, args.model_path, config)
                    if chosen_backend != "mock"
                    else _mock_backend_for_task(task.task_type, task)
                )
                manager = (
                    _memory_manager(
                        root / f"{task.task_id}-{mode}.sqlite3",
                        time_unit=str(config.get("time_unit", "seconds")),
                    )
                    if mode == "full" else None
                )
                agent = SageAgent(
                    backend, environment, memory_manager=manager,
                    generation_config={
                        "temperature": float(config.get("temperature", 0.0)),
                        "max_tokens": int(config.get("max_tokens", 256)),
                    },
                )
                result = agent.run(
                    task, mode=mode,
                    max_iterations=args.max_iterations or int(config.get("max_iterations", 10)),
                )
                reports.append({
                    "task_id": task.task_id, "mode": mode,
                    "backend": result["backend"], "metrics": result["metrics"],
                    "status": result["status"], "success": result["success"],
                    "iterations": result["iterations"],
                })
    data = {
        "evaluation_scope": "small functional comparison; not paper benchmark performance",
        "backend_type": chosen_backend,
        "reports": reports,
        "summary": [
            {
                "task_id": report["task_id"], "mode": report["mode"],
                **{key: report["metrics"][key] for key in (
                    "success", "iterations", "checker_calls", "reflection_count",
                    "memory_retrieval_count", "memory_write_count",
                    "memory_migration_count", "memory_forget_count",
                    "token_usage", "runtime_seconds",
                )},
            }
            for report in reports
        ],
    }
    _emit(data, _output_path(args.out, "evaluation.json"))
    return 0


def command_memory_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    original = (
        "For Atlas Notes, follow the product-to-company chain and cite both linking passages. "
        "The failed decision guessed a city without verifying passage p2. "
    ) * 4
    reflection = {
        "error_description": "The answer guessed a city without checking p2.",
        "root_cause": "The entity link from product to company was not verified.",
        "correction": "Follow the company-to-location fact in p2.",
        "next_action": "Check both p1 and p2 before returning the city.",
        "reusable_experience": "Resolve multi-hop answers by citing every linking passage.",
        "confidence": 0.82,
        "evidence_refs": ["p1", "p2"],
    }
    syntax = MemorySyntax()
    optimized = syntax.optimize(
        original, reflection,
        task_constraints=["Must cite both p1 and p2."],
        evidence_refs=["p1", "p2"],
    )
    time_unit = str(config.get("time_unit", "seconds"))
    curve = ForgettingCurve(time_unit)
    estimator = StrengthEstimator(time_unit=time_unit)
    base, enhanced, score = estimator.estimate(
        checker_quality=0.8, key_constraint=True, task_importance=0.8,
        reusability=0.82, evidence_count=2,
        optimization_valid=optimized.optimization_valid,
        compression_ratio=optimized.statistics["compression_ratio"],
    )
    with tempfile.TemporaryDirectory(prefix="sage-memory-demo-") as temporary:
        clock = FakeClock(1000.0)
        manager = _memory_manager(
            Path(temporary) / "demo.sqlite3", clock=clock, time_unit=time_unit,
        )
        memory = MemoryRecord(
            memory_id=new_id("mem"), task_type="multi_hop_qa",
            original_information=original, optimized_information=optimized.optimized_information,
            reflection=reflection, evidence_refs=["p1", "p2"],
            initial_strength=8.0, optimized_strength=20.0,
            created_at=clock.now(), last_accessed_at=clock.now(),
            memory_state=MemoryState.STM, retention_rate=1.0,
            reinforcement_count=0, state_since=clock.now(),
            optimization_stats=optimized.statistics,
            strength_unit=time_unit,
        )
        manager.add(memory)
        migration_elapsed = -memory.optimized_strength * math.log(0.69)
        migration_elapsed_seconds = migration_elapsed * curve.seconds_per_unit
        clock.advance(migration_elapsed_seconds)
        migration_events = manager.tick()
        migrated = manager.ltm.get(memory.memory_id)
        forgetting_elapsed = -memory.initial_strength * math.log(0.29)
        forgetting_elapsed_seconds = forgetting_elapsed * curve.seconds_per_unit
        clock.advance(forgetting_elapsed_seconds)
        forgetting_events = manager.tick()
        data = {
            "demo_type": "memory_syntax_and_ebbinghaus",
            "assumptions": {
                "time_unit": time_unit,
                "strength_mapping": f"decay resistance in {time_unit}; engineering assumption",
            },
            "forgetting_example": {
                "tau": 30.0, "strength": 60.0, "time_unit": time_unit,
                "retention": ForgettingCurve.retention(30.0, 60.0),
                "theta1": 0.7, "theta2": 0.3,
                "boundary_states": {
                    "R=0.7": ThresholdPolicy().classify(0.7).value,
                    "R=0.3": ThresholdPolicy().classify(0.3).value,
                    "R=0.700001": ThresholdPolicy().classify(0.700001).value,
                    "R=0.299999": ThresholdPolicy().classify(0.299999).value,
                },
            },
            "memory_syntax": {
                "original_information": optimized.original_information,
                "optimized_information": optimized.optimized_information,
                "important_facts": optimized.important_facts,
                "key_entities": optimized.key_entities,
                "constraints": optimized.constraints,
                "quality_checks": optimized.statistics,
                "strength_score": score,
                "initial_strength": base,
                "optimized_strength": enhanced,
                "strength_unit": time_unit,
                "strength_increased": enhanced > base,
            },
            "lifecycle": {
                "initial_state": "STM",
                "migration_elapsed": migration_elapsed,
                "migration_elapsed_seconds": migration_elapsed_seconds,
                "migration_events": migration_events,
                "state_after_migration": migrated.memory_state.value if migrated else None,
                "retention_at_migration": (
                    migration_events[0].get("retention_at_transition")
                    if migration_events else (migrated.retention_rate if migrated else None)
                ),
                "ltm_decay_elapsed": forgetting_elapsed,
                "ltm_decay_elapsed_seconds": forgetting_elapsed_seconds,
                "forget_events": forgetting_events,
                "present_after_forgetting": manager.ltm.get(memory.memory_id) is not None,
            },
        }
    _emit(data, _output_path(args.out, "memory_demo.json"))
    return 0


def command_inspect_memory(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = _path_from_config(config.get("memory_db"), ROOT / "outputs" / "sage_memory.sqlite3")
    manager = _memory_manager(path, time_unit=str(config.get("time_unit", "seconds")))
    lifecycle_events = manager.tick()
    data = {
        "database": str(path),
        "stm_memories_in_this_process": [item.to_dict() for item in manager.stm.memories()],
        "ltm_memories": [item.to_dict() for item in manager.ltm.all()],
        "lifecycle_events": lifecycle_events,
    }
    _emit(data, _output_path(args.out, "memory_inspection.json"))
    return 0


def _add_common(subparser: argparse.ArgumentParser, *, include_mode: bool = True, include_task: bool = True) -> None:
    subparser.add_argument("--backend", choices=["mock", "local", "openai-compatible"])
    subparser.add_argument("--model-path")
    if include_mode:
        subparser.add_argument("--mode", choices=["base", "reflection", "full"])
    if include_task:
        subparser.add_argument("--task")
    subparser.add_argument("--max-iterations", type=int)
    subparser.add_argument("--config")
    subparser.add_argument("--out")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sage", description="SAGE paper-based reproduction CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("env", help="inspect Python, GPU, backends, and configured local model")
    _add_common(env, include_mode=False, include_task=False)
    run = commands.add_parser("run", help="run one QA or SQLite task")
    _add_common(run)
    evaluate = commands.add_parser("evaluate", help="run the small base/reflection/full comparison")
    _add_common(evaluate, include_mode=False)
    memory_demo = commands.add_parser("memory-demo", help="show MemorySyntax, decay, migration, and deletion")
    _add_common(memory_demo, include_mode=False, include_task=False)
    inspect = commands.add_parser("inspect-memory", help="read and age active SQLite LTM records")
    inspect.add_argument("--config")
    inspect.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handlers = {
            "env": command_env,
            "run": command_run,
            "evaluate": command_evaluate,
            "memory-demo": command_memory_demo,
            "inspect-memory": command_inspect_memory,
        }
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Configuration or execution error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
