from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "sage.cli", *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )


def test_cli_entry_point_runs_help_and_memory_demo():
    help_result = run_cli("--help")
    assert help_result.returncode == 0
    assert "memory-demo" in help_result.stdout
    demo_result = run_cli("memory-demo", "--out", "-")
    assert demo_result.returncode == 0, demo_result.stderr
    start = demo_result.stdout.find("{")
    payload = json.loads(demo_result.stdout[start:])
    assert payload["lifecycle"]["state_after_migration"] == "LTM"
    assert payload["lifecycle"]["present_after_forgetting"] is False


def test_cli_run_writes_checkable_json_trace(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text(
        f"model_path: {PROJECT_ROOT.as_posix()}/missing.gguf\n"
        f"memory_db: {(tmp_path / 'cli-memory.db').as_posix()}\n"
        "mode: full\nmax_iterations: 3\ntemperature: 0\nmax_tokens: 256\n",
        encoding="utf-8",
    )
    output = tmp_path / "run.json"
    result = run_cli(
        "run", "--backend", "mock", "--task", "qa", "--mode", "full",
        "--max-iterations", "3", "--config", str(config), "--out", str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["backend"]["backend_type"] == "mock"
    assert report["iterations"][1]["state_before_action"]["prior_feedback_ids"]
