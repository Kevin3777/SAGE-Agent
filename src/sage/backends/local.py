"""Local GGUF inference through llama-cpp-python; never substitutes a mock."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import GenerationResult


class LocalModelBackend:
    backend_type = "local"

    def __init__(
        self, model_path: str | Path, *, n_ctx: int = 4096,
        n_threads: int | None = None, n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local model file does not exist: {self.model_path}")
        self.model_identifier = str(self.model_path)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "Local GGUF inference requires llama-cpp-python. Install the CPU wheel with: "
                "python -m pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from exc
        self._load_options = {
            "n_ctx": int(n_ctx), "n_gpu_layers": int(n_gpu_layers), "verbose": False,
        }
        if n_threads is not None:
            self._load_options["n_threads"] = int(n_threads)
        try:
            self._model = Llama(model_path=str(self.model_path), **self._load_options)
        except Exception as exc:
            raise RuntimeError(f"Failed to load GGUF model {self.model_path}: {exc}") from exc
        self.load_status = "loaded"

    @staticmethod
    def _render_messages(messages: list[dict[str, str]]) -> str:
        # Qwen2.5 Instruct uses ChatML markers; llama-cpp-python will apply the
        # model's native chat template through create_chat_completion instead.
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    def generate(self, messages: list[dict[str, str]], config: dict[str, Any] | None = None) -> GenerationResult:
        options = {
            "max_tokens": 256, "temperature": 0.0, "top_p": 0.9,
            "stop": ["</s>"],
        }
        options.update(config or {})
        started = time.perf_counter()
        try:
            response = self._model.create_chat_completion(messages=messages, **options)
            text = response["choices"][0]["message"]["content"] or ""
            usage = response.get("usage", {})
            return GenerationResult(
                text=text, backend_type=self.backend_type,
                model_identifier=self.model_identifier,
                generation_parameters={**options, "load_options": self._load_options},
                latency=time.perf_counter() - started,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                raw_response=response,
            )
        except Exception as exc:
            raise RuntimeError(f"Local model generation failed: {exc}") from exc
