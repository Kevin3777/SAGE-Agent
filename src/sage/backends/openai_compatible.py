"""Optional OpenAI-compatible HTTP backend using only the standard library."""
from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import GenerationResult


class OpenAICompatibleBackend:
    backend_type = "openai-compatible"

    def __init__(
        self, base_url: str | None = None, model: str | None = None,
        api_key: str | None = None, timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("SAGE_API_BASE_URL", "")).rstrip("/")
        self.model_identifier = model or os.getenv("SAGE_API_MODEL", "")
        self.api_key = api_key or os.getenv("SAGE_API_KEY", "")
        self.timeout = timeout
        if not self.base_url or not self.model_identifier:
            raise ValueError(
                "Set SAGE_API_BASE_URL and SAGE_API_MODEL (and SAGE_API_KEY only if required) "
                "to use --backend openai-compatible."
            )
        self.backend_type = "openai-compatible"

    def generate(self, messages: list[dict[str, str]], config: dict[str, Any] | None = None) -> GenerationResult:
        options = {"temperature": 0.0, "max_tokens": 256}
        options.update(config or {})
        payload = {"model": self.model_identifier, "messages": messages, **options}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            return GenerationResult(
                text=text, backend_type=self.backend_type,
                model_identifier=self.model_identifier,
                generation_parameters=options, latency=time.perf_counter() - started,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"), raw_response=result,
            )
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
