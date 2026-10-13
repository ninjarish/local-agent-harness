from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


class ModelError(RuntimeError):
    """Raised when Foundry Local cannot produce a completion."""


@dataclass(slots=True)
class FoundryModel:
    model: str = "qwen2.5-1.5b"
    max_tokens: int = 500
    temperature: float = 0.1
    timeout_seconds: int = 180

    def complete(self, prompt: str) -> str:
        executable = shutil.which("foundry")
        if not executable:
            raise ModelError(
                "Foundry Local is not on PATH. Install it and confirm "
                "`foundry --version` works."
            )

        command = [
            executable,
            "complete",
            self.model,
            prompt,
            "--max-tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
            "--output",
            "json",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ModelError(
                f"Foundry Local timed out after {self.timeout_seconds} seconds."
            ) from error

        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise ModelError(
                "Foundry Local completion failed. Run `foundry server status` "
                f"and `foundry model load {self.model}`.\n{details}"
            )

        output = result.stdout.strip()
        if not output:
            raise ModelError("Foundry Local returned an empty completion.")
        return _extract_completion(output)


def _extract_completion(output: str) -> str:
    clean_output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output).strip()
    try:
        payload = json.loads(clean_output)
    except json.JSONDecodeError:
        return clean_output

    text = _find_text(payload)
    if text is None:
        raise ModelError("Foundry Local returned JSON without completion text.")
    return text.strip()


def _find_text(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("content", "text", "completion", "response", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
            nested = _find_text(candidate)
            if nested is not None:
                return nested
        for candidate in value.values():
            nested = _find_text(candidate)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for candidate in value:
            nested = _find_text(candidate)
            if nested is not None:
                return nested
    return None

