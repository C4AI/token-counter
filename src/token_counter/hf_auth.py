"""
Helpers for authenticating with the Hugging Face Hub from a local `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import login


@dataclass(frozen=True)
class HFAuthContext:
    token: Optional[str]
    source: Optional[str]


_LOGGED_IN_TOKEN: Optional[str] = None


def _find_dotenv(start_dir: Optional[Path] = None) -> Optional[Path]:
    current = (start_dir or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            return dotenv_path
    return None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_var_from_file(path: Path, key: str) -> Optional[str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        line_key, line_value = line.split("=", 1)
        if line_key.strip() != key:
            continue
        return _strip_quotes(line_value)
    return None


def load_hf_token(start_dir: Optional[Path] = None) -> HFAuthContext:
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return HFAuthContext(token=env_token, source="environment")

    dotenv_path = _find_dotenv(start_dir)
    if dotenv_path is None:
        return HFAuthContext(token=None, source=None)

    dotenv_token = _read_env_var_from_file(dotenv_path, "HF_TOKEN")
    if not dotenv_token:
        return HFAuthContext(token=None, source=None)

    os.environ["HF_TOKEN"] = dotenv_token
    return HFAuthContext(token=dotenv_token, source=str(dotenv_path))


def ensure_hf_auth(start_dir: Optional[Path] = None) -> HFAuthContext:
    global _LOGGED_IN_TOKEN

    context = load_hf_token(start_dir)
    if not context.token:
        return context

    if _LOGGED_IN_TOKEN != context.token:
        login(
            token=context.token,
            add_to_git_credential=False,
            skip_if_logged_in=False,
        )
        _LOGGED_IN_TOKEN = context.token

    return context
