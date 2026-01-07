from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

_PROMPT_CACHE: Dict[str, object] = {
    "text": "",
    "source": "",
    "path": "",
    "mtime": 0.0,
    "loaded_at": 0.0,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _default_prompt_path() -> Path:
    local_py = Path(__file__).with_name("prompt.py")
    if local_py.exists():
        return local_py
    local_txt = Path(__file__).with_name("prompt.txt")
    if local_txt.exists():
        return local_txt
    return local_py


def _read_text_any(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...TRUNCATED...]"


def _load_base_prompt() -> str:
    allow_env = _env_flag("PROMPT_ALLOW_ENV_OVERRIDE", True)
    max_chars = _env_int("PROMPT_MAX_CHARS", 250000)
    reload_sec = float(_env_int("PROMPT_RELOAD_SEC", 0))
    now = time.time()

    prompt_text = (os.getenv("PROMPT_TEXT") or "").strip() if allow_env else ""
    if prompt_text:
        cached = _PROMPT_CACHE
        if (
            cached.get("source") == "env_text"
            and cached.get("text") == prompt_text
            and (reload_sec <= 0 or (now - float(cached.get("loaded_at") or 0.0) < reload_sec))
        ):
            return str(cached.get("text") or "")
        text = _truncate(prompt_text, max_chars).strip()
        _PROMPT_CACHE.update(
            {"text": text, "source": "env_text", "path": "", "mtime": 0.0, "loaded_at": now}
        )
        return text

    prompt_file = (os.getenv("PROMPT_FILE") or "").strip() if allow_env else ""
    path = Path(prompt_file).expanduser() if prompt_file else _default_prompt_path()
    cached = _PROMPT_CACHE
    if path.exists():
        mtime = path.stat().st_mtime
        if (
            cached.get("source") == "file"
            and cached.get("path") == str(path)
            and float(cached.get("mtime") or 0.0) == mtime
            and (reload_sec <= 0 or (now - float(cached.get("loaded_at") or 0.0) < reload_sec))
        ):
            return str(cached.get("text") or "")

        text = _truncate(_read_text_any(path).strip(), max_chars)
        _PROMPT_CACHE.update(
            {"text": text, "source": "file", "path": str(path), "mtime": mtime, "loaded_at": now}
        )
        return text

    _PROMPT_CACHE.update(
        {"text": "", "source": "missing", "path": str(path), "mtime": 0.0, "loaded_at": now}
    )
    return ""


def build_messages_for_chat_completions(
    user_message: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    base = _load_base_prompt()
    parts: List[str] = []
    if base:
        parts.append(base)
    if system_prompt and system_prompt.strip():
        parts.append("MEMORY_CONTEXT:\n" + system_prompt.strip())

    sys_msg = "\n\n".join(parts).strip()
    msgs: List[Dict[str, str]] = []
    if sys_msg:
        msgs.append({"role": "system", "content": sys_msg})
    msgs.append({"role": "user", "content": (user_message or "").strip()})
    return msgs
