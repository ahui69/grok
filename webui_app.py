from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import inspect
import json
import logging
import mimetypes
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, List, Callable

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import httpx  # optional but recommended (fallback Grok call)
except Exception:
    httpx = None  # type: ignore

try:
    from neural_memory import NeuralMemory, OptionalDeps  # type: ignore
except Exception as e:
    raise RuntimeError(f"Nie da się zaimportować neural_memory.py: {e}") from e


# ------------------------ utils ------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    name = name.replace(" ", "_")
    return name[:120] or "file"


def _is_text_mime(m: str) -> bool:
    m = (m or "").lower()
    return m.startswith("text/") or m in {
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/javascript",
    }


def _looks_text_ext(filename: str) -> bool:
    fn = (filename or "").lower()
    return any(fn.endswith(ext) for ext in (
        ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".log", ".csv", ".ts", ".js", ".html", ".css", ".env.example"
    ))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def _maybe_await(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


def _env_defaults() -> Tuple[str, str]:
    user = os.getenv("NM_USER", "ahui69")
    project = os.getenv("NM_PROJECT", "grok")
    return user, project


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _mask_secrets(d: Any) -> Any:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            kk = str(k).upper()
            if kk.endswith("KEY") or kk in {"GROK_API_KEY", "OPENAI_API_KEY"}:
                out[k] = "***"
            else:
                out[k] = _mask_secrets(v)
        return out
    if isinstance(d, list):
        return [_mask_secrets(x) for x in d]
    return d


def _data_root() -> Path:
    # DB_PATH jest w /data w dockerze; folder danych to /data
    db_path = Path(os.getenv("DB_PATH", "/data/mem.db")).expanduser()
    if db_path.is_absolute():
        return Path("/data")
    return (Path.cwd() / "data").resolve()


def _sessions_path() -> Path:
    p = _data_root() / "webui"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sessions_file() -> Path:
    return _sessions_path() / "sessions.json"


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    f = _sessions_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:
        with contextlib.suppress(Exception):
            f.rename(f.with_suffix(".json.bak"))
        return {}


def _save_sessions(s: Dict[str, Dict[str, Any]]) -> None:
    f = _sessions_file()
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(f)


def _touch_session(meta: Dict[str, Dict[str, Any]], sid: str, last_msg: Optional[str] = None) -> None:
    now = _now_ms()
    item = meta.get(sid) or {}
    item.setdefault("session_id", sid)
    item.setdefault("title", "Chat")
    item.setdefault("created_ms", now)
    item["updated_ms"] = now
    if last_msg:
        item["last_msg"] = last_msg[:200]
    meta[sid] = item


def _new_session_id() -> str:
    return "s_" + secrets.token_urlsafe(12)


def _uploads_dir(session_id: str) -> Path:
    p = _data_root() / "uploads" / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _attachments_index_file(session_id: str) -> Path:
    return _uploads_dir(session_id) / "attachments.jsonl"


def _append_attachment(session_id: str, rec: Dict[str, Any]) -> None:
    f = _attachments_index_file(session_id)
    with f.open("a", encoding="utf-8") as w:
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _list_attachments(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    f = _attachments_index_file(session_id)
    if not f.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        lines = f.read_text("utf-8").splitlines()
        for line in lines[-limit:]:
            with contextlib.suppress(Exception):
                out.append(json.loads(line))
    except Exception:
        return []
    return out


# ------------------------ promot loader ------------------------

class PromotLoadError(RuntimeError):
    pass


@dataclass
class PromotRuntime:
    path: str
    module: Optional[ModuleType]
    build_messages: Optional[Callable[..., Any]]
    loaded_ms: int
    error: Optional[str]


def _load_promot_module(path: str) -> PromotRuntime:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return PromotRuntime(path=str(p), module=None, build_messages=None, loaded_ms=_now_ms(),
                            error=f"promot missing: {p}")

    spec = importlib.util.spec_from_file_location("promot_runtime", str(p))
    if spec is None or spec.loader is None:
        return PromotRuntime(path=str(p), module=None, build_messages=None, loaded_ms=_now_ms(),
                            error=f"cannot load spec: {p}")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        return PromotRuntime(path=str(p), module=None, build_messages=None, loaded_ms=_now_ms(),
                            error=f"promot import error: {e}")

    fn = getattr(mod, "build_messages_for_chat_completions", None)
    if not callable(fn):
        return PromotRuntime(path=str(p), module=mod, build_messages=None, loaded_ms=_now_ms(),
                            error="build_messages_for_chat_completions() missing in promot.py")

    return PromotRuntime(path=str(p), module=mod, build_messages=fn, loaded_ms=_now_ms(), error=None)


def _promot_path() -> str:
    env_path = (os.getenv("PROMOT_PATH") or "").strip()
    if env_path:
        return env_path
    local = Path(__file__).with_name("promot.py")
    if local.exists():
        return str(local)
    return "/root/promot.py"


_PROMOT: Optional[PromotRuntime] = None


def _build_messages_with_promot(user_message: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    global _PROMOT
    if _PROMOT is None:
        _PROMOT = _load_promot_module(_promot_path())

    if _PROMOT.build_messages is None:
        # hard fallback: no promot
        msgs: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    fn = _PROMOT.build_messages
    # próbujemy “ładnie”: user_message + system_prompt
    try:
        out = fn(user_message=user_message, system_prompt=system_prompt)
        if isinstance(out, list):
            return out
    except TypeError:
        pass
    except Exception:
        pass

    # fallback: tylko user_message
    try:
        out = fn(user_message=user_message)
        if isinstance(out, list):
            return out
    except Exception:
        pass

    msgs2: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        msgs2.append({"role": "system", "content": system_prompt.strip()})
    msgs2.append({"role": "user", "content": user_message})
    return msgs2


# ------------------------ grok fallback ------------------------

def _grok_base_url() -> str:
    return (os.getenv("GROK_BASE_URL") or "").strip() or "https://api.x.ai/v1"


def _grok_model() -> str:
    return (os.getenv("GROK_MODEL") or "").strip() or "grok-4-1-fast-reasoning"


def _grok_api_key() -> str:
    return (os.getenv("GROK_API_KEY") or "").strip()


async def _call_grok(messages: List[Dict[str, str]]) -> str:
    if httpx is None:
        return "[grok error] httpx not installed"
    key = _grok_api_key()
    if not key:
        return "[grok error] GROK_API_KEY not set"

    url = _grok_base_url().rstrip("/") + "/chat/completions"
    payload = {"model": _grok_model(), "messages": messages}

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]


# ------------------------ models ------------------------

class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=80_000)
    user: Optional[str] = None
    project: Optional[str] = None
    top_k: int = Field(default=12, ge=1, le=64)
    expand_k: int = Field(default=10, ge=0, le=64)


class ChatResp(BaseModel):
    ok: bool = True
    session_id: str
    answer: str
    used_llm: bool
    context: List[Dict[str, Any]]


class NewSessionResp(BaseModel):
    ok: bool = True
    session_id: str


class SessionListResp(BaseModel):
    ok: bool = True
    sessions: List[Dict[str, Any]]


class DebugResp(BaseModel):
    ok: bool = True
    debug: Dict[str, Any]


class ReflectResp(BaseModel):
    ok: bool = True
    result: Any


class ForgetReq(BaseModel):
    days: Optional[int] = Field(default=None, ge=0, le=36500)
    contains: Optional[str] = Field(default=None, max_length=5000)
    metadata: Optional[Dict[str, Any]] = None


class ForgetResp(BaseModel):
    ok: bool = True
    result: Any


# ------------------------ app ------------------------

app = FastAPI(title="Grok Memory WebUI", version="2.1.0")

# WAŻNE: nie wywalaj procesu jeśli webui_static nie istnieje przy imporcie
app.mount("/static", StaticFiles(directory="webui_static", check_dir=False), name="static")

_mem: Optional[Any] = None


def _init_memory() -> Any:
    """
    Bezpieczna inicjalizacja pod różne podpisy __init__.
    Próby:
      - NeuralMemory(deps=deps)
      - NeuralMemory(deps)
      - NeuralMemory()
      - NeuralMemory(DB_PATH)
      - NeuralMemory(db_path=DB_PATH)
    """
    db_path = os.getenv("DB_PATH", "/data/mem.db")
    last_err: Optional[Exception] = None

    deps_logger = logging.getLogger("neural_memory.deps")
    if not deps_logger.handlers:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        deps_logger.addHandler(sh)
        deps_logger.setLevel(logging.INFO)
    deps_logger.propagate = False

    deps = OptionalDeps(log=deps_logger)
    deps.ensure(do_install=_env_flag("NM_ENSURE_DEPS", True))

    for variant in ("deps_kw", "deps_pos", "noargs", "pos", "kw"):
        try:
            if variant == "deps_kw":
                return NeuralMemory(deps=deps)
            if variant == "deps_pos":
                return NeuralMemory(deps)
            if variant == "noargs":
                return NeuralMemory()
            if variant == "pos":
                return NeuralMemory(db_path)
            if variant == "kw":
                return NeuralMemory(db_path=db_path)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"NeuralMemory init failed for DB_PATH={db_path}: {last_err}")


@app.on_event("startup")
async def _startup() -> None:
    global _mem, _PROMOT
    _data_root().mkdir(parents=True, exist_ok=True)
    _sessions_path()

    _mem = _init_memory()
    if hasattr(_mem, "start"):
        await _maybe_await(getattr(_mem, "start"))

    # ładuj promot na starcie (user request)
    _PROMOT = _load_promot_module(_promot_path())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _mem
    if _mem is None:
        return
    if hasattr(_mem, "stop"):
        await _maybe_await(getattr(_mem, "stop"))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    p = Path("webui_static/index.html")
    if not p.exists():
        raise HTTPException(status_code=500, detail="webui_static/index.html missing")
    return HTMLResponse(p.read_text("utf-8"))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts_ms": _now_ms(),
        "data_root": str(_data_root()),
        "db_path": os.getenv("DB_PATH", "/data/mem.db"),
        "grok_model": _grok_model(),
        "has_grok_key": bool(_grok_api_key()),
        "promot_path": _promot_path(),
        "promot_loaded": bool(_PROMOT and _PROMOT.error is None),
        "promot_error": (_PROMOT.error if _PROMOT else "not loaded"),
        "memory_started": bool(getattr(_mem, "_started", False)) if _mem else False,
    }


@app.post("/api/session/new", response_model=NewSessionResp)
async def api_new_session() -> NewSessionResp:
    s = _load_sessions()
    sid = _new_session_id()
    s[sid] = {"session_id": sid, "title": "Chat", "created_ms": _now_ms(), "updated_ms": _now_ms(), "last_msg": ""}
    _save_sessions(s)
    return NewSessionResp(session_id=sid)


@app.get("/api/session/list", response_model=SessionListResp)
async def api_list_sessions() -> SessionListResp:
    s = _load_sessions()
    sessions = sorted(s.values(), key=lambda x: x.get("updated_ms", 0), reverse=True)[:50]
    return SessionListResp(sessions=sessions)


@app.post("/api/session/title")
async def api_set_title(session_id: str = Form(...), title: str = Form(...)) -> Dict[str, Any]:
    title = (title or "").strip()[:60]
    if not title:
        raise HTTPException(status_code=400, detail="empty title")
    s = _load_sessions()
    if session_id not in s:
        raise HTTPException(status_code=404, detail="session not found")
    s[session_id]["title"] = title
    s[session_id]["updated_ms"] = _now_ms()
    _save_sessions(s)
    return {"ok": True}


@app.get("/api/attachments")
async def api_list_attachments(session_id: str) -> Dict[str, Any]:
    if not session_id:
        raise HTTPException(status_code=400, detail="missing session_id")
    return {"ok": True, "items": _list_attachments(session_id)}


@app.get("/api/file")
async def api_get_file(session_id: str, name: str) -> FileResponse:
    if not session_id or not name:
        raise HTTPException(status_code=400, detail="missing params")
    p = _uploads_dir(session_id) / _safe_name(name)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    mt, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mt or "application/octet-stream", filename=p.name)


@app.post("/api/upload")
async def api_upload(
    session_id: Optional[str] = Form(default=None),
    user: Optional[str] = Form(default=None),
    project: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    global _mem
    if _mem is None:
        raise HTTPException(status_code=503, detail="memory not ready")

    s = _load_sessions()
    if not session_id:
        session_id = _new_session_id()
        s[session_id] = {"session_id": session_id, "title": "Chat", "created_ms": _now_ms(), "updated_ms": _now_ms(), "last_msg": ""}
    else:
        if session_id not in s:
            s[session_id] = {"session_id": session_id, "title": "Chat", "created_ms": _now_ms(), "updated_ms": _now_ms(), "last_msg": ""}

    user_env, project_env = _env_defaults()
    user = (user or user_env).strip()[:80]
    project = (project or project_env).strip()[:80]

    filename = _safe_name(file.filename or "upload.bin")
    target = _uploads_dir(session_id) / filename

    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e}") from e
    finally:
        with contextlib.suppress(Exception):
            await file.close()

    size = target.stat().st_size
    sha = _sha256_file(target)
    mt = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    ingested_text = False
    text_preview = ""
    if _is_text_mime(mt) or _looks_text_ext(filename):
        try:
            text = target.read_text("utf-8", errors="replace")
            max_chars = 200_000
            cut = text[:max_chars]
            if len(text) > max_chars:
                cut += "\n\n[...TRUNCATED...]"
            text_preview = cut[:4000]

            ingest_fn = getattr(_mem, "ingest", None)
            if callable(ingest_fn):
                await _maybe_await(
                    ingest_fn,
                    {
                        "content": f"[FILE:{filename}]\n{cut}",
                        "role": "user",
                        "level": "L1",
                        "metadata": {
                            "role": "user", "user": user, "project": project, "session_id": session_id,
                            "source": "upload", "filename": filename, "mime": mt, "sha256": sha, "bytes": size
                        },
                    },
                )
                ingested_text = True
        except Exception:
            ingested_text = False

    rec = {
        "ts_ms": _now_ms(),
        "session_id": session_id,
        "filename": filename,
        "mime": mt,
        "bytes": size,
        "sha256": sha,
        "ingested_text": ingested_text,
        "download_url": f"/api/file?session_id={session_id}&name={filename}",
        "preview": text_preview,
    }
    _append_attachment(session_id, rec)

    _touch_session(s, session_id, last_msg=f"[file] {filename}")
    _save_sessions(s)

    return {"ok": True, "session_id": session_id, "attachment": rec}


@app.get("/api/debug", response_model=DebugResp)
async def api_debug() -> DebugResp:
    global _mem, _PROMOT
    if _mem is None:
        raise HTTPException(status_code=503, detail="memory not ready")

    debug: Dict[str, Any] = {"ts_ms": _now_ms()}

    if _PROMOT is not None:
        debug["promot_path"] = _PROMOT.path
        debug["promot_loaded_ms"] = _PROMOT.loaded_ms
        debug["promot_ok"] = _PROMOT.error is None
        if _PROMOT.error:
            debug["promot_error"] = _PROMOT.error

    # NeuralMemory.debug() jeśli istnieje
    if hasattr(_mem, "debug"):
        try:
            d = await _maybe_await(getattr(_mem, "debug"))
            if isinstance(d, dict):
                debug["memory"] = d
            else:
                debug["memory"] = {"value": str(d)}
        except Exception as e:
            debug["memory_debug_error"] = str(e)

    return DebugResp(debug=_mask_secrets(debug))


@app.post("/api/reflect", response_model=ReflectResp)
async def api_reflect() -> ReflectResp:
    global _mem
    if _mem is None:
        raise HTTPException(status_code=503, detail="memory not ready")
    if not hasattr(_mem, "reflect"):
        raise HTTPException(status_code=400, detail="memory has no reflect()")
    r = await _maybe_await(getattr(_mem, "reflect"))
    return ReflectResp(result=_mask_secrets(r))


@app.post("/api/forget", response_model=ForgetResp)
async def api_forget(req: ForgetReq) -> ForgetResp:
    global _mem
    if _mem is None:
        raise HTTPException(status_code=503, detail="memory not ready")
    if not hasattr(_mem, "forget"):
        raise HTTPException(status_code=400, detail="memory has no forget()")

    criteria: Dict[str, Any] = {}
    if req.days is not None:
        criteria["days"] = req.days
    if req.contains:
        criteria["contains"] = req.contains
    if req.metadata:
        criteria["metadata"] = req.metadata

    r = await _maybe_await(getattr(_mem, "forget"), criteria)
    return ForgetResp(result=_mask_secrets(r))


async def _safe_resonate(query: str, top_k: int, expand_k: int) -> Any:
    global _mem
    if _mem is None:
        return []

    fn = getattr(_mem, "resonate", None)
    if not callable(fn):
        return []

    # próbujemy kilka wariantów
    payload = {"query": query, "top_k": top_k, "expand_k": expand_k}

    for args, kwargs in (
        ((payload,), {}),
        ((query,), {}),
        ((), payload),
        ((), {"query": query}),
    ):
        try:
            return await _maybe_await(fn, *args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return []


async def _safe_ingest(role: str, content: str, user: str, project: str, sid: str) -> None:
    global _mem
    if _mem is None:
        return
    fn = getattr(_mem, "ingest", None)
    if not callable(fn):
        return

    obj = {
        "content": content,
        "role": role,
        "level": "L1",
        "metadata": {"role": role, "user": user, "project": project, "session_id": sid},
    }
    try:
        await _maybe_await(fn, obj)
        return
    except TypeError:
        pass
    except Exception:
        return

    # fallback: ingest(role, content)
    with contextlib.suppress(Exception):
        await _maybe_await(fn, role, content)


async def _safe_llm_answer(message: str, messages_payload: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    Prefer NeuralMemory.{chat/ask/generate/respond} jeśli istnieje.
    Jeśli nie ma — fallback call do Grok przez httpx (z promotem).
    """
    global _mem
    if _mem is not None:
        for meth in ("chat", "ask", "generate", "respond"):
            fn = getattr(_mem, meth, None)
            if callable(fn):
                out = await _maybe_await(fn, message)
                return True, (out if isinstance(out, str) else str(out))

    # fallback do Grok
    ans = await _call_grok(messages_payload)
    return True, ans


@app.post("/api/chat", response_model=ChatResp)
async def api_chat(req: ChatReq) -> ChatResp:
    global _mem
    if _mem is None:
        raise HTTPException(status_code=503, detail="memory not ready")

    sessions = _load_sessions()
    sid = (req.session_id or "").strip()
    if not sid:
        sid = _new_session_id()
        sessions[sid] = {"session_id": sid, "title": "Chat", "created_ms": _now_ms(), "updated_ms": _now_ms(), "last_msg": ""}
    else:
        if sid not in sessions:
            sessions[sid] = {"session_id": sid, "title": "Chat", "created_ms": _now_ms(), "updated_ms": _now_ms(), "last_msg": ""}

    user_env, project_env = _env_defaults()
    user = (req.user or user_env).strip()[:80]
    project = (req.project or project_env).strip()[:80]

    await _safe_ingest("user", req.message, user, project, sid)

    # resonate
    context_items: List[Dict[str, Any]] = []
    ctx_for_system = ""
    try:
        res = await _safe_resonate(req.message, req.top_k, req.expand_k)
        if isinstance(res, dict):
            ctx_for_system = str(res.get("context_prompt") or "")
            for key in ("semantic_hits", "graph_hits", "facts", "procedures", "recency"):
                items = res.get(key)
                if not isinstance(items, list):
                    continue
                for x in items:
                    if hasattr(x, "to_json"):
                        d = x.to_json()
                    elif hasattr(x, "__dict__"):
                        d = dict(x.__dict__)
                    elif isinstance(x, dict):
                        d = x
                    else:
                        d = {"value": str(x)}
                    d.setdefault("kind", key)
                    context_items.append(_mask_secrets(d))
            if not context_items:
                context_items.append(_mask_secrets({"value": res}))
        elif isinstance(res, list):
            for x in res[:64]:
                if hasattr(x, "to_json"):
                    d = x.to_json()
                elif hasattr(x, "__dict__"):
                    d = dict(x.__dict__)
                elif isinstance(x, dict):
                    d = x
                else:
                    d = {"value": str(x)}
                context_items.append(_mask_secrets(d))
        else:
            context_items.append(_mask_secrets({"value": res}))
    except Exception as e:
        context_items = [{"error": f"resonate failed: {e}"}]

    # buduj system_prompt z kontekstu (bez żadnych “śmieciowych wstawek”)
    # -> czysty JSON context (max)
    if not ctx_for_system:
        try:
            ctx_for_system = json.dumps(context_items[:12], ensure_ascii=False, indent=2)
        except Exception:
            ctx_for_system = ""

    system_prompt = ctx_for_system if ctx_for_system else None
    messages_payload = _build_messages_with_promot(req.message, system_prompt)

    used_llm, answer = await _safe_llm_answer(req.message, messages_payload)

    await _safe_ingest("assistant", answer, user, project, sid)

    _touch_session(sessions, sid, last_msg=req.message)
    _save_sessions(sessions)

    return ChatResp(session_id=sid, answer=answer, used_llm=used_llm, context=context_items)
