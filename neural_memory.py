#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_memory.py — Python 3.11+ — produkcyjna implementacja hierarchicznej pamięci (STM + Graf Asocjacyjny + Procesor Refleksji)
====================================================================================================

Struktura katalogów (tworzy się automatycznie przy starcie):
.
├── neural_memory.py
└── data/
    ├── neural_memory.sqlite
    ├── neural_memory.sqlite-wal
    ├── neural_memory.sqlite-shm
    ├── archive.jsonl
    ├── archive.jsonl.1
    ├── archive.jsonl.2
    └── logs/
        └── neural_memory.log

Zależności:
- Wymagane: networkx
- Opcjonalne, ale tutaj obsłużone "na full" (auto-ensure install w CLI, da się wyłączyć):
  - numpy
  - faiss-cpu
  - httpx
  - python-louvain (import jako "community")

ENV:
- GROK_API_KEY    (opcjonalne — wymagane, jeśli chcesz w CLI dzwonić do Grok)
- DB_PATH         (domyślnie: ./data/neural_memory.sqlite)
- MAX_BYTES       (domyślnie: 5GB; akceptuje np. "512MB", "5GB")
- PSY_DEBUG       (1 -> DEBUG logi, inaczej INFO)
- NM_ENSURE_DEPS  (1 -> CLI spróbuje doinstalować opcjonalne paczki pipem; 0 -> nie rusza pip)

Specyfikacja:
- Trójkąt Rezonansowy: STM (deque z wagą + decay), Graf Asocjacyjny (NetworkX), Procesor Refleksji (background loop).
- Warstwa danych:
  - Vector Engine: FAISS (lokalny, 3072-dim), fallback na LIKE jeśli brak.
  - Graph: NetworkX (węzły: User, Project, Concept, Error; krawędzie: hates, depends_on, fixed_by itp.).
  - Relational Meta-Store: SQLite z JSONB-like polami + JSONL archive.
- Ingest: enrich (timestamp, sentyment regex, topic tag, NER simple regex), entity extraction, relational linker (wzmocnij wagę w grafie).
- Active Recall: semantic retrieval (FAISS) + graph expansion (sąsiedzi max wagi).
- Context Synthesis: prompt budowany z: Core Persona (L4), Relevant Facts (L2), Active Procedures (L3), Recency Stream (L0).
- Procesy tła (async loop): konsolidacja L1→L2 (grupuj 50 msg w fakt), conflict resolution (sprzeczne fakty → zadanie refleksyjne), synaptic decay (obniż wagę starych).
- Metody klasy NeuralMemory: ingest(data), resonate(query), reflect(), forget(criteria).
- MemoryObject: content, metadata, resonance_score, provenance.
- Na koniec: prosty CLI czat testowy (input → ingest → resonate → Grok API → output).

Dodatki (włączone i działające gdy deps są dostępne):
1) Async Batch Ingest: ingest kolejkowany; batch sync w asyncio.to_thread żeby nie blokować event loop.
2) Graph Community Detection: reflect robi communities (Louvain jeśli jest, inaczej greedy modularity).
3) Cross-Reference in Summarization: konsolidacja L1->L2 bierze encje i relacje z grafu i robi gęstsze fakty.

MAX_BYTES:
- storage pressure: db + wal/shm + jsonl (+ rotacje jsonl.*)
- przy przekroczeniu: rotacja archive, kasowanie tombstone batch, WAL checkpoint(TRUNCATE)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import logging
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("Brak zależności: networkx. Zainstaluj: pip install networkx") from e


DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5GB
ENV_GROK_API_KEY = "GROK_API_KEY"
ENV_DB_PATH = "DB_PATH"
ENV_MAX_BYTES = "MAX_BYTES"
ENV_PSY_DEBUG = "PSY_DEBUG"
ENV_ENSURE_DEPS = "NM_ENSURE_DEPS"

XAI_BASE_URL = "https://api.x.ai"
XAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

EMBED_DIM = 3072
STM_MAXLEN = 500
STM_HALF_LIFE_SECONDS = 60 * 60 * 6  # 6h

REFLECT_INTERVAL_SECONDS = 6.0
DECAY_INTERVAL_SECONDS = 30.0
CONSOLIDATE_MIN_BATCH = 50

TOPK_SEMANTIC = 18
TOPK_GRAPH_NEIGHBORS = 24
TOPK_FACTS = 24
TOPK_PROCS = 12
TOPK_RECENCY = 30

NODE_TYPES = ("User", "Project", "Concept", "Error")
EDGE_TYPES = ("hates", "depends_on", "fixed_by", "mentions", "related_to", "causes", "blocks", "prefers")

# Async batch ingest tuning
INGEST_QUEUE_MAX = 20000
INGEST_BATCH_MAX = 200
INGEST_BATCH_MAX_WAIT_MS = 80

# Community detection tuning
COMMUNITY_MIN_SIZE = 4
COMMUNITY_MAX_COUNT = 12
COMMUNITY_RUN_MIN_INTERVAL_S = 90.0
COMMUNITY_DIRTY_EDGE_THRESHOLD = 120
COMMUNITY_TOP_ENTITIES = 12
COMMUNITY_TOP_RELATIONS = 16
COMMUNITY_MEM_PER_ENTITY = 8

# Housekeeping tuning
HOUSEKEEP_INTERVAL_S = 12.0
HOUSEKEEP_DELETE_TOMBSTONE_BATCH = 8000
HOUSEKEEP_MAX_ROTATIONS = 5


def _env_bytes(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    try:
        lo = val.lower()
        if lo.endswith("gb"):
            return int(float(val[:-2].strip()) * 1024 * 1024 * 1024)
        if lo.endswith("mb"):
            return int(float(val[:-2].strip()) * 1024 * 1024)
        if lo.endswith("kb"):
            return int(float(val[:-2].strip()) * 1024)
        return int(val)
    except Exception:
        return default


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_loads(s: str) -> Any:
    return json.loads(s)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class OptionalDeps:
    """
    Manager opcjonalnych zależności. Pozwala:
    - wykryć brakujące moduły
    - (opcjonalnie) doinstalować pipem
    - wystawić zimportowane handle: np, faiss, httpx, community_louvain
    """

    def __init__(self, log: logging.Logger) -> None:
        self.log = log
        self.np = None
        self.faiss = None
        self.httpx = None
        self.community_louvain = None

    @staticmethod
    def _pip_install(packages: List[str], log: logging.Logger) -> None:
        if not packages:
            return
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
        log.warning("Installing optional deps via pip: %s", " ".join(packages))
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"pip install failed (code={proc.returncode}). Output:\n{proc.stdout[-4000:]}")
        log.warning("pip install ok. Output:\n%s", proc.stdout[-4000:])

    @staticmethod
    def _try_import(mod: str):
        try:
            return importlib.import_module(mod)
        except Exception:
            return None

    def ensure(self, do_install: bool) -> None:
        # Map: module -> pip package
        want = [
            ("numpy", "numpy"),
            ("faiss", "faiss-cpu"),
            ("httpx", "httpx"),
            ("community", "python-louvain"),
        ]

        missing_pkgs: List[str] = []
        found: Dict[str, Any] = {}

        for mod, pkg in want:
            m = self._try_import(mod)
            if m is None:
                missing_pkgs.append(pkg)
            else:
                found[mod] = m

        if missing_pkgs and do_install:
            # install and retry imports
            self._pip_install(missing_pkgs, self.log)
            found = {}
            missing2: List[str] = []
            for mod, pkg in want:
                m = self._try_import(mod)
                if m is None:
                    missing2.append(pkg)
                else:
                    found[mod] = m
            if missing2:
                self.log.warning("Some optional deps still missing after install: %s", ", ".join(missing2))

        # Assign handles (may still be None -> code keeps fallback)
        self.np = found.get("numpy")
        self.faiss = found.get("faiss")
        self.httpx = found.get("httpx")
        self.community_louvain = found.get("community")

    def faiss_ready(self) -> bool:
        return self.faiss is not None and self.np is not None

    def httpx_ready(self) -> bool:
        return self.httpx is not None

    def louvain_ready(self) -> bool:
        return self.community_louvain is not None


@dataclasses.dataclass(slots=True)
class MemoryObject:
    content: str
    metadata: Dict[str, Any]
    resonance_score: float = 0.0
    provenance: str = "unknown"

    def to_json(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "resonance_score": self.resonance_score,
            "provenance": self.provenance,
        }


@dataclasses.dataclass(slots=True)
class _STMItem:
    mem_id: int
    weight: float
    ts: float


@dataclasses.dataclass(slots=True)
class _IngestJob:
    data: Dict[str, Any]
    fut: asyncio.Future[MemoryObject]


class LocalEmbedding3072:
    _token_re = re.compile(r"[A-Za-z0-9_./:-]{2,}")

    def __init__(self, dim: int = EMBED_DIM, seed: int = 1337) -> None:
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * self.dim

        tokens = self._token_re.findall(text.lower())
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        freq = Counter(tokens)
        maxf = max(freq.values()) if freq else 1

        for tok, f in freq.items():
            h = hashlib.blake2b((tok + "|" + str(self.seed)).encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(h[:8], "little") % self.dim
            sign = 1.0 if (h[8] & 1) == 1 else -1.0
            tf = 0.5 + 0.5 * (f / maxf)
            tl = _clamp(len(tok) / 12.0, 0.5, 1.6)
            vec[idx] += sign * tf * tl

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return vec
        inv = 1.0 / norm
        return [v * inv for v in vec]


class MetaStore:
    def __init__(self, db_path: Path, archive_path: Path, max_bytes: int, log: logging.Logger) -> None:
        self.db_path = db_path
        self.archive_path = archive_path
        self.max_bytes = int(max_bytes)
        self.log = log

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None, timeout=30.0)
        self._conn.row_factory = sqlite3.Row

        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA busy_timeout=8000;")

        self._lock = threading.RLock()
        self._init_schema()

        self._last_housekeep_ts = 0.0

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(Exception):
                self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tombstone INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memory_level_created ON memory(level, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_tombstone ON memory(tombstone);

                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    etype TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(name, etype)
                );
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(etype);

                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, entity_id),
                    FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE,
                    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_mem_entities_entity ON memory_entities(entity_id);

                CREATE TABLE IF NOT EXISTS graph_edges (
                    src_entity_id INTEGER NOT NULL,
                    dst_entity_id INTEGER NOT NULL,
                    rel TEXT NOT NULL,
                    weight REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(src_entity_id, dst_entity_id, rel),
                    FOREIGN KEY(src_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(dst_entity_id) REFERENCES entities(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(dst_entity_id);

                CREATE TABLE IF NOT EXISTS reflection_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reflection_tasks_status ON reflection_tasks(status, created_at);

                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vec_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS derived_marks (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.close()

    def _path_sizes(self) -> int:
        total = 0
        for p in (
            self.db_path,
            self.db_path.with_suffix(self.db_path.suffix + "-wal"),
            self.db_path.with_suffix(self.db_path.suffix + "-shm"),
            self.archive_path,
        ):
            try:
                if p.exists():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        for i in range(1, HOUSEKEEP_MAX_ROTATIONS + 1):
            rp = self.archive_path.with_name(self.archive_path.name + f".{i}")
            try:
                if rp.exists():
                    total += int(rp.stat().st_size)
            except Exception:
                continue
        return total

    def _wal_checkpoint_truncate(self) -> None:
        try:
            with self._lock:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            return

    def _delete_tombstoned(self, limit: int) -> int:
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    DELETE FROM memory
                    WHERE id IN (
                        SELECT id FROM memory
                        WHERE tombstone=1
                        ORDER BY id ASC
                        LIMIT ?
                    )
                    """,
                    (int(limit),),
                )
                deleted = cur.rowcount if cur.rowcount is not None else 0
                cur.close()
                return int(deleted)
        except Exception:
            return 0

    def _rotate_archive(self) -> None:
        try:
            if not self.archive_path.exists():
                return

            for i in range(HOUSEKEEP_MAX_ROTATIONS, 0, -1):
                src = self.archive_path.with_name(self.archive_path.name + f".{i}")
                dst = self.archive_path.with_name(self.archive_path.name + f".{i+1}")
                if src.exists():
                    if i == HOUSEKEEP_MAX_ROTATIONS:
                        src.unlink(missing_ok=True)
                    else:
                        src.replace(dst)

            dst1 = self.archive_path.with_name(self.archive_path.name + ".1")
            self.archive_path.replace(dst1)
        except Exception as e:
            self.log.error("Archive rotation failed: %s", e)

    def _housekeep_if_needed(self) -> None:
        now = time.time()
        if (now - self._last_housekeep_ts) < HOUSEKEEP_INTERVAL_S:
            return
        self._last_housekeep_ts = now

        total = self._path_sizes()
        if total <= self.max_bytes:
            return

        self.log.warning("MAX_BYTES pressure: total_size=%d > max_bytes=%d -> housekeeping", total, self.max_bytes)
        self._rotate_archive()

        deleted = self._delete_tombstoned(HOUSEKEEP_DELETE_TOMBSTONE_BATCH)
        if deleted:
            self.log.warning("Housekeeping: deleted tombstoned rows: %d", deleted)

        self._wal_checkpoint_truncate()

        total2 = self._path_sizes()
        if total2 > self.max_bytes:
            self.log.warning("Housekeeping done but still above MAX_BYTES: total_size=%d max=%d", total2, self.max_bytes)

    def archive_append(self, rec: Dict[str, Any]) -> None:
        self._housekeep_if_needed()
        line = _safe_json_dumps(rec) + "\n"
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def insert_memory(self, level: str, role: str, content: str, metadata: Dict[str, Any]) -> int:
        self._housekeep_if_needed()
        now = _utc_now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO memory(level, role, content, metadata_json, created_at, updated_at, tombstone) VALUES(?,?,?,?,?,?,0)",
                (level, role, content, _safe_json_dumps(metadata), now, now),
            )
            mem_id = int(cur.lastrowid)
            cur.close()
        return mem_id

    def update_memory(self, mem_id: int, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        now = _utc_now_iso()
        with self._lock:
            if content is not None and metadata is not None:
                self._conn.execute(
                    "UPDATE memory SET content=?, metadata_json=?, updated_at=? WHERE id=?",
                    (content, _safe_json_dumps(metadata), now, mem_id),
                )
            elif content is not None:
                self._conn.execute(
                    "UPDATE memory SET content=?, updated_at=? WHERE id=?",
                    (content, now, mem_id),
                )
            elif metadata is not None:
                self._conn.execute(
                    "UPDATE memory SET metadata_json=?, updated_at=? WHERE id=?",
                    (_safe_json_dumps(metadata), now, mem_id),
                )

    def mark_tombstone(self, mem_id: int) -> None:
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute("UPDATE memory SET tombstone=1, updated_at=? WHERE id=?", (now, mem_id))

    def fetch_memory_by_ids(self, ids: Sequence[int]) -> List[sqlite3.Row]:
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory WHERE id IN ({placeholders}) AND tombstone=0", tuple(ids)
            ).fetchall()
        row_by_id = {int(r["id"]): r for r in rows}
        return [row_by_id[i] for i in ids if i in row_by_id]

    def search_like(self, query: str, limit: int) -> List[sqlite3.Row]:
        q = "%" + query.replace("%", "").replace("_", "") + "%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory
                WHERE tombstone=0 AND (content LIKE ? OR metadata_json LIKE ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (q, q, int(limit)),
            ).fetchall()
        return rows

    def upsert_entity(self, name: str, etype: str) -> int:
        if etype not in NODE_TYPES:
            etype = "Concept"
        now = _utc_now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO entities(name, etype, created_at) VALUES(?,?,?) ON CONFLICT(name, etype) DO NOTHING",
                (name, etype, now),
            )
            cur.close()
            row = self._conn.execute("SELECT id FROM entities WHERE name=? AND etype=?", (name, etype)).fetchone()
            if not row:
                raise RuntimeError("Entity upsert failed unexpectedly.")
            return int(row["id"])

    def link_memory_entity(self, memory_id: int, entity_id: int, weight: float) -> None:
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_entities(memory_id, entity_id, weight, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(memory_id, entity_id)
                DO UPDATE SET weight = weight + excluded.weight
                """,
                (int(memory_id), int(entity_id), float(weight), now),
            )

    def upsert_edge(self, src_entity_id: int, dst_entity_id: int, rel: str, weight_delta: float) -> None:
        if rel not in EDGE_TYPES:
            rel = "related_to"
        now = _utc_now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT weight FROM graph_edges WHERE src_entity_id=? AND dst_entity_id=? AND rel=?",
                (int(src_entity_id), int(dst_entity_id), rel),
            ).fetchone()
            if row:
                neww = float(row["weight"]) + float(weight_delta)
                self._conn.execute(
                    "UPDATE graph_edges SET weight=?, updated_at=? WHERE src_entity_id=? AND dst_entity_id=? AND rel=?",
                    (neww, now, int(src_entity_id), int(dst_entity_id), rel),
                )
            else:
                self._conn.execute(
                    "INSERT INTO graph_edges(src_entity_id, dst_entity_id, rel, weight, updated_at) VALUES(?,?,?,?,?)",
                    (int(src_entity_id), int(dst_entity_id), rel, float(weight_delta), now),
                )

    def load_graph_edges(self) -> List[Tuple[int, int, str, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT src_entity_id, dst_entity_id, rel, weight FROM graph_edges").fetchall()
        return [(int(r[0]), int(r[1]), str(r[2]), float(r[3])) for r in rows]

    def load_entities(self) -> List[Tuple[int, str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, name, etype FROM entities").fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]

    def fetch_entity_by_id(self, entity_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM entities WHERE id=?", (int(entity_id),)).fetchone()

    def fetch_entity_by_name_type(self, name: str, etype: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM entities WHERE name=? AND etype=?", (name, etype)).fetchone()

    def fetch_memory_ids_for_entity(self, entity_id: int, limit: int) -> List[int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT memory_id FROM memory_entities
                WHERE entity_id=?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (int(entity_id), int(limit)),
            ).fetchall()
        return [int(r[0]) for r in rows]

    def insert_reflection_task(self, kind: str, payload: Dict[str, Any]) -> int:
        now = _utc_now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO reflection_tasks(status, kind, payload_json, created_at, updated_at) VALUES('open',?,?,?,?)",
                (kind, _safe_json_dumps(payload), now, now),
            )
            task_id = int(cur.lastrowid)
            cur.close()
        return task_id

    def list_open_reflection_tasks(self, limit: int = 50) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM reflection_tasks WHERE status='open' ORDER BY id ASC LIMIT ?",
                (int(limit),),
            ).fetchall()

    def mark_reflection_done(self, task_id: int) -> None:
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE reflection_tasks SET status='done', updated_at=? WHERE id=?",
                (now, int(task_id)),
            )

    def upsert_embedding(self, memory_id: int, vec: List[float]) -> None:
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embeddings(memory_id, dim, vec_json, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(memory_id) DO UPDATE SET dim=excluded.dim, vec_json=excluded.vec_json, updated_at=excluded.updated_at
                """,
                (int(memory_id), int(len(vec)), _safe_json_dumps(vec), now),
            )

    def load_embeddings(self) -> List[Tuple[int, List[float]]]:
        with self._lock:
            rows = self._conn.execute("SELECT memory_id, vec_json, dim FROM embeddings").fetchall()
        out: List[Tuple[int, List[float]]] = []
        for r in rows:
            mid = int(r[0])
            dim = int(r[2])
            try:
                vec = _json_loads(str(r[1]))
                if isinstance(vec, list) and len(vec) == dim:
                    out.append((mid, [float(x) for x in vec]))
            except Exception:
                continue
        return out

    def fetch_latest_by_level(self, level: str, limit: int) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM memory WHERE level=? AND tombstone=0 ORDER BY id DESC LIMIT ?",
                (level, int(limit)),
            ).fetchall()

    def count_level(self, level: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(1) AS c FROM memory WHERE level=? AND tombstone=0", (level,)
            ).fetchone()
        return int(row["c"]) if row else 0

    def fetch_range_level(self, level: str, offset: int, limit: int) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM memory WHERE level=? AND tombstone=0 ORDER BY id ASC LIMIT ? OFFSET ?",
                (level, int(limit), int(offset)),
            ).fetchall()

    def fetch_top_entities_for_memory_ids(self, mem_ids: Sequence[int], limit: int) -> List[Tuple[int, float]]:
        if not mem_ids:
            return []
        placeholders = ",".join(["?"] * len(mem_ids))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT entity_id, SUM(weight) AS w
                FROM memory_entities
                WHERE memory_id IN ({placeholders})
                GROUP BY entity_id
                ORDER BY w DESC
                LIMIT ?
                """,
                tuple(int(x) for x in mem_ids) + (int(limit),),
            ).fetchall()
        out: List[Tuple[int, float]] = []
        for r in rows:
            out.append((int(r["entity_id"]), float(r["w"])))
        return out

    def fetch_top_edges_between_entities(self, entity_ids: Sequence[int], limit: int) -> List[Tuple[int, int, str, float]]:
        if not entity_ids:
            return []
        placeholders = ",".join(["?"] * len(entity_ids))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT src_entity_id, dst_entity_id, rel, weight
                FROM graph_edges
                WHERE src_entity_id IN ({placeholders}) AND dst_entity_id IN ({placeholders})
                ORDER BY weight DESC
                LIMIT ?
                """,
                tuple(int(x) for x in entity_ids) + tuple(int(x) for x in entity_ids) + (int(limit),),
            ).fetchall()
        return [(int(r[0]), int(r[1]), str(r[2]), float(r[3])) for r in rows]

    def has_signature(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM derived_marks WHERE key=? LIMIT 1", (key,)).fetchone()
        return bool(row)

    def set_signature(self, key: str, value: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO derived_marks(key, value, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )


class VectorEngine:
    def __init__(self, deps: OptionalDeps, embedder: LocalEmbedding3072, metastore: MetaStore, log: logging.Logger) -> None:
        self.deps = deps
        self.embedder = embedder
        self.metastore = metastore
        self.log = log

        self._faiss_ok = self.deps.faiss_ready()
        self._index = None
        self._id_map: List[int] = []
        self._id_to_pos: Dict[int, int] = {}
        self._tombstones: set[int] = set()
        self._lock = threading.RLock()

        if self._faiss_ok:
            self._init_faiss()
        else:
            self.log.warning("FAISS/numpy not ready -> vector engine fallback LIKE aktywny.")

    def _init_faiss(self) -> None:
        with self._lock:
            try:
                faiss = self.deps.faiss
                if faiss is None:
                    raise RuntimeError("faiss not loaded")
                idx = faiss.IndexFlatIP(EMBED_DIM)
                self._index = idx
                self._rebuild_from_store_locked()
                self.log.info("FAISS vector index initialized (dim=%d).", EMBED_DIM)
            except Exception as e:
                self._faiss_ok = False
                self._index = None
                self.log.warning("FAISS init failed; fallback to SQLite LIKE. reason=%s", e)

    def _rebuild_from_store_locked(self) -> None:
        pairs = self.metastore.load_embeddings()
        self._id_map = []
        self._id_to_pos = {}
        if not self._index:
            return
        self._index.reset()

        vecs: List[List[float]] = []
        for mem_id, vec in pairs:
            self._id_to_pos[mem_id] = len(self._id_map)
            self._id_map.append(mem_id)
            vecs.append(vec)

        if vecs:
            np = self.deps.np
            if np is None:
                raise RuntimeError("numpy not loaded")
            mat = np.asarray(vecs, dtype="float32")
            self._index.add(mat)
        self.log.info("FAISS rebuilt: %d vectors.", len(self._id_map))

    def mark_deleted(self, mem_id: int) -> None:
        with self._lock:
            self._tombstones.add(int(mem_id))

    def add_or_update(self, mem_id: int, text: str) -> List[float]:
        vec = self.embedder.embed(text)
        self.metastore.upsert_embedding(mem_id, vec)

        if not self._faiss_ok or self._index is None:
            return vec

        with self._lock:
            if mem_id in self._id_to_pos:
                self._tombstones.add(mem_id)
            try:
                np = self.deps.np
                if np is None:
                    raise RuntimeError("numpy not loaded")
                v = np.asarray([vec], dtype="float32")
                self._index.add(v)
                self._id_to_pos[mem_id] = len(self._id_map)
                self._id_map.append(mem_id)
            except Exception as e:
                self.log.warning("FAISS add failed; disabling FAISS. reason=%s", e)
                self._faiss_ok = False
                self._index = None
        return vec

    def semantic_search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        if not query.strip():
            return []
        if not self._faiss_ok or self._index is None:
            return []

        qv = self.embedder.embed(query)
        with self._lock:
            try:
                np = self.deps.np
                if np is None:
                    raise RuntimeError("numpy not loaded")
                q = np.asarray([qv], dtype="float32")
                D, I = self._index.search(q, int(topk * 4))
                scores = D[0].tolist()
                idxs = I[0].tolist()
            except Exception as e:
                self.log.warning("FAISS search failed; disabling FAISS. reason=%s", e)
                self._faiss_ok = False
                self._index = None
                return []

        out: List[Tuple[int, float]] = []
        for pos, sc in zip(idxs, scores):
            if pos < 0 or pos >= len(self._id_map):
                continue
            mem_id = int(self._id_map[pos])
            if mem_id in self._tombstones:
                continue
            out.append((mem_id, float(sc)))
            if len(out) >= topk:
                break
        return out

    def maybe_rebuild(self) -> None:
        if not self._faiss_ok or self._index is None:
            return
        with self._lock:
            if len(self._tombstones) < 200 and len(self._tombstones) < (len(self._id_map) // 10 + 1):
                return
            self._tombstones.clear()
            try:
                self._rebuild_from_store_locked()
            except Exception as e:
                self.log.warning("FAISS rebuild failed; disabling FAISS. reason=%s", e)
                self._faiss_ok = False
                self._index = None


class Enricher:
    POS_WORDS = (
        "super", "git", "działa", "dobrze", "świetnie", "zajebiście", "ok", "thanks", "dzieki", "done", "fixed",
        "success", "wygrałem", "łatwo", "perfekcyjnie"
    )
    NEG_WORDS = (
        "kurwa", "chuj", "jeb", "wypier", "cwel", "błąd", "error", "fail", "nie działa", "zepsute", "crash",
        "wywala", "problem", "fatal", "panic", "traceback", "exception"
    )

    TOPIC_RULES: List[Tuple[str, re.Pattern[str]]] = [
        ("devops", re.compile(r"\b(docker|systemd|nginx|vps|hetzner|ubuntu|debian|ssh|ufw)\b", re.I)),
        ("backend", re.compile(r"\b(fastapi|flask|django|api|endpoint|router|uvicorn|gunicorn|sqlite|postgres)\b", re.I)),
        ("frontend", re.compile(r"\b(react|vite|tailwind|angular|css|html|tsx|js|typescript)\b", re.I)),
        ("ml", re.compile(r"\b(embedding|faiss|vector|model|llm|grok|inference)\b", re.I)),
        ("errors", re.compile(r"\b(traceback|exception|error|stack|failed|fatal)\b", re.I)),
        ("biz", re.compile(r"\b(monet|biznes|sprzedaż|affiliate|token|marketing|oferta|klient)\b", re.I)),
    ]

    RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
    RE_URL = re.compile(r"\bhttps?://[^\s]+\b", re.I)
    RE_ERR = re.compile(r"\b([A-Za-z_]*Error|Exception|Traceback|HTTP\s*[45]\d{2}|E\d{3,5})\b")
    RE_CAPS = re.compile(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-zA-Ząćęłńóśźż0-9_-]{2,}\b")

    def enrich(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(metadata or {})
        text = content or ""
        meta.setdefault("timestamp", _utc_now_iso())

        low = text.lower()
        pos = sum(1 for w in self.POS_WORDS if w in low)
        neg = sum(1 for w in self.NEG_WORDS if w in low)
        score = _clamp((pos - neg) / 6.0, -2.0, 2.0)
        meta["sentiment"] = {
            "pos_hits": pos,
            "neg_hits": neg,
            "score": float(score),
            "label": "positive" if score > 0.35 else "negative" if score < -0.35 else "neutral",
        }

        topic = "general"
        for t, pat in self.TOPIC_RULES:
            if pat.search(text):
                topic = t
                break
        meta["topic"] = topic

        ents: List[str] = []
        ents.extend(self.RE_EMAIL.findall(text))
        ents.extend(self.RE_URL.findall(text))
        ents.extend(self.RE_ERR.findall(text))
        ents.extend(self.RE_CAPS.findall(text))

        uniq = []
        seen = set()
        for e in ents:
            e2 = _normalize_ws(e)
            if not e2:
                continue
            key = e2.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e2)

        meta["ner"] = uniq
        return meta

    def extract_entities(self, content: str, metadata: Dict[str, Any]) -> List[Tuple[str, str]]:
        text = content or ""
        meta = metadata or {}
        out: List[Tuple[str, str]] = []

        user = meta.get("user")
        if isinstance(user, str) and user.strip():
            out.append((_normalize_ws(user), "User"))

        proj = meta.get("project")
        if isinstance(proj, str) and proj.strip():
            out.append((_normalize_ws(proj), "Project"))

        for m in self.RE_ERR.findall(text):
            out.append((_normalize_ws(m), "Error"))

        ner_list = meta.get("ner", [])
        if isinstance(ner_list, list):
            for e in ner_list:
                if not isinstance(e, str):
                    continue
                et = "Concept"
                if self.RE_ERR.fullmatch(e):
                    et = "Error"
                out.append((_normalize_ws(e), et))

        topic = meta.get("topic")
        if isinstance(topic, str) and topic.strip() and topic != "general":
            out.append((_normalize_ws(topic), "Concept"))

        seen = set()
        final: List[Tuple[str, str]] = []
        for name, etype in out:
            if not name:
                continue
            k = (name.lower(), etype)
            if k in seen:
                continue
            seen.add(k)
            final.append((name, etype))
        return final


class AssociativeGraph:
    def __init__(self, metastore: MetaStore, log: logging.Logger) -> None:
        self.metastore = metastore
        self.log = log
        self.g = nx.MultiDiGraph()
        self._lock = threading.RLock()
        self._dirty_edges = 0
        self._load_from_store()

    def _load_from_store(self) -> None:
        with self._lock:
            self.g.clear()
            for eid, name, etype in self.metastore.load_entities():
                self.g.add_node(eid, name=name, etype=etype)

            for src, dst, rel, w in self.metastore.load_graph_edges():
                if not self.g.has_node(src):
                    er = self.metastore.fetch_entity_by_id(src)
                    if er:
                        self.g.add_node(int(er["id"]), name=str(er["name"]), etype=str(er["etype"]))
                if not self.g.has_node(dst):
                    er = self.metastore.fetch_entity_by_id(dst)
                    if er:
                        self.g.add_node(int(er["id"]), name=str(er["name"]), etype=str(er["etype"]))
                self.g.add_edge(src, dst, key=rel, rel=rel, weight=float(w))

            self._dirty_edges = 0
            self.log.info("Graph loaded: nodes=%d edges=%d", self.g.number_of_nodes(), self.g.number_of_edges())

    def dirty_edges(self) -> int:
        with self._lock:
            return int(self._dirty_edges)

    def reset_dirty_edges(self) -> None:
        with self._lock:
            self._dirty_edges = 0

    def upsert_entity(self, name: str, etype: str) -> int:
        eid = self.metastore.upsert_entity(name, etype)
        with self._lock:
            if not self.g.has_node(eid):
                self.g.add_node(eid, name=name, etype=etype)
        return eid

    def boost_relation(self, src_eid: int, dst_eid: int, rel: str, delta: float) -> None:
        delta = float(delta)
        if delta == 0.0:
            return
        rel2 = rel if rel in EDGE_TYPES else "related_to"
        self.metastore.upsert_edge(src_eid, dst_eid, rel2, delta)

        with self._lock:
            updated = False
            if self.g.has_edge(src_eid, dst_eid, key=rel2):
                data = self.g.get_edge_data(src_eid, dst_eid, key=rel2) or {}
                w = float(data.get("weight", 0.0)) + delta
                self.g[src_eid][dst_eid][rel2]["weight"] = w
                self.g[src_eid][dst_eid][rel2]["rel"] = rel2
                updated = True
            if not updated:
                self.g.add_edge(src_eid, dst_eid, key=rel2, rel=rel2, weight=delta)
            self._dirty_edges += 1

    def neighbors_by_weight(self, seed_entity_ids: Iterable[int], max_neighbors: int) -> List[Tuple[int, float]]:
        weights: Dict[int, float] = {}
        with self._lock:
            for sid in seed_entity_ids:
                if not self.g.has_node(sid):
                    continue
                for _, nb, _, data in self.g.out_edges(sid, keys=True, data=True):
                    w = float(data.get("weight", 0.0))
                    weights[nb] = weights.get(nb, 0.0) + max(0.0, w)
                for nb, _, _, data in self.g.in_edges(sid, keys=True, data=True):
                    w = float(data.get("weight", 0.0))
                    weights[nb] = weights.get(nb, 0.0) + max(0.0, w)

        items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return items[: int(max_neighbors)]

    def synaptic_decay(self, factor: float) -> None:
        factor = float(factor)
        if not (0.0 < factor < 1.0):
            return

        with self._lock:
            try:
                now = _utc_now_iso()
                with self.metastore._lock:
                    self.metastore._conn.execute(
                        "UPDATE graph_edges SET weight = weight * ?, updated_at=?",
                        (factor, now),
                    )
                for _, _, _, data in self.g.edges(keys=True, data=True):
                    data["weight"] = float(data.get("weight", 0.0)) * factor
            except Exception as e:
                self.log.warning("Graph synaptic_decay failed: %s", e)

    def snapshot_undirected_weighted(self) -> nx.Graph:
        ug = nx.Graph()
        with self._lock:
            for nid, attrs in self.g.nodes(data=True):
                ug.add_node(int(nid), **dict(attrs))
            acc: Dict[Tuple[int, int], float] = {}
            for u, v, _, data in self.g.edges(keys=True, data=True):
                uu = int(u)
                vv = int(v)
                if uu == vv:
                    continue
                a, b = (uu, vv) if uu < vv else (vv, uu)
                acc[(a, b)] = acc.get((a, b), 0.0) + float(data.get("weight", 0.0))
            for (a, b), w in acc.items():
                if w <= 0.0:
                    continue
                ug.add_edge(a, b, weight=w)
        return ug


class NeuralMemory:
    def __init__(self, deps: OptionalDeps) -> None:
        self.deps = deps

        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.data_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.max_bytes = _env_bytes(ENV_MAX_BYTES, DEFAULT_MAX_BYTES)
        db_path = Path(os.getenv(ENV_DB_PATH, str(self.data_dir / "neural_memory.sqlite"))).expanduser().resolve()
        archive_path = self.data_dir / "archive.jsonl"

        self.log = self._init_logging()
        self.metastore = MetaStore(db_path=db_path, archive_path=archive_path, max_bytes=self.max_bytes, log=self.log)

        self.enricher = Enricher()
        self.embedder = LocalEmbedding3072(dim=EMBED_DIM)
        self.vector = VectorEngine(deps=self.deps, embedder=self.embedder, metastore=self.metastore, log=self.log)
        self.graph = AssociativeGraph(metastore=self.metastore, log=self.log)

        self._stm: Deque[_STMItem] = deque(maxlen=STM_MAXLEN)
        self._stm_lock = threading.RLock()

        self._bg_stop = asyncio.Event()
        self._bg_task: Optional[asyncio.Task[None]] = None
        self._decay_task: Optional[asyncio.Task[None]] = None

        self._ingest_q: asyncio.Queue[_IngestJob] = asyncio.Queue(maxsize=INGEST_QUEUE_MAX)
        self._ingest_worker_task: Optional[asyncio.Task[None]] = None

        self._started = False
        self._last_community_run_ts = 0.0

        self._ensure_core_persona()

    def _init_logging(self) -> logging.Logger:
        log = logging.getLogger("neural_memory")
        log.setLevel(logging.DEBUG if os.getenv(ENV_PSY_DEBUG, "").strip() == "1" else logging.INFO)
        log.propagate = False
        if log.handlers:
            return log

        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        fh = logging.FileHandler(str(self.logs_dir / "neural_memory.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)

        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG if os.getenv(ENV_PSY_DEBUG, "").strip() == "1" else logging.INFO)

        log.addHandler(fh)
        log.addHandler(sh)
        return log

    def _ensure_core_persona(self) -> None:
        try:
            rows = self.metastore.fetch_latest_by_level("L4", 1)
            if rows:
                return
            persona = (
                "Core Persona (L4):\n"
                "- Styl: bez lania wody, konkret.\n"
                "- Priorytet: stabilność, produkcyjność, brak zgadywania.\n"
                "- Zasada: pamięć hierarchiczna — recency, fakty, procedury, persona.\n"
                "- Uwaga: jeśli pojawia się konflikt faktów, tworzę zadanie refleksyjne i nie udaję pewności.\n"
            )
            meta = {"timestamp": _utc_now_iso(), "seed": True, "topic": "persona"}
            mem_id = self.metastore.insert_memory(level="L4", role="system", content=persona, metadata=meta)
            self.metastore.archive_append({"kind": "seed", "mem_id": mem_id, "level": "L4", "role": "system", "content": persona, "metadata": meta})
            self.log.info("Seeded Core Persona (L4). mem_id=%d", mem_id)
        except Exception as e:
            self.log.error("Failed to ensure core persona: %s", e)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._bg_stop.clear()
        self._ingest_worker_task = asyncio.create_task(self._ingest_worker_loop(), name="ingest_worker")
        self._bg_task = asyncio.create_task(self._reflection_loop(), name="reflection_loop")
        self._decay_task = asyncio.create_task(self._decay_loop(), name="decay_loop")
        self.log.info("NeuralMemory background loops started.")

    async def stop(self) -> None:
        if not self._started:
            return
        self._bg_stop.set()

        if self._ingest_worker_task:
            self._ingest_worker_task.cancel()

        tasks = [t for t in (self._ingest_worker_task, self._bg_task, self._decay_task) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(Exception):
                await t

        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                job = self._ingest_q.get_nowait()
                if not job.fut.done():
                    job.fut.set_exception(RuntimeError("NeuralMemory stopped before ingest could complete."))
                self._ingest_q.task_done()
                continue
            break

        self._started = False
        self.metastore.close()
        self.log.info("NeuralMemory stopped.")

    def _stm_add(self, mem_id: int, weight: float) -> None:
        now = time.time()
        with self._stm_lock:
            self._stm.append(_STMItem(mem_id=int(mem_id), weight=float(weight), ts=now))

    def _stm_snapshot(self) -> List[_STMItem]:
        with self._stm_lock:
            return list(self._stm)

    def _stm_apply_decay(self) -> None:
        now = time.time()
        half_life = float(STM_HALF_LIFE_SECONDS)
        if half_life <= 0:
            return
        ln2 = math.log(2.0)

        with self._stm_lock:
            newq: Deque[_STMItem] = deque(maxlen=STM_MAXLEN)
            for it in self._stm:
                age = max(0.0, now - it.ts)
                decay = math.exp(-ln2 * (age / half_life))
                w = it.weight * decay
                if w < 0.01:
                    continue
                newq.append(_STMItem(mem_id=it.mem_id, weight=w, ts=it.ts))
            self._stm = newq

    def _stm_remove_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        s = set(int(x) for x in ids)
        with self._stm_lock:
            newq: Deque[_STMItem] = deque(maxlen=STM_MAXLEN)
            for it in self._stm:
                if it.mem_id in s:
                    continue
                newq.append(it)
            self._stm = newq

    async def ingest(self, data: Dict[str, Any]) -> MemoryObject:
        if not self._started:
            raise RuntimeError("NeuralMemory nie jest uruchomione. Zrób await nm.start().")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[MemoryObject] = loop.create_future()

        job = _IngestJob(data=dict(data or {}), fut=fut)
        try:
            self._ingest_q.put_nowait(job)
        except asyncio.QueueFull:
            self.log.error("Ingest queue full (%d). Backpressure.", INGEST_QUEUE_MAX)
            raise RuntimeError("Ingest queue full. Backpressure triggered.")

        return await fut

    async def _ingest_worker_loop(self) -> None:
        try:
            while not self._bg_stop.is_set():
                batch: List[_IngestJob] = []
                try:
                    first = await asyncio.wait_for(self._ingest_q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                batch.append(first)

                t0 = time.time()
                while len(batch) < INGEST_BATCH_MAX:
                    remain_ms = INGEST_BATCH_MAX_WAIT_MS - int((time.time() - t0) * 1000)
                    if remain_ms <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._ingest_q.get(), timeout=remain_ms / 1000.0)
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

                try:
                    results = await asyncio.to_thread(self._ingest_batch_sync, batch)
                    for job, res in zip(batch, results):
                        if not job.fut.done():
                            job.fut.set_result(res)
                except Exception as e:
                    self.log.error("Batch ingest failed: %s\n%s", e, traceback.format_exc())
                    for job in batch:
                        if not job.fut.done():
                            job.fut.set_exception(e)
                finally:
                    for _ in batch:
                        self._ingest_q.task_done()
        except asyncio.CancelledError:
            return

    def _ingest_batch_sync(self, batch: List[_IngestJob]) -> List[MemoryObject]:
        out: List[MemoryObject] = []
        for job in batch:
            out.append(self._ingest_one_sync(job.data))
        return out

    def _ingest_one_sync(self, data: Dict[str, Any]) -> MemoryObject:
        content = _normalize_ws(str(data.get("content", "")))
        if not content:
            raise ValueError("ingest: content is empty")

        role = str(data.get("role", "other"))
        level = str(data.get("level", "L1"))
        if level not in ("L0", "L1", "L2", "L3", "L4"):
            level = "L1"
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"meta_raw": str(metadata)}

        metadata = self.enricher.enrich(content, metadata)
        provenance = str(metadata.get("provenance") or data.get("provenance") or "ingest")

        mem_id = self.metastore.insert_memory(level=level, role=role, content=content, metadata=metadata)
        self.metastore.archive_append(
            {"kind": "memory", "mem_id": mem_id, "level": level, "role": role, "content": content, "metadata": metadata}
        )

        self.vector.add_or_update(mem_id, content)

        ents = self.enricher.extract_entities(content, metadata)
        entity_ids: List[int] = []
        for name, etype in ents:
            eid = self.graph.upsert_entity(name, etype)
            entity_ids.append(eid)
            boost = 1.0
            if etype == "Error":
                boost = 2.0
            if etype == "Project":
                boost = 1.5
            if etype == "User":
                boost = 1.2
            self.metastore.link_memory_entity(mem_id, eid, boost)

        self._link_relations(entity_ids=entity_ids, content=content, metadata=metadata)

        base_w = 1.0 if role == "user" else 0.9 if role == "assistant" else 0.7
        sent = metadata.get("sentiment", {})
        sscore = float(sent.get("score", 0.0)) if isinstance(sent, dict) else 0.0
        emo = 1.0 + 0.15 * abs(sscore)
        self._stm_add(mem_id, base_w * emo)

        metadata2 = dict(metadata)
        metadata2["embedding"] = {"dim": EMBED_DIM, "engine": "faiss" if self.deps.faiss_ready() else "like"}
        metadata2["mem_id"] = mem_id
        metadata2["level"] = level
        metadata2["source_role"] = role
        self.metastore.update_memory(mem_id, metadata=metadata2)

        return MemoryObject(content=content, metadata=metadata2, resonance_score=0.0, provenance=provenance)

    def _link_relations(self, entity_ids: List[int], content: str, metadata: Dict[str, Any]) -> None:
        if not entity_ids:
            return

        typed: List[Tuple[int, str, str]] = []
        for eid in entity_ids:
            row = self.metastore.fetch_entity_by_id(eid)
            if not row:
                continue
            typed.append((int(row["id"]), str(row["name"]), str(row["etype"])))

        users = [t for t in typed if t[2] == "User"]
        projects = [t for t in typed if t[2] == "Project"]
        errors = [t for t in typed if t[2] == "Error"]
        concepts = [t for t in typed if t[2] == "Concept"]

        low = content.lower()
        is_fix = bool(re.search(r"\b(fix|fixed|napraw|naprawione|rozwiąz|solved|działa)\b", low))
        is_neg = (metadata.get("sentiment", {}) or {}).get("label") == "negative"

        for u in users:
            for p in projects:
                self.graph.boost_relation(u[0], p[0], "depends_on", 0.6)

        for er in errors:
            for p in projects:
                self.graph.boost_relation(er[0], p[0], "blocks", 0.8)
            for c in concepts:
                self.graph.boost_relation(er[0], c[0], "causes", 0.5)

        if is_fix:
            for er in errors:
                for c in concepts:
                    self.graph.boost_relation(er[0], c[0], "fixed_by", 0.7)
            for p in projects:
                for c in concepts:
                    self.graph.boost_relation(p[0], c[0], "fixed_by", 0.3)

        if is_neg:
            hate_targets = [c for c in concepts if re.search(r"\b(openai|admin|model|policy|limit|cenzur)\b", c[1].lower())]
            for u in users:
                for t in hate_targets:
                    self.graph.boost_relation(u[0], t[0], "hates", 0.9)

        for i in range(len(typed)):
            for j in range(i + 1, len(typed)):
                a = typed[i]
                b = typed[j]
                self.graph.boost_relation(a[0], b[0], "mentions", 0.08)
                self.graph.boost_relation(b[0], a[0], "mentions", 0.08)
                if a[2] == "Concept" and b[2] == "Concept":
                    self.graph.boost_relation(a[0], b[0], "related_to", 0.05)
                    self.graph.boost_relation(b[0], a[0], "related_to", 0.05)

    def _row_to_memobj(self, r: sqlite3.Row) -> MemoryObject:
        meta: Dict[str, Any]
        try:
            meta_raw = _json_loads(str(r["metadata_json"]))
            meta = meta_raw if isinstance(meta_raw, dict) else {"metadata_raw": str(meta_raw)}
        except Exception:
            meta = {"metadata_raw": str(r["metadata_json"])}

        meta.setdefault("mem_id", int(r["id"]))
        meta.setdefault("level", str(r["level"]))
        meta.setdefault("source_role", str(r["role"]))
        return MemoryObject(
            content=str(r["content"]),
            metadata=meta,
            resonance_score=0.0,
            provenance=str(meta.get("provenance") or r["role"]),
        )

    def _build_context_prompt(
        self,
        core_persona: str,
        facts: List[MemoryObject],
        procedures: List[MemoryObject],
        recency: List[MemoryObject],
        dynamic_hits: List[Tuple[str, MemoryObject]],
        query: str,
    ) -> str:
        def fmt_block(title: str, lines: List[str]) -> str:
            if not lines:
                return f"{title}:\n- (brak)\n"
            return title + ":\n" + "\n".join(f"- {ln}" for ln in lines) + "\n"

        fact_lines: List[str] = []
        for f in facts[:TOPK_FACTS]:
            s = _normalize_ws(f.content)
            if len(s) > 420:
                s = s[:417] + "..."
            fact_lines.append(s)

        proc_lines: List[str] = []
        for p in procedures[:TOPK_PROCS]:
            s = _normalize_ws(p.content)
            if len(s) > 420:
                s = s[:417] + "..."
            proc_lines.append(s)

        rec_lines: List[str] = []
        for r in recency[:TOPK_RECENCY]:
            role = (str(r.metadata.get("source_role") or "")).strip()
            ts = (str(r.metadata.get("timestamp") or "")).strip()
            head = f"[{ts}]" if ts else ""
            rr = _normalize_ws(r.content)
            if len(rr) > 260:
                rr = rr[:257] + "..."
            if role:
                rec_lines.append(f"{head} ({role}) {rr}".strip())
            else:
                rec_lines.append(f"{head} {rr}".strip())

        dyn_lines: List[str] = []
        for tag, o in dynamic_hits:
            s = _normalize_ws(o.content)
            if len(s) > 360:
                s = s[:357] + "..."
            dyn_lines.append(f"{tag}({o.resonance_score:.3f}) {s}")

        out = []
        out.append("=== CONTEXT SYNTHESIS ===")
        out.append("")
        out.append(core_persona.strip() if core_persona.strip() else "Core Persona (L4):\n- (brak)\n")
        out.append("")
        out.append(fmt_block("Relevant Facts (L2)", fact_lines))
        out.append(fmt_block("Active Procedures (L3)", proc_lines))
        out.append(fmt_block("Recency Stream (L0)", rec_lines))
        out.append(fmt_block("Active Recall Hits (Semantic+Graph)", dyn_lines))
        out.append("User Query:")
        out.append(query)
        out.append("=========================")
        return "\n".join(out).strip() + "\n"

    async def resonate(self, query: str) -> Dict[str, Any]:
        query = _normalize_ws(query)
        if not query:
            return {
                "context_prompt": "",
                "core_persona": "",
                "facts": [],
                "procedures": [],
                "recency": [],
                "semantic_hits": [],
                "graph_hits": [],
                "debug": {"note": "empty query"},
            }

        self._stm_apply_decay()

        semantic_ids_scores: List[Tuple[int, float]] = []
        if self.deps.faiss_ready():
            semantic_ids_scores = self.vector.semantic_search(query, TOPK_SEMANTIC)

        semantic_rows: List[sqlite3.Row] = []
        if semantic_ids_scores:
            ids = [mid for mid, _ in semantic_ids_scores]
            semantic_rows = self.metastore.fetch_memory_by_ids(ids)
        else:
            semantic_rows = self.metastore.search_like(query, TOPK_SEMANTIC)

        semantic_objs = [self._row_to_memobj(r) for r in semantic_rows]

        seed_eids = set()
        q_meta = self.enricher.enrich(query, {"topic": "query"})
        q_ents = self.enricher.extract_entities(query, q_meta)
        for name, etype in q_ents:
            row = self.metastore.fetch_entity_by_name_type(name, etype)
            if row:
                seed_eids.add(int(row["id"]))

        sem_ids = [int(r["id"]) for r in semantic_rows if r]
        if sem_ids:
            with self.metastore._lock:
                placeholders = ",".join(["?"] * len(sem_ids))
                rows = self.metastore._conn.execute(
                    f"SELECT entity_id, SUM(weight) AS w FROM memory_entities WHERE memory_id IN ({placeholders}) GROUP BY entity_id ORDER BY w DESC LIMIT 64",
                    tuple(sem_ids),
                ).fetchall()
            for rr in rows:
                seed_eids.add(int(rr["entity_id"]))

        neighbor_eids = self.graph.neighbors_by_weight(seed_eids, TOPK_GRAPH_NEIGHBORS)

        graph_mem_ids: List[int] = []
        for eid, _w in neighbor_eids:
            graph_mem_ids.extend(self.metastore.fetch_memory_ids_for_entity(eid, limit=6))

        seen = set()
        graph_mem_ids2: List[int] = []
        for mid in graph_mem_ids:
            if mid in seen:
                continue
            seen.add(mid)
            graph_mem_ids2.append(mid)

        graph_rows = self.metastore.fetch_memory_by_ids(graph_mem_ids2[:TOPK_SEMANTIC])
        graph_objs = [self._row_to_memobj(r) for r in graph_rows]

        persona_rows = self.metastore.fetch_latest_by_level("L4", 1)
        core_persona = str(persona_rows[0]["content"]) if persona_rows else ""

        facts_rows = self.metastore.fetch_latest_by_level("L2", TOPK_FACTS)
        proc_rows = self.metastore.fetch_latest_by_level("L3", TOPK_PROCS)

        stm_items = self._stm_snapshot()
        stm_items_sorted = sorted(stm_items, key=lambda x: x.weight, reverse=True)[:TOPK_RECENCY]
        rec_ids = [it.mem_id for it in stm_items_sorted]
        rec_rows = self.metastore.fetch_memory_by_ids(rec_ids)
        if not rec_rows:
            rec_rows = self.metastore.fetch_latest_by_level("L1", TOPK_RECENCY)

        facts = [self._row_to_memobj(r) for r in facts_rows]
        procs = [self._row_to_memobj(r) for r in proc_rows]
        recency = [self._row_to_memobj(r) for r in rec_rows]

        sem_score_by_id: Dict[int, float] = {}
        for mid, sc in semantic_ids_scores:
            sem_score_by_id[mid] = float(sc)
        stm_weight_by_id = {it.mem_id: it.weight for it in stm_items}

        def score_obj(obj: MemoryObject, mem_id: int) -> float:
            s = sem_score_by_id.get(mem_id, 0.0)
            r = stm_weight_by_id.get(mem_id, 0.0)
            ents = obj.metadata.get("ner", [])
            g = 0.08 * min(10, len(ents)) if isinstance(ents, list) else 0.0
            ln = len(obj.content)
            len_pen = _clamp((ln - 800) / 2000.0, 0.0, 0.6)
            return float(s) + 0.25 * float(r) + g - len_pen

        for row, obj in zip(semantic_rows, semantic_objs):
            mid = int(row["id"])
            obj.resonance_score = score_obj(obj, mid)
        for row, obj in zip(graph_rows, graph_objs):
            mid = int(row["id"])
            obj.resonance_score = score_obj(obj, mid) * 0.92

        dyn: List[Tuple[str, MemoryObject]] = []
        for obj in sorted(semantic_objs, key=lambda o: o.resonance_score, reverse=True)[:12]:
            dyn.append(("SEM", obj))
        for obj in sorted(graph_objs, key=lambda o: o.resonance_score, reverse=True)[:10]:
            dyn.append(("GRA", obj))

        seenh = set()
        dyn2: List[Tuple[str, MemoryObject]] = []
        for tag, o in dyn:
            h = hashlib.blake2b(o.content.encode("utf-8"), digest_size=12).hexdigest()
            if h in seenh:
                continue
            seenh.add(h)
            dyn2.append((tag, o))

        prompt = self._build_context_prompt(
            core_persona=core_persona,
            facts=facts,
            procedures=procs,
            recency=recency,
            dynamic_hits=dyn2,
            query=query,
        )

        debug = {
            "faiss": bool(self.deps.faiss_ready()),
            "httpx": bool(self.deps.httpx_ready()),
            "louvain": bool(self.deps.louvain_ready()),
            "semantic_hit_count": len(semantic_objs),
            "graph_hit_count": len(graph_objs),
            "seed_entities": len(seed_eids),
            "neighbors": len(neighbor_eids),
            "stm_size": len(stm_items),
            "dirty_edges": self.graph.dirty_edges(),
        }

        return {
            "context_prompt": prompt,
            "core_persona": core_persona,
            "facts": facts,
            "procedures": procs,
            "recency": recency,
            "semantic_hits": semantic_objs,
            "graph_hits": graph_objs,
            "debug": debug,
        }

    async def reflect(self) -> None:
        await self._do_consolidation_once()
        await self._do_conflict_resolution_once()
        await self._do_community_detection_once()
        self.vector.maybe_rebuild()

    async def _reflection_loop(self) -> None:
        try:
            while not self._bg_stop.is_set():
                try:
                    await self.reflect()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.log.warning("reflection loop tick failed: %s", e)
                await asyncio.sleep(REFLECT_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _decay_loop(self) -> None:
        try:
            while not self._bg_stop.is_set():
                try:
                    self._stm_apply_decay()
                    self.graph.synaptic_decay(0.997)
                    self.vector.maybe_rebuild()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.log.warning("decay loop tick failed: %s", e)
                await asyncio.sleep(DECAY_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _do_consolidation_once(self) -> None:
        try:
            l1_count = self.metastore.count_level("L1")
            if l1_count < CONSOLIDATE_MIN_BATCH:
                return

            rows = self.metastore.fetch_range_level("L1", offset=0, limit=CONSOLIDATE_MIN_BATCH)
            if len(rows) < CONSOLIDATE_MIN_BATCH:
                return

            texts: List[str] = []
            metas: List[Dict[str, Any]] = []
            ids: List[int] = []
            for r in rows:
                ids.append(int(r["id"]))
                texts.append(str(r["content"]))
                try:
                    m = _json_loads(str(r["metadata_json"]))
                    metas.append(m if isinstance(m, dict) else {"metadata_raw": str(m)})
                except Exception:
                    metas.append({"metadata_raw": str(r["metadata_json"])})

            top_entities = self.metastore.fetch_top_entities_for_memory_ids(ids, limit=18)
            ent_ids = [eid for eid, _w in top_entities]
            ent_info: List[Dict[str, Any]] = []
            for eid, w in top_entities:
                er = self.metastore.fetch_entity_by_id(eid)
                if not er:
                    continue
                ent_info.append({"id": int(eid), "name": str(er["name"]), "etype": str(er["etype"]), "weight": float(w)})

            edges_info: List[Dict[str, Any]] = []
            if ent_ids:
                edges = self.metastore.fetch_top_edges_between_entities(ent_ids, limit=20)
                for s, d, rel, w in edges:
                    es = self.metastore.fetch_entity_by_id(s)
                    ed = self.metastore.fetch_entity_by_id(d)
                    if not es or not ed:
                        continue
                    edges_info.append(
                        {
                            "src": {"id": s, "name": str(es["name"]), "etype": str(es["etype"])},
                            "dst": {"id": d, "name": str(ed["name"]), "etype": str(ed["etype"])},
                            "rel": rel,
                            "weight": float(w),
                        }
                    )

            entity_context = {"entities": ent_info, "relations": edges_info}
            fact = self._summarize_to_fact(texts, entity_context=entity_context)

            meta = {
                "timestamp": _utc_now_iso(),
                "source": "consolidation",
                "batch_size": CONSOLIDATE_MIN_BATCH,
                "source_ids": ids,
                "topic": self._dominant_topic(metas),
                "entity_context": entity_context,
            }
            mem_id = self.metastore.insert_memory(level="L2", role="system", content=fact, metadata=meta)
            self.metastore.archive_append({"kind": "consolidated_fact", "mem_id": mem_id, "level": "L2", "role": "system", "content": fact, "metadata": meta})
            self.vector.add_or_update(mem_id, fact)

            for mid in ids:
                self.metastore.mark_tombstone(mid)
                self.vector.mark_deleted(mid)

            self._stm_remove_ids(ids)

            self.log.info("Consolidated L1->L2: new_fact_id=%d, tombstoned=%d", mem_id, len(ids))
        except Exception as e:
            self.log.warning("consolidation failed: %s", e)

    def _dominant_topic(self, metas: List[Dict[str, Any]]) -> str:
        topics = []
        for m in metas:
            t = m.get("topic")
            if isinstance(t, str) and t.strip():
                topics.append(t.strip())
        if not topics:
            return "general"
        return Counter(topics).most_common(1)[0][0]

    def _summarize_to_fact(self, texts: List[str], entity_context: Dict[str, Any]) -> str:
        joined = " ".join(_normalize_ws(t) for t in texts if t and t.strip())
        joined = joined.strip()
        if not joined:
            return "Consolidated Fact (L2):\n- (pusto)"

        entities = entity_context.get("entities", [])
        relations = entity_context.get("relations", [])

        users: List[str] = []
        projects: List[str] = []
        errors: List[str] = []
        concepts: List[str] = []
        if isinstance(entities, list):
            for e in entities:
                if not isinstance(e, dict):
                    continue
                name = str(e.get("name", "")).strip()
                et = str(e.get("etype", "")).strip()
                if not name:
                    continue
                if et == "User":
                    users.append(name)
                elif et == "Project":
                    projects.append(name)
                elif et == "Error":
                    errors.append(name)
                else:
                    concepts.append(name)

        sents = re.split(r"(?<=[.!?])\s+", joined)
        keep: List[str] = []
        pat = re.compile(
            r"\b(jest|to|ma|miał|działa|nie działa|napraw|fixed|error|błąd|exception|traceback|requires|depends|wymaga)\b",
            re.I,
        )
        for s in sents:
            s2 = _normalize_ws(s)
            if len(s2) < 18:
                continue
            if pat.search(s2):
                keep.append(s2)
            if len(keep) >= 10:
                break

        toks = re.findall(r"[a-zA-Ząćęłńóśźż0-9_./:-]{3,}", joined.lower())
        stop = {"the", "and", "that", "this", "jest", "się", "nie", "dla", "które", "oraz", "jak", "żeby", "tylko", "from", "with", "have", "has"}
        toks = [t for t in toks if t not in stop]
        kw = [t for t, _ in Counter(toks).most_common(10)]

        rel_lines: List[str] = []
        if isinstance(relations, list):
            for r in relations[:12]:
                if not isinstance(r, dict):
                    continue
                src = r.get("src", {})
                dst = r.get("dst", {})
                rel = str(r.get("rel", "related_to"))
                w = float(r.get("weight", 0.0))
                sname = str(src.get("name", "")).strip()
                dname = str(dst.get("name", "")).strip()
                if not sname or not dname:
                    continue
                rel_lines.append(f"{sname} {rel} {dname} (w={w:.2f})")

        out = []
        out.append("Consolidated Fact (L2):")

        head_parts: List[str] = []
        if users:
            head_parts.append("User=" + ", ".join(users[:3]))
        if projects:
            head_parts.append("Project=" + ", ".join(projects[:3]))
        if errors:
            head_parts.append("Error=" + ", ".join(errors[:4]))
        if head_parts:
            out.append("- Context: " + " | ".join(head_parts))

        if rel_lines:
            out.append("- Graph:")
            for ln in rel_lines[:8]:
                out.append(f"  - {ln}")

        if keep:
            out.append("- Evidence:")
            for s in keep[:8]:
                if len(s) > 260:
                    s = s[:257] + "..."
                out.append(f"  - {s}")
        else:
            for s in sents[:4]:
                s2 = _normalize_ws(s)
                if not s2:
                    continue
                if len(s2) > 260:
                    s2 = s2[:257] + "..."
                out.append(f"- {s2}")

        if kw:
            out.append("- Keywords: " + ", ".join(kw))
        if concepts:
            out.append("- Concepts: " + ", ".join(concepts[:12]))

        return "\n".join(out).strip()

    async def _do_conflict_resolution_once(self) -> None:
        try:
            facts = self.metastore.fetch_latest_by_level("L2", 120)
            if len(facts) < 8:
                return

            parsed: List[Tuple[int, str, List[Tuple[str, bool]]]] = []
            for r in facts:
                mid = int(r["id"])
                txt = str(r["content"])
                claims = self._extract_claims(txt)
                if claims:
                    parsed.append((mid, txt, claims))

            if len(parsed) < 6:
                return

            by_subj: Dict[str, List[Tuple[int, str, bool]]] = {}
            for mid, txt, claims in parsed:
                for subj, is_neg in claims:
                    by_subj.setdefault(subj, []).append((mid, txt, is_neg))

            conflicts = []
            for subj, items in by_subj.items():
                if len(items) < 2:
                    continue
                has_pos = any(not neg for _, _, neg in items)
                has_neg = any(neg for _, _, neg in items)
                if has_pos and has_neg:
                    pos = next((it for it in items if not it[2]), None)
                    neg = next((it for it in items if it[2]), None)
                    if pos and neg:
                        conflicts.append((subj, pos, neg))
                if len(conflicts) >= 6:
                    break

            if not conflicts:
                return

            for subj, pos, neg in conflicts:
                payload = {
                    "subject": subj,
                    "positive": {"mem_id": pos[0], "excerpt": pos[1][:600]},
                    "negative": {"mem_id": neg[0], "excerpt": neg[1][:600]},
                    "note": "Detected contradictory facts; requires reflection/verification.",
                }
                tid = self.metastore.insert_reflection_task("conflict", payload)
                self.log.warning("Conflict detected -> reflection_task_id=%d subject=%s", tid, subj)
        except Exception as e:
            self.log.warning("conflict resolution failed: %s", e)

    def _extract_claims(self, fact_text: str) -> List[Tuple[str, bool]]:
        out: List[Tuple[str, bool]] = []
        lines = [l.strip("- ").strip() for l in fact_text.splitlines() if l.strip().startswith("-")]
        pat1 = re.compile(r"^(?P<subj>[A-Za-z0-9_./:-]{3,})\s+(?P<neg>nie\s+)?działa\b", re.I)
        pat2 = re.compile(r"^(?P<subj>[A-Za-z0-9_./:-]{3,})\s+(is|jest|to)\s+(?P<neg>not|nie)?\b", re.I)
        for l in lines:
            m1 = pat1.search(l)
            if m1:
                subj = m1.group("subj")
                neg = bool(m1.group("neg"))
                out.append((subj.lower(), neg))
                continue
            m2 = pat2.search(l)
            if m2:
                subj = m2.group("subj")
                neg = bool(m2.group("neg"))
                out.append((subj.lower(), neg))
                continue
        return out

    async def _do_community_detection_once(self) -> None:
        now = time.time()
        if (now - self._last_community_run_ts) < COMMUNITY_RUN_MIN_INTERVAL_S:
            return
        if self.graph.dirty_edges() < COMMUNITY_DIRTY_EDGE_THRESHOLD:
            return

        try:
            ug = await asyncio.to_thread(self.graph.snapshot_undirected_weighted)
            if ug.number_of_nodes() < 8 or ug.number_of_edges() < 8:
                self.graph.reset_dirty_edges()
                self._last_community_run_ts = now
                return

            communities = await asyncio.to_thread(self._detect_communities_sync, ug)
            if not communities:
                self.graph.reset_dirty_edges()
                self._last_community_run_ts = now
                return

            made = 0
            for comm in communities[:COMMUNITY_MAX_COUNT]:
                if len(comm) < COMMUNITY_MIN_SIZE:
                    continue
                sig = self._community_signature(ug, comm, topn=COMMUNITY_TOP_ENTITIES)
                key = f"community:{sig}"
                if self.metastore.has_signature(key):
                    continue

                super_fact, meta = self._build_super_fact_from_community(ug, comm, sig)
                mem_id = self.metastore.insert_memory(level="L2", role="system", content=super_fact, metadata=meta)
                self.metastore.archive_append({"kind": "community_super_fact", "mem_id": mem_id, "level": "L2", "role": "system", "content": super_fact, "metadata": meta})
                self.vector.add_or_update(mem_id, super_fact)
                self.metastore.set_signature(key, str(mem_id))
                made += 1

            self.log.info(
                "Community detection done. communities=%d super_facts_created=%d method=%s",
                len(communities),
                made,
                "louvain" if self.deps.louvain_ready() else "greedy_modularity",
            )
        except Exception as e:
            self.log.warning("community detection failed: %s", e)
        finally:
            self.graph.reset_dirty_edges()
            self._last_community_run_ts = now

    def _detect_communities_sync(self, ug: nx.Graph) -> List[set[int]]:
        if self.deps.louvain_ready():
            try:
                cl = self.deps.community_louvain
                if cl is None:
                    raise RuntimeError("community not loaded")
                part = cl.best_partition(ug, weight="weight")
                by_c: Dict[int, set[int]] = {}
                for nid, cid in part.items():
                    by_c.setdefault(int(cid), set()).add(int(nid))
                comms = list(by_c.values())
                comms.sort(key=lambda s: len(s), reverse=True)
                return comms
            except Exception:
                pass

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(ug, weight="weight"))
            comms2 = [set(int(x) for x in c) for c in comms]
            comms2.sort(key=lambda s: len(s), reverse=True)
            return comms2
        except Exception as e:
            self.log.warning("greedy modularity community detection failed: %s", e)
            return []

    def _community_signature(self, ug: nx.Graph, comm: set[int], topn: int) -> str:
        sub = ug.subgraph(comm)
        scores: List[Tuple[int, float]] = []
        for n in sub.nodes():
            deg = 0.0
            for _, _, data in sub.edges(n, data=True):
                deg += float(data.get("weight", 0.0))
            scores.append((int(n), deg))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [str(n) for n, _ in scores[:topn]]
        raw = "COMM|" + "|".join(top)
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

    def _build_super_fact_from_community(self, ug: nx.Graph, comm: set[int], sig: str) -> Tuple[str, Dict[str, Any]]:
        sub = ug.subgraph(comm)

        scores: List[Tuple[int, float]] = []
        for n in sub.nodes():
            deg = 0.0
            for _, _, data in sub.edges(n, data=True):
                deg += float(data.get("weight", 0.0))
            scores.append((int(n), deg))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in scores[:COMMUNITY_TOP_ENTITIES]]

        ent_info: List[Dict[str, Any]] = []
        score_map = {nid: sc for nid, sc in scores}
        for nid in top_nodes:
            er = self.metastore.fetch_entity_by_id(nid)
            if not er:
                continue
            ent_info.append({"id": int(er["id"]), "name": str(er["name"]), "etype": str(er["etype"]), "score": float(score_map.get(nid, 0.0))})

        edge_lines: List[str] = []
        edge_info: List[Dict[str, Any]] = []
        edges_sorted = sorted(sub.edges(data=True), key=lambda e: float(e[2].get("weight", 0.0)), reverse=True)
        for u, v, data in edges_sorted[:COMMUNITY_TOP_RELATIONS]:
            w = float(data.get("weight", 0.0))
            eu = self.metastore.fetch_entity_by_id(int(u))
            ev = self.metastore.fetch_entity_by_id(int(v))
            if not eu or not ev:
                continue
            sname = str(eu["name"])
            dname = str(ev["name"])
            edge_lines.append(f"{sname} <-> {dname} (w={w:.2f})")
            edge_info.append(
                {
                    "src": {"id": int(u), "name": sname, "etype": str(eu["etype"])},
                    "dst": {"id": int(v), "name": dname, "etype": str(ev["etype"])},
                    "weight": w,
                }
            )

        mem_ids: List[int] = []
        for eid in top_nodes[: min(len(top_nodes), 10)]:
            mem_ids.extend(self.metastore.fetch_memory_ids_for_entity(eid, limit=COMMUNITY_MEM_PER_ENTITY))

        seen = set()
        mem_ids2: List[int] = []
        for mid in mem_ids:
            if mid in seen:
                continue
            seen.add(mid)
            mem_ids2.append(mid)

        rows = self.metastore.fetch_memory_by_ids(mem_ids2[:120])
        objs = [self._row_to_memobj(r) for r in rows]

        evidence: List[str] = []
        for o in objs[:16]:
            s = _normalize_ws(o.content)
            if len(s) > 220:
                s = s[:217] + "..."
            evidence.append(s)

        users = [e["name"] for e in ent_info if e.get("etype") == "User"]
        projects = [e["name"] for e in ent_info if e.get("etype") == "Project"]
        errors = [e["name"] for e in ent_info if e.get("etype") == "Error"]
        concepts = [e["name"] for e in ent_info if e.get("etype") == "Concept"]

        out_lines: List[str] = []
        out_lines.append("Community Super-Fact (L2):")
        ctx = []
        if users:
            ctx.append("User=" + ", ".join(users[:3]))
        if projects:
            ctx.append("Project=" + ", ".join(projects[:3]))
        if errors:
            ctx.append("Error=" + ", ".join(errors[:5]))
        if ctx:
            out_lines.append("- Context: " + " | ".join(ctx))

        if concepts:
            out_lines.append("- Concepts: " + ", ".join(concepts[:12]))

        if edge_lines:
            out_lines.append("- Community Links:")
            for ln in edge_lines[:10]:
                out_lines.append(f"  - {ln}")

        if evidence:
            out_lines.append("- Evidence:")
            for ev in evidence[:10]:
                out_lines.append(f"  - {ev}")

        text = "\n".join(out_lines).strip()

        meta = {
            "timestamp": _utc_now_iso(),
            "source": "community_detection",
            "community_signature": sig,
            "community_size": int(len(comm)),
            "entities": ent_info,
            "links": edge_info,
            "source_memory_ids": mem_ids2[:160],
        }
        return text, meta

    async def forget(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        crit = criteria or {}
        older = crit.get("older_than_days")
        contains = crit.get("contains")
        level = crit.get("level")
        entity = crit.get("entity")

        where = ["tombstone=0"]
        params: List[Any] = []

        if isinstance(level, str) and level in ("L0", "L1", "L2", "L3", "L4"):
            where.append("level=?")
            params.append(level)

        if isinstance(contains, str) and contains.strip():
            where.append("content LIKE ?")
            params.append("%" + contains.strip().replace("%", "").replace("_", "") + "%")

        if isinstance(older, int) and older > 0:
            cutoff = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=int(older))
            where.append("created_at < ?")
            params.append(cutoff.isoformat())

        ids: List[int] = []
        with self.metastore._lock:
            q = "SELECT id FROM memory WHERE " + " AND ".join(where) + " ORDER BY id ASC LIMIT 2000"
            rows = self.metastore._conn.execute(q, tuple(params)).fetchall()
            ids = [int(r["id"]) for r in rows] if rows else []

        if isinstance(entity, str) and entity.strip():
            ent = entity.strip()
            with self.metastore._lock:
                erows = self.metastore._conn.execute("SELECT id FROM entities WHERE name=?", (ent,)).fetchall()
            eids = [int(r["id"]) for r in erows] if erows else []
            if eids and ids:
                placeholders = ",".join(["?"] * len(ids))
                placeholders2 = ",".join(["?"] * len(eids))
                with self.metastore._lock:
                    rows2 = self.metastore._conn.execute(
                        f"""
                        SELECT DISTINCT memory_id FROM memory_entities
                        WHERE memory_id IN ({placeholders}) AND entity_id IN ({placeholders2})
                        """,
                        tuple(ids) + tuple(eids),
                    ).fetchall()
                ids = [int(r["memory_id"]) for r in rows2] if rows2 else []
            elif eids and not ids:
                placeholders2 = ",".join(["?"] * len(eids))
                with self.metastore._lock:
                    rows2 = self.metastore._conn.execute(
                        f"""
                        SELECT DISTINCT memory_id FROM memory_entities
                        WHERE entity_id IN ({placeholders2})
                        LIMIT 2000
                        """,
                        tuple(eids),
                    ).fetchall()
                ids = [int(r["memory_id"]) for r in rows2] if rows2 else []
            else:
                ids = []

        for mid in ids:
            self.metastore.mark_tombstone(mid)
            self.vector.mark_deleted(mid)

        self._stm_remove_ids(ids)
        self.vector.maybe_rebuild()

        return {"forgotten": len(ids), "criteria": crit}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.metastore.close()


class GrokClient:
    def __init__(self, deps: OptionalDeps, api_key: str, log: logging.Logger, timeout_s: float = 60.0) -> None:
        self.deps = deps
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ValueError("Brak GROK_API_KEY")
        self.log = log
        self.timeout_s = float(timeout_s)

    async def chat(self, messages: List[Dict[str, str]], model: str = "grok-3-mini", temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        }
        url = XAI_BASE_URL + XAI_CHAT_COMPLETIONS_PATH
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.deps.httpx_ready():
            return await self._chat_httpx(url, headers, payload)
        return await self._chat_urllib_in_thread(url, headers, payload)

    async def _chat_httpx(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
        try:
            httpx = self.deps.httpx
            if httpx is None:
                raise RuntimeError("httpx not loaded")
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            return self._extract_chat_content(data)
        except Exception as e:
            self.log.error("Grok httpx chat failed: %s", e)
            raise

    async def _chat_urllib_in_thread(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
        def do_req() -> Dict[str, Any]:
            req = urllib.request.Request(
                url=url,
                data=_safe_json_dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    return _json_loads(body)
            except urllib.error.HTTPError as he:
                body = he.read().decode("utf-8", errors="replace") if hasattr(he, "read") else ""
                raise RuntimeError(f"HTTPError {he.code}: {body[:1200]}") from he

        data = await asyncio.to_thread(do_req)
        return self._extract_chat_content(data)

    def _extract_chat_content(self, data: Dict[str, Any]) -> str:
        try:
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
        except Exception:
            pass
        return _safe_json_dumps(data)[:4000]


async def _cli_chat(ensure_deps: bool) -> int:
    # deps
    tmp_logger = logging.getLogger("neural_memory.bootstrap")
    tmp_logger.setLevel(logging.DEBUG)
    if not tmp_logger.handlers:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        tmp_logger.addHandler(sh)

    deps = OptionalDeps(log=tmp_logger)
    deps.ensure(do_install=ensure_deps)

    nm = NeuralMemory(deps=deps)
    await nm.start()

    api_key = os.getenv(ENV_GROK_API_KEY, "").strip()
    client: Optional[GrokClient] = None
    if api_key:
        client = GrokClient(deps=deps, api_key=api_key, log=nm.log)
    else:
        nm.log.warning("Brak GROK_API_KEY — CLI działa bez wywołań Grok (tylko pamięć/resonans).")

    default_user = os.getenv("NM_USER", "User")
    default_project = os.getenv("NM_PROJECT", "Overmind")

    print("NeuralMemory CLI Chat — wpisz /exit żeby wyjść.")
    print("Komendy: /debug, /forget <days>, /forget_contains <txt>, /reflect, /tasks\n")

    stop_flag = asyncio.Event()

    def _handle_sig(*_: Any) -> None:
        stop_flag.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        with contextlib.suppress(Exception):
            signal.signal(sig, _handle_sig)

    try:
        while not stop_flag.is_set():
            try:
                user_in = await asyncio.to_thread(lambda: input("> ").strip())
            except EOFError:
                break

            if not user_in:
                continue
            if user_in == "/exit":
                break

            if user_in == "/debug":
                snap = nm._stm_snapshot()
                print(
                    f"STM size={len(snap)}  FAISS={deps.faiss_ready()}  HTTPX={deps.httpx_ready()}  LOUVAIN={deps.louvain_ready()}  "
                    f"L2={nm.metastore.count_level('L2')}  L3={nm.metastore.count_level('L3')}  dirty_edges={nm.graph.dirty_edges()}"
                )
                continue

            if user_in.startswith("/forget "):
                parts = user_in.split(maxsplit=1)
                try:
                    days = int(parts[1].strip())
                except Exception:
                    print("Format: /forget <days:int>")
                    continue
                res = await nm.forget({"older_than_days": days})
                print(res)
                continue

            if user_in.startswith("/forget_contains "):
                parts = user_in.split(maxsplit=1)
                txt = parts[1].strip() if len(parts) > 1 else ""
                if not txt:
                    print("Format: /forget_contains <text>")
                    continue
                res = await nm.forget({"contains": txt})
                print(res)
                continue

            if user_in == "/reflect":
                await nm.reflect()
                print("ok")
                continue

            if user_in == "/tasks":
                tasks = nm.metastore.list_open_reflection_tasks(20)
                if not tasks:
                    print("(no open tasks)")
                else:
                    for t in tasks:
                        payload = {}
                        try:
                            payload = _json_loads(str(t["payload_json"]))
                        except Exception:
                            payload = {"raw": str(t["payload_json"])}
                        print(f"- #{int(t['id'])} kind={t['kind']} status={t['status']} subject={payload.get('subject')}")
                continue

            await nm.ingest(
                {
                    "content": user_in,
                    "role": "user",
                    "level": "L1",
                    "metadata": {"user": default_user, "project": default_project, "provenance": "cli"},
                }
            )

            pack = await nm.resonate(user_in)
            ctx = pack.get("context_prompt", "")

            if client is None:
                print("(no GROK_API_KEY) Context built. Top semantic hits:")
                hits = pack.get("semantic_hits", [])[:5]
                for h in hits:
                    s = _normalize_ws(h.content)
                    if len(s) > 200:
                        s = s[:197] + "..."
                    print(" -", s)
                continue

            messages = [
                {"role": "system", "content": ctx},
                {"role": "user", "content": user_in},
            ]
            try:
                out = await client.chat(messages=messages, model=os.getenv("GROK_MODEL", "grok-3-mini"), temperature=0.2)
            except Exception as e:
                print(f"[grok error] {e}")
                continue

            print(out.strip())

            await nm.ingest(
                {
                    "content": out,
                    "role": "assistant",
                    "level": "L1",
                    "metadata": {"user": default_user, "project": default_project, "provenance": "grok_api"},
                }
            )

    finally:
        await nm.stop()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="neural_memory.py", add_help=True)
    parser.add_argument(
        "--ensure-deps",
        action="store_true",
        help="Spróbuj doinstalować opcjonalne paczki (numpy, faiss-cpu, httpx, python-louvain) jeśli brakuje.",
    )
    parser.add_argument(
        "--no-ensure-deps",
        action="store_true",
        help="Nie uruchamiaj pip auto-install (przydatne na hostach bez neta).",
    )
    args = parser.parse_args()

    env_flag = os.getenv(ENV_ENSURE_DEPS, "").strip()
    env_default = True if env_flag == "" else (env_flag == "1")

    if args.ensure_deps:
        ensure = True
    elif args.no_ensure_deps:
        ensure = False
    else:
        ensure = env_default

    try:
        raise SystemExit(asyncio.run(_cli_chat(ensure_deps=ensure)))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
