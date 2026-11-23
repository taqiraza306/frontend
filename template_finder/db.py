"""SQLite storage for asset metadata, screenshots, embedding records and selections.

The vectors themselves live in FAISS; `asset_embeddings.id` is the FAISS vector id,
so every vector can be traced back to its asset (and page) and the text it encodes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id                TEXT PRIMARY KEY,
    asset_type        TEXT NOT NULL,          -- website_template | page_template | ui_block | ui_component
    level             INTEGER NOT NULL,       -- 1 project, 2 page, 3 section, 4 element
    name              TEXT NOT NULL,
    folder_path       TEXT NOT NULL UNIQUE,   -- relative to the library folder
    short_description TEXT,
    description       TEXT,
    category          TEXT,                   -- domain: saas, portfolio, ecommerce ...
    subcategory       TEXT,                   -- kind: dashboard page, hero block, button ...
    framework         TEXT,
    language          TEXT,
    styling           TEXT,
    ui_library        TEXT,
    theme             TEXT,                   -- dark | light | both
    design_style      TEXT DEFAULT '[]',      -- JSON array
    features          TEXT DEFAULT '[]',      -- JSON array
    ui_elements       TEXT DEFAULT '[]',      -- JSON array
    color_palette     TEXT DEFAULT '[]',      -- JSON array of hex colors
    typography        TEXT,
    layout            TEXT,
    responsive        INTEGER,
    use_cases         TEXT DEFAULT '[]',      -- JSON array
    keywords          TEXT DEFAULT '[]',      -- JSON array
    dependencies      TEXT DEFAULT '[]',      -- JSON array (from package.json if present)
    repository_url    TEXT,
    preview_url       TEXT,
    license           TEXT,
    status            TEXT DEFAULT 'pending', -- verified | pending | rejected
    quality_score     REAL,
    content_hash      TEXT,
    analysis_model    TEXT,
    raw_analysis      TEXT,                   -- full JSON returned by the model
    created_at        TEXT DEFAULT (datetime('now')),
    updated_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS asset_screenshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    file_path   TEXT NOT NULL,                -- relative to the library folder
    page_name   TEXT,
    route       TEXT,
    description TEXT,
    sort_order  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS asset_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,   -- == FAISS vector id
    asset_id        TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    screenshot_id   INTEGER REFERENCES asset_screenshots(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                       -- 'asset' or 'page'
    searchable_text TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimension       INTEGER NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
    asset_id UNINDEXED,
    name,
    description,
    tags,
    searchable_text,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS project_sessions (
    project_id        TEXT PRIMARY KEY,
    selected_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
    query             TEXT,
    user_requirements TEXT,
    status            TEXT DEFAULT 'selected',
    created_at        TEXT DEFAULT (datetime('now')),
    updated_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS search_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    requirements TEXT,
    result_ids   TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_screens_asset ON asset_screenshots(asset_id);
CREATE INDEX IF NOT EXISTS idx_embed_asset ON asset_embeddings(asset_id);
"""

JSON_COLUMNS = (
    "design_style", "features", "ui_elements", "color_palette",
    "use_cases", "keywords", "dependencies",
)

ASSET_COLUMNS = (
    "id", "asset_type", "level", "name", "folder_path", "short_description", "description",
    "category", "subcategory", "framework", "language", "styling", "ui_library", "theme",
    *JSON_COLUMNS, "typography", "layout", "responsive", "repository_url", "preview_url",
    "license", "status", "quality_score", "content_hash", "analysis_model", "raw_analysis",
)


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    """Short-lived connection: commit on success, roll back on error, always close."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for col in JSON_COLUMNS:
        if col in d:
            try:
                d[col] = json.loads(d[col] or "[]")
            except (TypeError, json.JSONDecodeError):
                d[col] = []
    return d


# --------------------------------------------------------------------------- meta

def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# --------------------------------------------------------------------------- assets

def asset_index(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """folder_path -> {id, content_hash, n_embeddings} for incremental ingestion."""
    rows = conn.execute(
        """SELECT a.id, a.folder_path, a.content_hash,
                  (SELECT COUNT(*) FROM asset_embeddings e WHERE e.asset_id = a.id) AS n_embeddings
           FROM assets a"""
    ).fetchall()
    return {r["folder_path"]: dict(r) for r in rows}


def upsert_asset(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    values = []
    for col in ASSET_COLUMNS:
        v = record.get(col)
        if col in JSON_COLUMNS:
            v = json.dumps(v or [])
        elif col == "responsive" and v is not None:
            v = int(bool(v))
        values.append(v)
    placeholders = ", ".join("?" for _ in ASSET_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in ASSET_COLUMNS if c != "id")
    conn.execute(
        f"INSERT INTO assets ({', '.join(ASSET_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}, updated_at = datetime('now')",
        values,
    )


def replace_screenshots(
    conn: sqlite3.Connection, asset_id: str, screenshots: list[dict[str, Any]]
) -> list[int]:
    """Replace all screenshots of an asset; returns the new screenshot ids in order."""
    conn.execute("DELETE FROM asset_screenshots WHERE asset_id = ?", (asset_id,))
    ids = []
    for i, s in enumerate(screenshots):
        cur = conn.execute(
            "INSERT INTO asset_screenshots(asset_id, file_path, page_name, route, description, sort_order) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, s["file_path"], s.get("page_name"), s.get("route"), s.get("description"), i),
        )
        ids.append(cur.lastrowid)
    return ids


def delete_asset(conn: sqlite3.Connection, asset_id: str) -> list[int]:
    """Delete an asset (cascades); returns the FAISS ids that must be removed."""
    ids = embedding_ids_for(conn, asset_id)
    conn.execute("DELETE FROM assets_fts WHERE asset_id = ?", (asset_id,))
    conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    return ids


def get_asset(conn: sqlite3.Connection, asset_id: str) -> dict[str, Any] | None:
    return _decode(conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone())


def get_assets(conn: sqlite3.Connection, asset_ids: list[str] | None = None,
               asset_type: str | None = None) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM assets", []
    clauses = []
    if asset_ids is not None:
        if not asset_ids:
            return []
        clauses.append(f"id IN ({', '.join('?' for _ in asset_ids)})")
        params += asset_ids
    if asset_type:
        clauses.append("asset_type = ?")
        params.append(asset_type)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY level, name"
    return [_decode(r) for r in conn.execute(sql, params).fetchall()]


def get_screenshots(conn: sqlite3.Connection, asset_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {a: [] for a in asset_ids}
    if not asset_ids:
        return out
    rows = conn.execute(
        f"SELECT * FROM asset_screenshots WHERE asset_id IN ({', '.join('?' for _ in asset_ids)}) "
        "ORDER BY asset_id, sort_order",
        asset_ids,
    ).fetchall()
    for r in rows:
        out[r["asset_id"]].append(dict(r))
    return out


def counts_by_type(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT asset_type, COUNT(*) AS n FROM assets GROUP BY asset_type").fetchall()
    return {r["asset_type"]: r["n"] for r in rows}


# --------------------------------------------------------------------------- embeddings

def embedding_ids_for(conn: sqlite3.Connection, asset_id: str) -> list[int]:
    rows = conn.execute("SELECT id FROM asset_embeddings WHERE asset_id = ?", (asset_id,)).fetchall()
    return [r["id"] for r in rows]


def delete_embeddings(conn: sqlite3.Connection, asset_id: str) -> list[int]:
    ids = embedding_ids_for(conn, asset_id)
    conn.execute("DELETE FROM asset_embeddings WHERE asset_id = ?", (asset_id,))
    return ids


def insert_embedding(conn: sqlite3.Connection, asset_id: str, kind: str, text: str,
                     model: str, dim: int, screenshot_id: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO asset_embeddings(asset_id, screenshot_id, kind, searchable_text, embedding_model, dimension) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (asset_id, screenshot_id, kind, text, model, dim),
    )
    return cur.lastrowid


def all_embedding_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, asset_id, screenshot_id, kind, searchable_text FROM asset_embeddings ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def embedding_lookup(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    out: dict[int, dict[str, Any]] = {}
    # SQLite caps bound parameters; chunk to stay well below the limit.
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        rows = conn.execute(
            f"""SELECT e.id, e.asset_id, e.kind, s.page_name
                FROM asset_embeddings e LEFT JOIN asset_screenshots s ON s.id = e.screenshot_id
                WHERE e.id IN ({', '.join('?' for _ in chunk)})""",
            chunk,
        ).fetchall()
        out.update({r["id"]: dict(r) for r in rows})
    return out


# --------------------------------------------------------------------------- keyword search

def update_fts(conn: sqlite3.Connection, asset_id: str, name: str, description: str,
               tags: str, searchable_text: str) -> None:
    conn.execute("DELETE FROM assets_fts WHERE asset_id = ?", (asset_id,))
    conn.execute(
        "INSERT INTO assets_fts(asset_id, name, description, tags, searchable_text) VALUES (?, ?, ?, ?, ?)",
        (asset_id, name, description, tags, searchable_text),
    )


def keyword_search(conn: sqlite3.Connection, fts_query: str, limit: int = 200) -> dict[str, float]:
    """asset_id -> positive BM25 relevance (higher is better)."""
    if not fts_query:
        return {}
    try:
        rows = conn.execute(
            "SELECT asset_id, -bm25(assets_fts, 0.0, 4.0, 2.0, 3.0, 1.0) AS score "
            "FROM assets_fts WHERE assets_fts MATCH ? ORDER BY score DESC LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["asset_id"]: max(0.0, r["score"]) for r in rows}


# --------------------------------------------------------------------------- sessions / logs

def save_selection(conn: sqlite3.Connection, project_id: str, asset_id: str,
                   query: str, requirements: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO project_sessions(project_id, selected_asset_id, query, user_requirements, status)
           VALUES (?, ?, ?, ?, 'selected')
           ON CONFLICT(project_id) DO UPDATE SET
               selected_asset_id = excluded.selected_asset_id,
               query = excluded.query,
               user_requirements = excluded.user_requirements,
               status = 'selected',
               updated_at = datetime('now')""",
        (project_id, asset_id, query, json.dumps(requirements)),
    )


def list_selections(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT p.project_id, p.selected_asset_id, a.name AS asset_name, a.asset_type,
                  a.folder_path, p.query, p.user_requirements, p.status, p.updated_at
           FROM project_sessions p LEFT JOIN assets a ON a.id = p.selected_asset_id
           ORDER BY p.updated_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def log_search(conn: sqlite3.Connection, query: str, requirements: dict[str, Any],
               result_ids: list[str]) -> None:
    conn.execute(
        "INSERT INTO search_logs(query, requirements, result_ids) VALUES (?, ?, ?)",
        (query, json.dumps(requirements), json.dumps(result_ids)),
    )
