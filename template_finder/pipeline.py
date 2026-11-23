"""Ingestion pipeline: scan folders -> Claude analysis -> SQLite metadata -> embeddings -> FAISS.

Incremental: an item is only re-analysed when its screenshots / notes / metadata change
(content hash), or when --force is used. Deleted folders are removed from SQLite and FAISS.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import db
from .config import ASSET_TYPE_LABELS, settings
from .llm import LLMError, analyze_asset
from .scanner import ScannedItem, scan_library
from .schemas import AssetAnalysis
from .vectors import Embedder, VectorStore

Progress = Callable[[str], None]

LIST_FIELDS = ("design_style", "features", "ui_elements", "color_palette", "use_cases", "keywords", "dependencies")
SCALAR_FIELDS = (
    "name", "short_description", "description", "category", "subcategory", "framework", "language",
    "styling", "ui_library", "theme", "license", "repository_url", "preview_url", "status",
    "quality_score", "responsive", "typography", "layout",
)


@dataclass
class IngestReport:
    scanned: int = 0
    analyzed: list[str] = field(default_factory=list)
    skipped: int = 0
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reindexed: bool = False
    vectors: int = 0
    stopped: str = ""   # set when a setup error (e.g. missing credentials) ended the run early

    def summary(self) -> str:
        return (
            f"{self.scanned} found, {len(self.analyzed)} analysed, {self.skipped} unchanged, "
            f"{len(self.removed)} removed, {len(self.failed)} failed, {self.vectors} vectors in FAISS"
        )


def asset_id_for(rel_folder: str) -> str:
    """Stable id derived from the folder path, so ids survive a full rebuild."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"template-finder:{rel_folder}"))


def _known(value: Any) -> Any:
    if value is None or str(value).strip().lower() in {"", "unknown", "n/a", "none", "not visible"}:
        return None
    return value


def _dedupe(values: list[Any]) -> list[Any]:
    seen, out = set(), []
    for v in values:
        key = str(v).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _human(term: str) -> str:
    return str(term).replace("_", " ").strip()


def _match_screenshots(item: ScannedItem, analysis: AssetAnalysis) -> list[dict[str, Any]]:
    """Attach the model's per-screenshot info to the real files (by name, then stem, then position)."""
    infos = analysis.screenshots
    by_name = {s.file_name.strip().lower(): s for s in infos}
    by_stem = {Path(s.file_name.strip().lower()).stem: s for s in infos}
    shots = []
    for i, path in enumerate(item.images):
        info = by_name.get(path.name.lower()) or by_stem.get(path.stem.lower())
        if info is None and len(infos) == len(item.images):
            info = infos[i]
        shots.append({
            "file_path": path.relative_to(settings.library_dir).as_posix(),
            "page_name": info.page_name if info else _human(path.stem).title(),
            "route": info.route if info else "",
            "description": info.description if info else "",
        })
    return shots


def build_record(item: ScannedItem, analysis: AssetAnalysis) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge model output < package.json stack < metadata.json overrides into one DB record."""
    a = analysis
    record: dict[str, Any] = {
        "id": asset_id_for(item.rel_folder),
        "asset_type": item.asset_type,
        "level": item.level,
        "name": a.name.strip() or item.display_name,
        "folder_path": item.rel_folder,
        "short_description": a.short_description,
        "description": a.description,
        "category": a.category,
        "subcategory": a.subcategory,
        "framework": _known(a.likely_framework),
        "language": None,
        "styling": _known(a.likely_styling),
        "ui_library": _known(a.likely_ui_library),
        "theme": a.theme,
        "design_style": _dedupe([s.lower() for s in a.design_style]),
        "features": _dedupe(a.features),
        "ui_elements": _dedupe(a.ui_elements),
        "color_palette": a.color_palette,
        "typography": a.typography,
        "layout": a.layout,
        "responsive": a.responsive_hint,
        "use_cases": a.use_cases,
        "keywords": _dedupe([k.lower() for k in a.keywords]),
        "dependencies": [],
        "repository_url": None,
        "preview_url": None,
        "license": None,
        "status": "pending",
        "quality_score": max(0.0, min(1.0, float(a.quality_score))),
        "content_hash": item.content_hash,
        "analysis_model": settings.bedrock_model_id,
        "raw_analysis": a.model_dump_json(),
    }
    record.update(item.stack)  # package.json is more reliable than a guess from pixels
    for key, value in item.metadata.items():  # metadata.json is authoritative
        if key in LIST_FIELDS and isinstance(value, list):
            record[key] = _dedupe([*value, *(record.get(key) or [])])
        elif key in SCALAR_FIELDS:
            record[key] = value
    return record, _match_screenshots(item, analysis)


def build_searchable_texts(record: dict[str, Any], shots: list[dict[str, Any]]) -> list[tuple[str, int | None, str]]:
    """Return [(kind, screenshot_index, text)]. One rich asset-level text, plus one per page/screenshot.

    The asset text combines description, design characteristics, features, pages and stack,
    not just the name, which makes semantic retrieval far more useful.
    """
    label = ASSET_TYPE_LABELS[record["asset_type"]]
    stack = ", ".join(v for v in (record.get("framework"), record.get("language"),
                                  record.get("styling"), record.get("ui_library")) if v)
    pages = "; ".join(s["page_name"] for s in shots if s.get("page_name"))
    lines = [
        f"{record['name']}: {label} ({_human(record.get('subcategory') or '')}) for {_human(record.get('category') or '')}.",
        record.get("short_description") or "",
        record.get("description") or "",
        f"Theme: {record.get('theme')}. Design style: {', '.join(record['design_style'])}.",
        f"Features: {', '.join(_human(f) for f in record['features'])}." if record["features"] else "",
        f"UI elements: {', '.join(_human(e) for e in record['ui_elements'])}." if record["ui_elements"] else "",
        f"Layout: {record.get('layout') or ''}. Typography: {record.get('typography') or ''}.",
        f"Use cases: {'; '.join(record['use_cases'])}." if record["use_cases"] else "",
        f"Pages: {pages}." if pages and record["asset_type"] in ("website_template", "page_template") else "",
        f"Tech stack: {stack}." if stack else "",
        f"Keywords: {', '.join(record['keywords'])}." if record["keywords"] else "",
    ]
    texts: list[tuple[str, int | None, str]] = [("asset", None, "\n".join(l for l in lines if l))]

    if len(shots) > 1:  # separate page-level records, e.g. "has a pricing page"
        styles = ", ".join(record["design_style"][:4])
        for i, s in enumerate(shots):
            if s.get("description"):
                texts.append((
                    "page", i,
                    f"{s['page_name']} ({label}: {record['name']}, {_human(record.get('category') or '')}, "
                    f"{record.get('theme')} theme, {styles}). {s['description']}",
                ))
    return texts


def _tags_text(record: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    terms = [
        record["asset_type"], ASSET_TYPE_LABELS[record["asset_type"]], record.get("category"),
        record.get("subcategory"), record.get("theme"), record.get("framework"), record.get("styling"),
        record.get("ui_library"), *record["design_style"], *record["features"], *record["ui_elements"],
        *record["keywords"], *(s["page_name"] for s in shots),
    ]
    return " ".join(_human(t) for t in terms if t)


def _index_asset(conn, store: VectorStore, embedder: Embedder,
                 record: dict[str, Any], shots: list[dict[str, Any]]) -> None:
    asset_id = record["id"]
    db.upsert_asset(conn, record)
    # Drop old vectors before screenshots: page embeddings reference screenshot rows.
    store.remove(db.delete_embeddings(conn, asset_id))
    shot_ids = db.replace_screenshots(conn, asset_id, shots)
    texts = build_searchable_texts(record, shots)
    vectors = embedder.embed_documents([t for _, _, t in texts])
    new_ids = [
        db.insert_embedding(conn, asset_id, kind, text, embedder.model_name, embedder.dim,
                            shot_ids[idx] if idx is not None else None)
        for kind, idx, text in texts
    ]
    store.add(new_ids, vectors)
    db.update_fts(conn, asset_id, record["name"], record.get("description") or "",
                  _tags_text(record, shots), texts[0][2])


def rebuild_index(conn, store: VectorStore, embedder: Embedder) -> int:
    """Re-embed every stored searchable_text (no LLM calls) and rebuild FAISS from scratch."""
    rows = db.all_embedding_rows(conn)
    store.reset()
    for i in range(0, len(rows), 256):
        batch = rows[i:i + 256]
        store.add([r["id"] for r in batch], embedder.embed_documents([r["searchable_text"] for r in batch]))
    conn.execute("UPDATE asset_embeddings SET embedding_model = ?, dimension = ?",
                 (embedder.model_name, embedder.dim))
    store.save()
    return len(rows)


def run_ingestion(force: bool = False, reembed: bool = False, embedder: Embedder | None = None,
                  progress: Progress | None = None) -> IngestReport:
    log = progress or (lambda _msg: None)
    report = IngestReport()

    items, warnings = scan_library()
    report.scanned = len(items)
    report.warnings = warnings
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"Found {len(items)} item(s) in {settings.library_dir}")

    if embedder is None:
        log(f"Loading embedding model {settings.embedding_model} ...")
        embedder = Embedder()
    store = VectorStore(embedder.dim)

    with db.session() as conn:
        previous_model = db.get_meta(conn, "embedding_model")
        if previous_model and previous_model != embedder.model_name:
            log(f"Embedding model changed ({previous_model} -> {embedder.model_name}); all vectors will be rebuilt.")
            reembed = True
        db.set_meta(conn, "embedding_model", embedder.model_name)

        existing = db.asset_index(conn)
        present = {it.rel_folder for it in items}
        for folder, row in existing.items():
            if folder not in present:
                store.remove(db.delete_asset(conn, row["id"]))
                report.removed.append(folder)
                log(f"Removed {folder} (folder no longer exists)")
        conn.commit()
        store.save()

        for n, item in enumerate(items, 1):
            row = existing.get(item.rel_folder)
            if not force and row and row["content_hash"] == item.content_hash and row["n_embeddings"] > 0:
                report.skipped += 1
                continue
            log(f"[{n}/{len(items)}] Analysing {item.rel_folder} ({len(item.images)} screenshot(s)) with Claude ...")
            try:
                analysis, warns = analyze_asset(item)
                for w in warns:
                    log(f"  WARNING: {w}")
                record, shots = build_record(item, analysis)
                _index_asset(conn, store, embedder, record, shots)
                conn.commit()
                store.save()
            except LLMError as exc:
                conn.rollback()
                report.failed.append((item.rel_folder, str(exc)))
                log(f"  FAILED: {exc}")
                if exc.fatal:  # setup problem: every remaining item would fail the same way
                    left = len(items) - n
                    if left:
                        msg = f"Stopped early: {left} more item(s) not analysed. Fix the problem above and run again."
                        report.warnings.append(msg)
                        log(msg)
                    report.stopped = str(exc)
                    break
                continue
            except Exception as exc:  # keep the batch going; report the item
                conn.rollback()
                report.failed.append((item.rel_folder, f"{type(exc).__name__}: {exc}"))
                log(f"  FAILED: {type(exc).__name__}: {exc}")
                continue
            report.analyzed.append(item.rel_folder)
            log(f"  OK: '{record['name']}' | {record['theme']} | {', '.join(record['design_style'][:4])}")

        # SQLite is the source of truth; repair FAISS if it drifted (crash, manual edits, model change).
        db_ids = {r["id"] for r in db.all_embedding_rows(conn)}
        if reembed or db_ids != store.ids():
            log("Rebuilding FAISS index from SQLite searchable_text ...")
            rebuild_index(conn, store, embedder)
            report.reindexed = True
        report.vectors = store.size

    log(f"Done: {report.summary()}")
    return report
