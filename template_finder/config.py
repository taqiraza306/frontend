"""Central configuration, read from environment variables / .env."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SETUP_HELP = (
    "Activate the project's virtual environment first:\n"
    "  Windows (Git Bash):  source .venv/Scripts/activate\n"
    "  Windows (PowerShell): .\\.venv\\Scripts\\Activate.ps1\n"
    "  macOS / Linux:       source .venv/bin/activate\n"
    "No .venv yet? See 'Setup after cloning' in README.md."
)
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required, but this is Python {sys.version.split()[0]}.\n{_SETUP_HELP}")
import importlib.util  # noqa: E402

_missing = [m for m in ("dotenv", "anthropic", "faiss", "sentence_transformers", "PIL", "pydantic", "numpy")
            if importlib.util.find_spec(m) is None]
if _missing:
    raise SystemExit(f"Missing packages in this Python ({sys.executable}): {', '.join(_missing)}.\n{_SETUP_HELP}\n"
                     "Then install them:  pip install -r requirements.txt")

from dotenv import load_dotenv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Folder name inside the library  ->  (asset_type, level)
# Level follows the 4-level asset model: 1 = complete project ... 4 = single element.
ASSET_FOLDERS: dict[str, tuple[str, int]] = {
    "Web Templates": ("website_template", 1),
    "Page Templates": ("page_template", 2),
    "UI Blocks": ("ui_block", 3),
    "UI Components": ("ui_component", 4),
}

ASSET_TYPE_LABELS: dict[str, str] = {
    "website_template": "Web Template",
    "page_template": "Page Template",
    "ui_block": "UI Block",
    "ui_component": "UI Component",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _path(env_name: str, default: str) -> Path:
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _region_from_url(url: str) -> str | None:
    m = re.search(r"bedrock(?:-mantle|-runtime)?(?:-fips)?\.([a-z]{2}(?:-gov)?-[a-z]+-\d)\.", url or "")
    return m.group(1) if m else None


def similarity_range(model: str) -> tuple[float, float]:
    """Cosine values that mean 'unrelated' (low) and 'excellent match' (high) for this embedding model.

    Used to turn raw cosine into an absolute 0-1 relevance, so a poor best-of-a-bad-bunch
    result never shows up as a high match. Override with EMBEDDING_SIM_RANGE=low,high.
    """
    env = os.getenv("EMBEDDING_SIM_RANGE")
    if env:
        try:
            lo, hi = (float(x) for x in env.split(","))
            if hi > lo:
                return lo, hi
        except ValueError:
            pass
    m = model.lower()
    if "bge" in m:
        return 0.50, 0.82
    if "e5" in m:
        return 0.72, 0.90
    if "minilm" in m or "mpnet" in m:
        return 0.15, 0.65
    return 0.20, 0.75


def _int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    library_dir: Path
    storage_dir: Path
    aws_region: str
    bedrock_base_url: str | None
    bedrock_model_id: str
    analysis_effort: str
    query_effort: str
    embedding_model: str
    max_image_blocks: int
    max_image_edge: int

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "catalog.db"

    @property
    def faiss_path(self) -> Path:
        return self.storage_dir / "faiss.index"


_base_url = (os.getenv("BEDROCK_BASE_URL") or "").strip() or None

settings = Settings(
    library_dir=_path("LIBRARY_DIR", "library"),
    storage_dir=_path("STORAGE_DIR", "storage"),
    aws_region=(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
                or _region_from_url(_base_url or "") or "us-east-1"),
    bedrock_base_url=_base_url,
    bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-5"),
    analysis_effort=os.getenv("ANALYSIS_EFFORT", "medium"),
    query_effort=os.getenv("QUERY_EFFORT", "low"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    max_image_blocks=_int("MAX_IMAGE_BLOCKS", 12),
    max_image_edge=_int("MAX_IMAGE_EDGE", 1920),
)
