"""Walk the library folder: every subfolder of a category folder is one asset.

library/
  Web Templates/
    aurora-saas/              <- one asset
      01-home.png             <- screenshots (top level, or inside screenshots/)
      02-dashboard.png
      notes.txt               <- optional free-text hints for the model
      metadata.json           <- optional overrides (license, preview_url, framework, ...)
      package.json            <- optional; tech stack is detected from it
  Page Templates/ ...
  UI Blocks/ ...
  UI Components/ ...
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ASSET_FOLDERS, IMAGE_EXTENSIONS, settings


@dataclass
class ScannedItem:
    asset_type: str
    level: int
    folder: Path                  # absolute path
    rel_folder: str               # posix path relative to the library folder
    images: list[Path]
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    stack: dict[str, Any] = field(default_factory=dict)   # detected from package.json
    content_hash: str = ""

    @property
    def display_name(self) -> str:
        return re.sub(r"[-_]+", " ", self.folder.name).strip().title()


def ensure_library_dirs() -> None:
    for name in ASSET_FOLDERS:
        (settings.library_dir / name).mkdir(parents=True, exist_ok=True)


def _natural_key(p: Path) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _find_images(folder: Path) -> list[Path]:
    """Images at the item's top level plus an optional screenshots/ subfolder.

    Deliberately not recursive, so a dropped-in source tree (node_modules, public/ ...)
    doesn't flood the analysis with icons.
    """
    candidates = [p for p in folder.iterdir() if p.is_file()]
    shots = folder / "screenshots"
    if shots.is_dir():
        candidates += [p for p in shots.iterdir() if p.is_file()]
    images = [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]
    # cover.* first (shown on the card), mobile screenshots last, the rest in natural order.
    images.sort(key=lambda p: (p.stem.lower() != "cover", "mobile" in p.stem.lower(), _natural_key(p)))
    return images


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def detect_stack(folder: Path) -> dict[str, Any]:
    """Deterministic tech-stack detection from package.json (if the source is present)."""
    pkg_path = folder / "package.json"
    if not pkg_path.is_file():
        return {}
    pkg = _read_json(pkg_path)
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    names = set(deps)

    def has(*keys: str) -> bool:
        return any(k in names for k in keys)

    framework = None
    for keys, label in [
        (("next",), "Next.js"), (("@remix-run/react",), "Remix"), (("gatsby",), "Gatsby"),
        (("astro",), "Astro"), (("nuxt",), "Nuxt"), (("@sveltejs/kit",), "SvelteKit"),
        (("svelte",), "Svelte"), (("@angular/core",), "Angular"), (("vue",), "Vue"),
        (("react",), "React"),
    ]:
        if has(*keys):
            framework = label
            break

    styling = None
    for keys, label in [
        (("tailwindcss",), "Tailwind CSS"), (("styled-components",), "styled-components"),
        (("@emotion/react", "@emotion/styled"), "Emotion"), (("sass",), "SCSS"),
    ]:
        if has(*keys):
            styling = label
            break

    ui_library = None
    shadcn = (folder / "components.json").is_file() or (
        has("class-variance-authority") and any(n.startswith("@radix-ui/") for n in names)
    )
    for cond, label in [
        (shadcn, "shadcn/ui"), (has("@mui/material"), "MUI"), (has("@chakra-ui/react"), "Chakra UI"),
        (has("antd"), "Ant Design"), (has("@mantine/core"), "Mantine"),
        (has("@headlessui/react"), "Headless UI"), (has("daisyui"), "daisyUI"),
        (has("react-bootstrap", "bootstrap"), "Bootstrap"),
    ]:
        if cond:
            ui_library = label
            break

    language = "TypeScript" if has("typescript") or (folder / "tsconfig.json").is_file() else "JavaScript"
    out = {
        "framework": framework,
        "language": language,
        "styling": styling or "CSS",
        "ui_library": ui_library,
        "dependencies": sorted(names)[:60],
    }
    return {k: v for k, v in out.items() if v}


def _hash_item(images: list[Path], extra_files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in [*images, *extra_files]:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def scan_library() -> tuple[list[ScannedItem], list[str]]:
    """Return (items, warnings)."""
    ensure_library_dirs()
    items: list[ScannedItem] = []
    warnings: list[str] = []

    for folder_name, (asset_type, level) in ASSET_FOLDERS.items():
        category_dir = settings.library_dir / folder_name
        for entry in sorted(category_dir.iterdir(), key=_natural_key):
            if entry.is_file():
                if entry.suffix.lower() in IMAGE_EXTENSIONS:
                    warnings.append(
                        f"Skipped loose image '{folder_name}/{entry.name}': put each item in its own subfolder."
                    )
                continue
            if entry.name.startswith((".", "_")):
                continue
            images = _find_images(entry)
            if not images:
                warnings.append(f"Skipped '{folder_name}/{entry.name}': no screenshots found.")
                continue
            notes_path, meta_path, pkg_path = entry / "notes.txt", entry / "metadata.json", entry / "package.json"
            item = ScannedItem(
                asset_type=asset_type,
                level=level,
                folder=entry,
                rel_folder=entry.relative_to(settings.library_dir).as_posix(),
                images=images,
                notes=notes_path.read_text(encoding="utf-8", errors="ignore").strip() if notes_path.is_file() else "",
                metadata=_read_json(meta_path) if meta_path.is_file() else {},
                stack=detect_stack(entry),
                content_hash=_hash_item(images, [notes_path, meta_path, pkg_path]),
            )
            items.append(item)
    return items, warnings
