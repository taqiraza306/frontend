"""Capture screenshots of live template demos straight into the library folders.

Batch (recommended):
    python capture.py capture_list.json            # skips items that already have screenshots
    python capture.py capture_list.json --overwrite

Single item:
    python capture.py --type web --name taxonomy --url https://tx.shadcn.com --url https://tx.shadcn.com/pricing \
        --repo https://github.com/shadcn-ui/taxonomy --license MIT --mobile
    python capture.py --type block --name taxonomy-hero --url https://tx.shadcn.com --selector "main section"

Each item becomes  library/<Type Folder>/<name>/  with 01-home.png, 02-pricing.png ... and a metadata.json
(repository_url, preview_url, license), ready for `python ingest.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from template_finder.config import ASSET_FOLDERS, IMAGE_EXTENSIONS, settings

try:
    from playwright.sync_api import Browser, Page, sync_playwright
    from playwright.sync_api import TimeoutError as PWTimeout
except ImportError:  # pragma: no cover
    sys.exit("Playwright is missing. Run:  pip install playwright  &&  python -m playwright install chromium")

for _stream in (sys.stdout, sys.stderr):  # never crash printing on a limited console encoding
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

TYPE_TO_FOLDER: dict[str, str] = {}
for _folder, (_atype, _level) in ASSET_FOLDERS.items():
    TYPE_TO_FOLDER[_folder.lower()] = _folder
    TYPE_TO_FOLDER[_atype] = _folder
TYPE_TO_FOLDER.update({"web": "Web Templates", "website": "Web Templates", "template": "Web Templates",
                       "page": "Page Templates", "block": "UI Blocks", "section": "UI Blocks",
                       "component": "UI Components", "element": "UI Components"})

DESKTOP_VIEWPORT = {"width": 1440, "height": 900}

# Applied only while the screenshot is taken. Common cookie/consent banners and chat widgets.
HIDE_SELECTORS = [
    "#onetrust-banner-sdk", "#onetrust-consent-sdk", "#CybotCookiebotDialog", ".cc-window", ".cookie-banner",
    "#cookie-banner", "[id^='cookie-consent']", "[class*='cookieconsent']", "#intercom-container",
    ".intercom-lightweight-app", "#crisp-chatbox", "[data-nextjs-toast]", "nextjs-portal",
]

SCROLL_JS = """async (maxSteps) => {
    // Scroll through the page so lazy images load and scroll-triggered animations play.
    const step = Math.max(200, Math.floor(window.innerHeight * 0.8));
    let y = 0;
    for (let i = 0; i < maxSteps; i++) {
        if (y >= document.documentElement.scrollHeight) break;
        window.scrollTo(0, y);
        y += step;
        await new Promise(r => setTimeout(r, 150));
    }
    window.scrollTo(0, 0);
    await new Promise(r => setTimeout(r, 300));
}"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def default_shot_name(index: int, url: str) -> str:
    path = urlsplit(url).path.strip("/")
    return f"{index:02d}-{slugify(path.split('/')[-1]) if path else 'home'}"


@dataclass
class Shot:
    url: str
    name: str
    selector: str | None = None
    click: list[str] = field(default_factory=list)


@dataclass
class CaptureItem:
    folder: str
    name: str
    shots: list[Shot]
    repository_url: str | None = None
    preview_url: str | None = None
    license: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mobile: bool = False
    color_scheme: str | None = None   # "dark" / "light" -> prefers-color-scheme
    hide: list[str] = field(default_factory=list)
    wait_ms: int = 800


def parse_item(raw: dict[str, Any]) -> CaptureItem:
    type_key = str(raw.get("type", "")).strip().lower()
    folder = TYPE_TO_FOLDER.get(type_key)
    if not folder:
        raise ValueError(f"unknown type '{raw.get('type')}' (use web, page, block or component)")
    urls = raw.get("urls") or ([raw["url"]] if raw.get("url") else [])
    if not urls:
        raise ValueError("item has no 'url' or 'urls'")

    shots: list[Shot] = []
    used: set[str] = set()
    for i, entry in enumerate(urls, 1):
        entry = {"url": entry} if isinstance(entry, str) else dict(entry)
        name = slugify(entry.get("name") or default_shot_name(i, entry["url"]))
        while name in used:
            name += "-x"
        used.add(name)
        click = entry.get("click", raw.get("click", []))
        shots.append(Shot(
            url=entry["url"], name=name,
            selector=entry.get("selector", raw.get("selector")),
            click=[click] if isinstance(click, str) else list(click),
        ))

    first = shots[0].url
    return CaptureItem(
        folder=folder,
        name=slugify(raw.get("name") or (urlsplit(first).netloc + urlsplit(first).path)),
        shots=shots,
        repository_url=raw.get("repository_url") or raw.get("repo"),
        preview_url=raw.get("preview_url") or first,
        license=raw.get("license"),
        notes=raw.get("notes"),
        metadata=dict(raw.get("metadata") or {}),
        mobile=bool(raw.get("mobile", False)),
        color_scheme=raw.get("color_scheme") or ("dark" if raw.get("dark") else None),
        hide=list(raw.get("hide") or []),
        wait_ms=int(raw.get("wait_ms", 800)),
    )


# --------------------------------------------------------------------------- capture

class CaptureError(RuntimeError):
    pass


def _open(page: Page, url: str, wait_ms: int) -> None:
    response = page.goto(url, wait_until="load", timeout=60_000)
    if response is not None and response.status >= 400:
        # e.g. 404, or Vercel's 402 "Deployment Paused": never save an error page as a template
        raise CaptureError(f"HTTP {response.status} ({page.title()[:60] or 'error page'})")
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except PWTimeout:
        pass  # sites with analytics/websockets never go idle; carry on
    page.evaluate(SCROLL_JS, 60)
    page.wait_for_timeout(wait_ms)


def _shoot(page: Page, shot: Shot, path: Path, hide_css: str) -> None:
    for sel in shot.click:  # e.g. open a dropdown or modal before capturing
        page.locator(sel).first.click(timeout=10_000)
        page.wait_for_timeout(600)
    if shot.selector:
        target = page.locator(shot.selector).first
        target.scroll_into_view_if_needed(timeout=20_000)
        page.wait_for_timeout(700)  # lazy iframes/images render after they enter the viewport
        target.screenshot(path=path, animations="disabled", style=hide_css, timeout=20_000)
    else:
        page.screenshot(path=path, full_page=True, animations="disabled", style=hide_css, timeout=60_000)


def capture_item(browser: Browser, item: CaptureItem, mobile_device: dict[str, Any],
                 overwrite: bool) -> tuple[str, list[str], list[str]]:
    """Returns (status, saved_files, errors). status: captured | partial | skipped | failed."""
    out = settings.library_dir / item.folder / item.name
    if out.is_dir() and not overwrite and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in out.iterdir()):
        return "skipped", [], []
    out.mkdir(parents=True, exist_ok=True)
    hide_css = ", ".join(HIDE_SELECTORS + item.hide) + " { display: none !important; }"
    saved, errors = [], []

    ctx_opts: dict[str, Any] = {"viewport": DESKTOP_VIEWPORT, "device_scale_factor": 1}
    if item.color_scheme:
        ctx_opts["color_scheme"] = item.color_scheme
    ctx = browser.new_context(**ctx_opts)
    for shot in item.shots:
        path = out / f"{shot.name}.png"
        page = ctx.new_page()  # fresh tab per page: a failed load can't poison the next one
        try:
            _open(page, shot.url, item.wait_ms)
            _shoot(page, shot, path, hide_css)
            saved.append(path.name)
        except Exception as exc:  # keep going with the other pages/items
            errors.append(f"{shot.url}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        finally:
            page.close()
    ctx.close()

    if item.mobile:
        first = item.shots[0]
        m_opts = {k: v for k, v in mobile_device.items() if k != "default_browser_type"}
        if item.color_scheme:
            m_opts["color_scheme"] = item.color_scheme
        mctx = browser.new_context(**m_opts)
        mpage = mctx.new_page()
        path = out / f"{first.name}-mobile.png"
        try:
            _open(mpage, first.url, item.wait_ms)
            _shoot(mpage, first, path, hide_css)
            saved.append(path.name)
        except Exception as exc:
            errors.append(f"{first.url} (mobile): {type(exc).__name__}: {str(exc).splitlines()[0]}")
        mctx.close()

    if saved:
        _write_metadata(out, item)
    status = "failed" if not saved else ("partial" if errors else "captured")
    if not saved and not any(out.iterdir()):
        out.rmdir()  # don't leave an empty folder behind (the scanner would warn about it)
    return status, saved, errors


def _write_metadata(out: Path, item: CaptureItem) -> None:
    meta_path = out / "metadata.json"
    existing: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    fresh = {k: v for k, v in {
        "repository_url": item.repository_url,
        "preview_url": item.preview_url,
        "license": item.license,
    }.items() if v}
    fresh["source_urls"] = [s.url for s in item.shots]
    merged = {**existing, **item.metadata, **fresh}
    meta_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    if item.notes:
        (out / "notes.txt").write_text(item.notes, encoding="utf-8")


# --------------------------------------------------------------------------- CLI

def _items_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.list_file:
        data = json.loads(Path(args.list_file).read_text(encoding="utf-8"))
        return data["items"] if isinstance(data, dict) else data
    if not (args.type and args.url):
        raise SystemExit("Give a list file, or at least --type and --url. See: python capture.py --help")
    return [{
        "type": args.type, "name": args.name, "urls": args.url, "selector": args.selector,
        "click": args.click or [], "repository_url": args.repo, "preview_url": args.preview,
        "license": args.license, "mobile": args.mobile, "dark": args.dark, "hide": args.hide or [],
        "wait_ms": args.wait,
    }]


def main() -> int:
    ap = argparse.ArgumentParser(description="Screenshot live demos into the template library.")
    ap.add_argument("list_file", nargs="?", help="JSON file with a list of items (see capture_list.example.json)")
    ap.add_argument("--type", help="web | page | block | component")
    ap.add_argument("--name", help="item folder name (default: from the URL)")
    ap.add_argument("--url", action="append", help="page URL; repeat for multiple pages")
    ap.add_argument("--selector", help="CSS selector: capture only this element (blocks/components)")
    ap.add_argument("--click", action="append", help="CSS selector to click before capturing; repeatable")
    ap.add_argument("--repo", help="source repository URL")
    ap.add_argument("--preview", help="live preview URL (default: first URL)")
    ap.add_argument("--license", help="license, e.g. MIT")
    ap.add_argument("--mobile", action="store_true", help="also capture the first URL at phone size")
    ap.add_argument("--dark", action="store_true", help="emulate prefers-color-scheme: dark")
    ap.add_argument("--hide", action="append", help="extra CSS selector to hide (banners, popups); repeatable")
    ap.add_argument("--wait", type=int, default=800, help="extra wait in ms after loading (default 800)")
    ap.add_argument("--overwrite", action="store_true", help="re-capture items that already have screenshots")
    ap.add_argument("--headed", action="store_true", help="show the browser window (debugging)")
    args = ap.parse_args()

    raw_items = _items_from_args(args)
    items: list[CaptureItem] = []
    for n, raw in enumerate(raw_items, 1):
        try:
            items.append(parse_item(raw))
        except (ValueError, KeyError) as exc:
            print(f"Item {n} ({raw.get('name', '?')}): {exc}")
    if not items:
        return 1

    totals = {"captured": 0, "partial": 0, "skipped": 0, "failed": 0}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=not args.headed)
        except Exception as exc:  # most often: the browser was never downloaded on this machine
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            print(f"Could not start Chromium: {first}")
            print("Run this once, then try again:  python -m playwright install chromium")
            return 1
        mobile_device = p.devices["iPhone 13"]
        for n, item in enumerate(items, 1):
            print(f"[{n}/{len(items)}] {item.folder}/{item.name} ({len(item.shots)} page(s))")
            status, saved, errors = capture_item(browser, item, mobile_device, args.overwrite)
            totals[status] += 1
            if status == "skipped":
                print("   skipped (already has screenshots; use --overwrite to re-capture)")
            for f in saved:
                print(f"   saved {f}")
            for e in errors:
                print(f"   ERROR {e}")
        browser.close()

    print("\n" + ", ".join(f"{v} {k}" for k, v in totals.items()))
    print(f"Next: review the images in {settings.library_dir}, then run  python ingest.py")
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
