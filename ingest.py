"""Run the ingestion pipeline from the command line.

    python ingest.py              # analyse new/changed items only
    python ingest.py --force      # re-analyse every item with Claude
    python ingest.py --reembed    # rebuild FAISS from stored text (no Claude calls)
    python ingest.py --scan       # just list what would be ingested
    python ingest.py --check      # test the Bedrock connection with one tiny request
"""

from __future__ import annotations

import argparse
import json
import sys

from template_finder.config import PROJECT_ROOT, settings
from template_finder.llm import LLMError, endpoint_info, extract_requirements
from template_finder.pipeline import run_ingestion
from template_finder.scanner import scan_library

# Never crash on a character the console can't show (e.g. Windows cp1252 terminals).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

ENV_HINT = "Note: no .env file found. Copy .env.example to .env and add your Bedrock credentials."


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the template library into SQLite + FAISS.")
    parser.add_argument("--force", action="store_true", help="re-analyse every item, even if unchanged")
    parser.add_argument("--reembed", action="store_true", help="re-embed stored text and rebuild FAISS (no LLM calls)")
    parser.add_argument("--scan", action="store_true", help="only scan the library and list items")
    parser.add_argument("--check", action="store_true", help="test the Bedrock connection with one small request")
    args = parser.parse_args()

    has_env = (PROJECT_ROOT / ".env").is_file()

    if args.check:
        if not has_env:
            print(ENV_HINT + "\n")
        info = endpoint_info()
        for k in ("client", "url", "region", "model", "auth"):
            print(f"{k:<7}: {info[k]}")
        print("\nSending a test query to Claude ...")
        try:
            reqs = extract_requirements("dark minimal SaaS dashboard with charts and a collapsible sidebar")
        except LLMError as exc:
            print(f"FAILED: {exc}")
            return 1
        print("OK - structured output received:")
        print(json.dumps(reqs.model_dump(), indent=2))
        return 0

    if args.scan:
        items, warnings = scan_library()
        for w in warnings:
            print(f"WARNING: {w}")
        for it in items:
            extras = [n for n, present in (("notes", it.notes), ("metadata", it.metadata), ("package.json", it.stack)) if present]
            print(f"- [{it.asset_type}] {it.rel_folder}: {len(it.images)} screenshot(s)"
                  + (f" + {', '.join(extras)}" if extras else ""))
        print(f"\n{len(items)} item(s) in {settings.library_dir}")
        return 0

    if not has_env and not args.reembed:
        print(ENV_HINT + "\n")
    print(f"Library : {settings.library_dir}")
    print(f"Storage : {settings.storage_dir}")
    print(f"Model   : {settings.bedrock_model_id} ({settings.aws_region})")
    print(f"Embedder: {settings.embedding_model}\n")
    report = run_ingestion(force=args.force, reembed=args.reembed, progress=print)
    if report.failed:
        print("\nFailed items:")
        for folder, err in report.failed:
            print(f"  - {folder}: {err}")
        if report.stopped:
            print("\nSetup problem - fix it, then run `python ingest.py` again (finished items are kept).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
