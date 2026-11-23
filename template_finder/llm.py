"""Claude Sonnet on Amazon Bedrock: screenshot analysis (ingestion) and query understanding (search)."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import os
import re
from urllib.parse import urlsplit

import anthropic
from anthropic import AnthropicBedrock, AnthropicBedrockMantle
from PIL import Image, UnidentifiedImageError

from .config import ASSET_TYPE_LABELS, settings
from .scanner import ScannedItem
from .schemas import AssetAnalysis, QueryRequirements

_client: AnthropicBedrock | AnthropicBedrockMantle | None = None


class LLMError(RuntimeError):
    """A Bedrock/Claude failure with a readable message.

    fatal=True means a setup problem (no credentials, wrong key, model or region) that will
    fail every request the same way, so batch jobs should stop instead of retrying each item.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


NO_CREDENTIALS = ("No AWS credentials found. Copy .env.example to .env and set AWS_BEARER_TOKEN_BEDROCK "
                  "(or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or AWS_PROFILE).")


def endpoint_info() -> dict[str, str]:
    """Which Bedrock endpoint/client will be used (no secrets)."""
    base = settings.bedrock_base_url or ""
    host = urlsplit(base).netloc if base else ""
    if "bedrock-runtime" in host:
        kind, url = "runtime", f"https://{host}"  # InvokeModel API: the SDK adds /model/{id}/invoke
    else:
        kind = "mantle"
        if base:
            path = re.sub(r"/v1(/messages)?/?$", "", urlsplit(base).path).rstrip("/")
            url = f"https://{host}{path or '/anthropic'}"
        else:
            url = f"https://bedrock-mantle.{settings.aws_region}.api.aws/anthropic"
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("ANTHROPIC_AWS_API_KEY"):
        auth = "Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)"
    elif os.getenv("AWS_ACCESS_KEY_ID"):
        auth = "IAM access keys (SigV4)"
    elif os.getenv("AWS_PROFILE"):
        auth = f"AWS profile '{os.getenv('AWS_PROFILE')}' (SigV4)"
    else:
        auth = "default AWS credential chain (SigV4)"
    return {"client": "AnthropicBedrock" if kind == "runtime" else "AnthropicBedrockMantle",
            "kind": kind, "url": url, "region": settings.aws_region,
            "model": settings.bedrock_model_id, "auth": auth}


def get_client() -> AnthropicBedrock | AnthropicBedrockMantle:
    """Pick the client from BEDROCK_BASE_URL.

    bedrock-runtime.<region>.amazonaws.com  -> AnthropicBedrock (InvokeModel API)
    bedrock-mantle.<region>.api.aws/...     -> AnthropicBedrockMantle (Messages API)  [default]
    Both read AWS_BEARER_TOKEN_BEDROCK automatically, else fall back to AWS SigV4 credentials.
    """
    global _client
    if _client is None:
        info = endpoint_info()
        try:
            if info["kind"] == "runtime":
                _client = AnthropicBedrock(aws_region=settings.aws_region, base_url=info["url"])
            else:
                _client = AnthropicBedrockMantle(aws_region=settings.aws_region, base_url=info["url"])
        except Exception as exc:  # missing region/credentials surface here as various types
            if "credential" in str(exc).lower():
                raise LLMError(NO_CREDENTIALS, fatal=True) from exc
            raise LLMError(f"Could not create the Bedrock client: {exc}", fatal=True) from exc
    return _client


# --------------------------------------------------------------------------- images

def _image_tiles(path: Path, max_tiles: int, max_edge: int) -> list[str]:
    """Return base64 JPEG tiles. Tall full-page screenshots are split top-to-bottom so text stays legible."""
    with Image.open(path) as img:
        img.seek(0)  # first frame of animated GIF/WebP
        img = img.convert("RGB")
    w, h = img.size
    tiles = [img]
    if max_tiles > 1 and h > 2.2 * w:
        n = min(max_tiles, math.ceil(h / (w * 1.6)))
        step = math.ceil(h / n)
        tiles = [img.crop((0, i * step, w, min(h, (i + 1) * step))) for i in range(n)]
    out = []
    for tile in tiles:
        tile.thumbnail((max_edge, max_edge))  # only ever shrinks, keeps aspect ratio
        buf = io.BytesIO()
        tile.save(buf, format="JPEG", quality=85, optimize=True)
        out.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    return out


def _image_blocks(images: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    max_blocks = max(1, settings.max_image_blocks)
    used = images[:max_blocks]
    per_image = max(1, min(4, max_blocks // max(1, len(used))))
    blocks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if len(images) > len(used):
        warnings.append(f"Only the first {len(used)} of {len(images)} screenshots were sent to the model.")
    for path in used:
        try:
            tiles = _image_tiles(path, per_image, settings.max_image_edge)
        except (UnidentifiedImageError, OSError) as exc:
            warnings.append(f"Could not read image {path.name}: {exc}")
            continue
        label = f"Screenshot file: {path.name}"
        if len(tiles) > 1:
            label += f" (tall page split into {len(tiles)} consecutive parts, top to bottom)"
        blocks.append({"type": "text", "text": label})
        for data in tiles:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    if not any(b["type"] == "image" for b in blocks):
        raise LLMError("None of the screenshots could be read.")
    return blocks, warnings


# --------------------------------------------------------------------------- calls

def _parse(system: str, content: list[dict[str, Any]] | str, schema: type, effort: str, max_tokens: int):
    try:
        response = get_client().messages.parse(
            model=settings.bedrock_model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort},
            output_format=schema,
        )
    except LLMError:
        raise
    except anthropic.AuthenticationError as exc:
        raise LLMError(f"Bedrock authentication failed - check AWS_BEARER_TOKEN_BEDROCK (or AWS keys) in .env "
                       f"({exc.message})", fatal=True) from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMError(f"Bedrock denied access - enable model access for {settings.bedrock_model_id} "
                       f"in {settings.aws_region}, and check the API key's permissions ({exc.message})",
                       fatal=True) from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(f"Model '{settings.bedrock_model_id}' not found at {endpoint_info()['url']} - check "
                       f"BEDROCK_MODEL_ID and the region in BEDROCK_BASE_URL ({exc.message})", fatal=True) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError(f"Bedrock rate limit hit, try again shortly ({exc.message})") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Bedrock API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"Could not reach Bedrock in {settings.aws_region}: {exc}", fatal=True) from exc
    except Exception as exc:  # e.g. the SDK's request signer raises RuntimeError when no credentials exist
        if "credential" in str(exc).lower():
            raise LLMError(NO_CREDENTIALS, fatal=True) from exc
        raise LLMError(f"Bedrock call failed: {type(exc).__name__}: {exc}") from exc

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMError(f"Model declined the request ({getattr(details, 'category', None) or 'refusal'}).")
    if response.stop_reason == "max_tokens":
        raise LLMError("Model output was cut off (max_tokens).")
    if response.parsed_output is None:
        raise LLMError("Model returned no structured output.")
    return response.parsed_output


ANALYSIS_SYSTEM = """You are a senior UI/UX designer cataloguing frontend design assets for a retrieval system. \
A coding agent will later pick assets from this catalog based on requests like "a dark, minimal SaaS dashboard with \
charts and a collapsible sidebar", so your metadata must be accurate, specific and searchable.

Asset levels:
- website_template: a complete multi-page project; each screenshot is usually a different page.
- page_template: one complete page.
- ui_block: one reusable page section (hero, pricing, testimonials, footer ...).
- ui_component: one reusable element (button, input, dropdown, modal, sidebar ...).

Describe what is visible in the screenshots plus any hints the owner supplied. Don't invent pages, features or \
tech-stack facts that aren't visible or stated. Use lowercase snake_case for category, subcategory, features and \
ui_elements so the vocabulary matches across assets. Be concrete about visual style: colour, density, typography, \
corner radius, shadows, imagery. quality_score reflects polish and consistency: 0.9+ premium and production-ready, \
about 0.5 average, below 0.3 rough."""


def analyze_asset(item: ScannedItem) -> tuple[AssetAnalysis, list[str]]:
    """Send an item's screenshots (+ optional hints) to Claude and get structured catalog metadata."""
    image_blocks, warnings = _image_blocks(item.images)
    hints = []
    if item.notes:
        hints.append(f"Owner notes:\n{item.notes[:4000]}")
    if item.metadata:
        hints.append(f"Owner-supplied metadata (authoritative):\n{json.dumps(item.metadata, indent=2)[:4000]}")
    if item.stack:
        stack = {k: v for k, v in item.stack.items() if k != "dependencies"}
        hints.append(f"Tech stack detected from package.json (authoritative): {json.dumps(stack)}")

    intro = (
        f"Asset type: {item.asset_type} ({ASSET_TYPE_LABELS[item.asset_type]}, level {item.level} of 4)\n"
        f"Folder name: {item.folder.name}\n"
        f"Number of screenshot files: {len(item.images)}\n"
    )
    if hints:
        intro += "\n" + "\n\n".join(hints) + "\n"

    content = [
        {"type": "text", "text": intro},
        *image_blocks,
        {"type": "text", "text": "Analyse the asset shown above and fill in every field of the schema."},
    ]
    analysis = _parse(ANALYSIS_SYSTEM, content, AssetAnalysis, settings.analysis_effort, 16000)
    return analysis, warnings


QUERY_SYSTEM = """You convert a user's request for frontend designs into structured search requirements for a catalog \
that holds complete website templates, single page templates, page sections (UI blocks) and single UI components.

Extract only what the user asked for or clearly implied; leave lists empty and theme='any' rather than guessing. \
Normalise vocabulary: lowercase, snake_case for features and UI elements (analytics_charts, collapsible_sidebar, \
authentication, data_table, pricing_tiers). In keywords, include close synonyms (charts -> graphs, analytics). \
Write semantic_query as a catalog-style description of the ideal asset."""


def extract_requirements(query: str) -> QueryRequirements:
    return _parse(QUERY_SYSTEM, query, QueryRequirements, settings.query_effort, 8000)
