"""Three-stage retrieval.

Stage 1  Requirement extraction  - Claude turns the prompt into structured requirements
                                   (heuristic fallback if Bedrock is unavailable).
Stage 2  Hybrid retrieval        - metadata filter (asset type) + FAISS vector similarity
                                   + SQLite FTS5 keyword search  -> top-20 candidates.
Stage 3  Ranking                 - Score = 0.35 R + 0.25 D + 0.20 F + 0.10 Q + 0.10 C
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import db
from .config import ASSET_TYPE_LABELS, settings, similarity_range
from .llm import LLMError, extract_requirements
from .schemas import QueryRequirements
from .vectors import Embedder, VectorStore

WEIGHTS = {"R": 0.35, "D": 0.25, "F": 0.20, "Q": 0.10, "C": 0.10}
CANDIDATES = 20

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "want", "need", "looking", "like", "some", "into",
    "have", "has", "make", "build", "create", "please", "should", "would", "very", "more", "using",
    "template", "templates", "design", "looking", "style", "page", "website", "app",
}
STYLE_VOCAB = [
    "minimal", "minimalist", "modern", "premium", "luxury", "clean", "elegant", "playful", "bold",
    "corporate", "professional", "futuristic", "retro", "vintage", "brutalist", "glassmorphism",
    "neumorphism", "gradient", "colorful", "aesthetic", "flat", "dark", "light", "simple", "creative",
]
REACT_FAMILY = {"nextjs", "remix", "gatsby", "react"}


@dataclass
class SearchResult:
    asset: dict[str, Any]
    screenshots: list[dict[str, Any]]
    score: float
    breakdown: dict[str, float]
    vector_score: float
    keyword_score: float
    matched_page: str | None
    matched_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    strength: str = ""


@dataclass
class SearchResponse:
    query: str
    requirements: dict[str, Any]
    results: list[SearchResult]
    notes: list[str]
    used_llm: bool
    n_candidates: int
    hidden_better: list[SearchResult] = field(default_factory=list)
    type_filter: list[str] | None = None


def match_strength(score: float) -> str:
    if score >= 0.70:
        return "Strong match"
    if score >= 0.45:
        return "Partial match"
    return "Weak match"


# --------------------------------------------------------------------------- text helpers

def _norm(text: str) -> str:
    return re.sub(r"[\s_\-/.]+", " ", str(text).lower()).strip()


def _stem(word: str) -> str:
    return word if len(word) <= 4 else word[:-2]


def term_matches(term: str, haystack: str) -> bool:
    """Loose match: the whole phrase, or every meaningful word (crudely stemmed), appears in the haystack."""
    t = _norm(term)
    if not t:
        return False
    if t in haystack:
        return True
    words = [w for w in t.split() if len(w) > 2]
    return bool(words) and all(_stem(w) in haystack for w in words)


def _haystack(asset: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    parts = [
        asset.get("name"), asset.get("short_description"), asset.get("description"), asset.get("category"),
        asset.get("subcategory"), asset.get("layout"), *asset.get("design_style", []),
        *asset.get("features", []), *asset.get("ui_elements", []), *asset.get("keywords", []),
        *asset.get("use_cases", []), *(s.get("page_name") or "" for s in shots),
        *(s.get("description") or "" for s in shots),
    ]
    return " " + _norm(" ".join(p for p in parts if p)) + " "


def _fts_query(reqs: QueryRequirements) -> str:
    terms = [*reqs.keywords, *reqs.features, *reqs.ui_elements, *reqs.design_style, reqs.project_type]
    tokens = set()
    for term in terms:
        for w in re.split(r"[^a-z0-9]+", str(term).lower()):
            if len(w) > 2 and w not in STOPWORDS:
                tokens.add(w)
    return " OR ".join(f'"{w}"' for w in sorted(tokens))


def heuristic_requirements(query: str) -> QueryRequirements:
    """Rule-based Stage 1 used when Claude is disabled or unreachable."""
    q = query.lower()
    types = []
    if re.search(r"\b(component|button|input|dropdown|modal|tooltip|badge|avatar|toast)s?\b", q):
        types.append("ui_component")
    if re.search(r"\b(section|block|hero|footer|header|testimonials?|faq|cta|navbar|pricing table)s?\b", q):
        types.append("ui_block")
    if re.search(r"\b(page|login|signup|sign up|checkout|landing)s?\b", q):
        types.append("page_template")
    if re.search(r"\b(website|site|web app|application|project|template)s?\b", q):
        types.append("website_template")
    framework = [f for f, pat in (("nextjs", r"next\.?js|\bnext\b"), ("react", r"\breact\b"),
                                  ("vue", r"\bvue\b"), ("svelte", r"\bsvelte\b"),
                                  ("angular", r"\bangular\b")) if re.search(pat, q)]
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2 and w not in STOPWORDS]
    return QueryRequirements(
        asset_types=types,
        project_type="",
        theme="dark" if re.search(r"\bdark\b", q) else "light" if re.search(r"\blight\b", q) else "any",
        design_style=[s for s in STYLE_VOCAB if re.search(rf"\b{s}\b", q) and s not in ("dark", "light")],
        features=[],
        ui_elements=[],
        framework=framework,
        keywords=list(dict.fromkeys(words))[:12],
        semantic_query=query,
    )


# --------------------------------------------------------------------------- scoring

def _theme_score(want: str, have: str | None) -> float | None:
    if want in ("any", ""):
        return None
    if not have:
        return 0.5
    if have == want or have == "both":
        return 1.0
    return 0.4 if want == "both" else 0.0


def _coverage(terms: list[str], haystack: str) -> tuple[float | None, list[str], list[str]]:
    terms = [t for t in dict.fromkeys(terms) if t]
    if not terms:
        return None, [], []
    hit = [t for t in terms if term_matches(t, haystack)]
    miss = [t for t in terms if t not in hit]
    return len(hit) / len(terms), hit, miss


def _mean(values: list[float | None], default: float | None = 1.0) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else default


def _compat(reqs: QueryRequirements, asset: dict[str, Any]) -> float | None:
    parts: list[float | None] = []
    if reqs.framework:
        have = re.sub(r"[^a-z0-9]", "", (asset.get("framework") or "").lower())
        wanted = {re.sub(r"[^a-z0-9]", "", f.lower()) for f in reqs.framework}
        if not have:
            parts.append(0.5)  # unknown stack (screenshot-only asset)
        elif have in wanted or ("react" in wanted and have in REACT_FAMILY):
            parts.append(1.0)
        else:
            parts.append(0.0)
    if reqs.asset_types:
        parts.append(1.0 if asset["asset_type"] in reqs.asset_types else 0.0)
    return _mean(parts, default=None)


# --------------------------------------------------------------------------- engine

class SearchEngine:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self._store: VectorStore | None = None
        self._store_mtime: float | None = None

    def store(self) -> VectorStore:
        """Reload the FAISS index whenever ingestion (CLI or UI) rewrote it."""
        path = settings.faiss_path
        mtime = path.stat().st_mtime if path.exists() else None
        if self._store is None or mtime != self._store_mtime:
            self._store = VectorStore(self.embedder.dim)
            self._store_mtime = mtime
        return self._store

    def understand(self, query: str, use_llm: bool) -> tuple[QueryRequirements, list[str], bool]:
        if use_llm:
            try:
                return extract_requirements(query), [], True
            except LLMError as exc:
                return heuristic_requirements(query), [f"Claude unavailable, used rule-based parsing instead: {exc}"], False
        return heuristic_requirements(query), [], False

    def search(self, query: str, asset_types: list[str] | None = None, top_k: int = 8,
               use_llm: bool = True) -> SearchResponse:
        reqs, notes, used_llm = self.understand(query, use_llm)
        store = self.store()
        if store.size == 0:
            notes.append("The index is empty. Add screenshots to the library folders and run ingestion.")
            return SearchResponse(query, reqs.model_dump(), [], notes, used_llm, 0)

        with db.session() as conn:
            # ---- Stage 2a: vector similarity over every vector (exact search; fine at demo scale)
            qtext = query if reqs.semantic_query.strip() in ("", query) else f"{query}\n{reqs.semantic_query}"
            hits = store.search(self.embedder.embed_query(qtext), k=min(store.size, 10000))
            lookup = db.embedding_lookup(conn, [i for i, _ in hits])
            vec: dict[str, float] = {}
            page: dict[str, str | None] = {}
            for vid, score in hits:
                meta = lookup.get(vid)
                if not meta:
                    continue
                aid = meta["asset_id"]
                if score > vec.get(aid, -1.0):
                    vec[aid] = score
                    page[aid] = meta["page_name"] if meta["kind"] == "page" else None

            # ---- Stage 2b: keyword search (BM25 over FTS5)
            kw = db.keyword_search(conn, _fts_query(reqs))

            # ---- Absolute relevance (never relative to the filtered pool, so a bad pool scores low)
            lo, hi = similarity_range(self.embedder.model_name)
            max_kw = max(kw.values(), default=0.0) or 1.0   # best keyword hit in the WHOLE library
            pool = set(vec) | set(kw)
            relevance = {
                a: 0.8 * min(1.0, max(0.0, (vec.get(a, lo) - lo) / (hi - lo))) + 0.2 * kw.get(a, 0.0) / max_kw
                for a in pool
            }

            # ---- Stage 2c: metadata filter. Also keep the best unfiltered candidates, so we can tell
            # the user when the filter hides a clearly better match.
            type_of = {r["id"]: r["asset_type"] for r in conn.execute("SELECT id, asset_type FROM assets")}
            filtered = {a for a in pool if not asset_types or type_of.get(a) in asset_types}
            if not filtered:
                notes.append("No assets match the selected asset types.")
            by_relevance = lambda ids: sorted(ids, key=relevance.get, reverse=True)[:CANDIDATES]  # noqa: E731
            candidates = by_relevance(filtered)
            to_score = list(dict.fromkeys(candidates + by_relevance(pool - filtered)))

            # ---- Stage 3: rerank with  0.35R + 0.25D + 0.20F + 0.10Q + 0.10C
            assets = {a["id"]: a for a in db.get_assets(conn, to_score)}
            shots = db.get_screenshots(conn, to_score)
            scored: list[SearchResult] = []
            for aid in to_score:
                asset = assets.get(aid)
                if asset is None:
                    continue
                hay = _haystack(asset, shots[aid])
                style_cov, style_hit, style_miss = _coverage(reqs.design_style, hay)
                feature_cov, feat_hit, feat_miss = _coverage([*reqs.features, *reqs.ui_elements], hay)
                theme = _theme_score(reqs.theme, asset.get("theme"))
                factors: dict[str, float | None] = {
                    "R": relevance[aid],
                    "D": _mean([theme, style_cov], default=None),   # None = user stated no design wishes
                    "F": feature_cov,                                # None = no features requested
                    "Q": asset["quality_score"] if asset.get("quality_score") is not None else 0.5,
                    "C": _compat(reqs, asset),                       # None = nothing to check
                }
                # Factors the user didn't ask about are left out and the weights renormalised,
                # instead of counting them as a perfect 1.0 (which inflated every score).
                applicable = {k: v for k, v in factors.items() if v is not None}
                score = sum(WEIGHTS[k] * v for k, v in applicable.items()) / sum(WEIGHTS[k] for k in applicable)
                scored.append(SearchResult(
                    asset=asset, screenshots=shots[aid], score=score, breakdown=applicable,
                    vector_score=vec.get(aid, 0.0), keyword_score=kw.get(aid, 0.0) / max_kw,
                    matched_page=page.get(aid), matched_terms=style_hit + feat_hit,
                    missing_terms=style_miss + feat_miss, strength=match_strength(score),
                ))
            scored.sort(key=lambda r: r.score, reverse=True)
            results = [r for r in scored if r.asset["id"] in filtered][:top_k]
            best_shown = results[0].score if results else 0.0
            hidden = [r for r in scored if r.asset["id"] not in filtered
                      and r.score > best_shown and r.strength != "Weak match"][:3]

            if hidden:
                names = ", ".join(f"{r.asset['name']} ({ASSET_TYPE_LABELS[r.asset['asset_type']]}, "
                                  f"{r.score:.0%})" for r in hidden)
                notes.append(f"Your asset-type filter hides better matches: {names}.")
            elif results and results[0].strength == "Weak match":
                notes.append("Nothing in the library is a strong match for this request. "
                             "Add more items of this kind, or rephrase the query.")
            db.log_search(conn, query, reqs.model_dump(), [r.asset["id"] for r in results])

        return SearchResponse(query, reqs.model_dump(), results, notes, used_llm, len(candidates),
                              hidden_better=hidden, type_filter=asset_types)
