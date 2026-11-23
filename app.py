"""Streamlit front end: describe what you need, get the closest templates / pages / blocks / components.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import streamlit as st

from template_finder import db
from template_finder.config import ASSET_FOLDERS, ASSET_TYPE_LABELS, settings
from template_finder.llm import endpoint_info
from template_finder.pipeline import run_ingestion
from template_finder.scanner import scan_library
from template_finder.search import WEIGHTS, SearchEngine, SearchResult
from template_finder.vectors import Embedder

st.set_page_config(page_title="Template Finder", page_icon="🧩", layout="wide")

TYPE_ORDER = list(ASSET_TYPE_LABELS)
EXAMPLES = [
    "I want a dark, minimal, premium-looking AI SaaS dashboard with charts and a collapsible sidebar",
    "Playful pricing section with three tiers and a monthly / yearly toggle",
    "Clean light login page with social sign-in buttons",
]
STATUS_LABEL = {"verified": "✅ Verified", "pending": "🕓 Pending review", "rejected": "⛔ Rejected"}
FACTOR_LABEL = {"R": "Semantic relevance", "D": "Design similarity", "F": "Feature coverage",
                "Q": "Quality", "C": "Compatibility"}
HEX = re.compile(r"^#[0-9a-fA-F]{3,8}$")
PAGE_SIZE = 24


@st.cache_resource(show_spinner="Loading embedding model ...")
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner=False)
def get_engine() -> SearchEngine:
    return SearchEngine(get_embedder())


st.session_state.setdefault("project_id", f"project_{uuid.uuid4().hex[:6]}")
st.session_state.setdefault("query", "")
st.session_state.setdefault("type_filter", TYPE_ORDER)
STRENGTH_ICON = {"Strong match": "🟢", "Partial match": "🟡", "Weak match": "🔴"}


# --------------------------------------------------------------------------- helpers

def _human(v: Any) -> str:
    return str(v or "").replace("_", " ").strip()


def _chips(values: list[Any], color: str, limit: int = 6) -> str:
    clean = [_human(v).replace("[", "").replace("]", "") for v in values[:limit] if v]
    return " ".join(f":{color}-badge[{c}]" for c in clean)


def _swatches(colors: list[str]) -> str:
    return "".join(
        f'<span title="{c}" style="display:inline-block;width:22px;height:22px;border-radius:5px;'
        f'margin-right:4px;border:1px solid rgba(128,128,128,.35);background:{c}"></span>'
        for c in colors if HEX.match(str(c))
    )


def _select(asset: dict[str, Any], query: str, requirements: dict[str, Any]) -> None:
    pid = st.session_state.project_id.strip() or "project_default"
    with db.session() as conn:
        db.save_selection(conn, pid, asset["id"], query, requirements)
    st.toast(f"Selected '{asset['name']}' for {pid}", icon="✅")


def render_card(asset: dict[str, Any], shots: list[dict[str, Any]], key: str,
                result: SearchResult | None = None, query: str = "",
                requirements: dict[str, Any] | None = None) -> None:
    with st.container(border=True):
        if shots:
            cover = settings.library_dir / shots[0]["file_path"]
            if cover.exists():
                st.image(str(cover), width="stretch")

        st.markdown(f"#### {asset['name']}")
        st.caption(" · ".join(x for x in (
            ASSET_TYPE_LABELS[asset["asset_type"]], _human(asset.get("subcategory")),
            _human(asset.get("category")), f"{asset.get('theme')} theme") if x))
        if result is not None:
            st.progress(min(1.0, max(0.0, result.score)),
                        text=f"{STRENGTH_ICON.get(result.strength, '')} {result.strength} · **{result.score:.0%}**")
        st.write(asset.get("short_description") or "")
        if asset.get("design_style"):
            st.markdown(_chips(asset["design_style"], "violet"))
        if asset.get("features"):
            st.markdown(_chips(asset["features"], "blue"))

        stack = " · ".join(v for v in (asset.get("framework"), asset.get("language"),
                                       asset.get("styling"), asset.get("ui_library")) if v)
        st.caption(f"**Stack:** {stack or 'not specified (screenshots only)'}")
        if asset["asset_type"] in ("website_template", "page_template") and len(shots) > 1:
            st.caption("**Pages:** " + ", ".join(s["page_name"] or "?" for s in shots))
        quality = asset.get("quality_score")
        st.caption(" · ".join([
            STATUS_LABEL.get(asset.get("status") or "pending", asset.get("status") or ""),
            f"Quality {quality:.2f}" if quality is not None else "Quality n/a",
            "📱 Responsive" if asset.get("responsive") else "🖥️ Desktop-first",
        ]))
        if result is not None and result.matched_page:
            st.caption(f"🎯 Best-matching page: **{result.matched_page}**")

        with st.expander("Details & screenshots"):
            if result is not None:
                st.markdown("**Why this result**")
                for k, v in result.breakdown.items():
                    st.markdown(f"- {FACTOR_LABEL[k]} ({k}, weight {WEIGHTS[k]:.2f}): `{v:.2f}`")
                st.caption(f"Vector cosine {result.vector_score:.3f} · keyword BM25 (normalised) {result.keyword_score:.2f}")
                if result.matched_terms:
                    st.markdown("Matched: " + _chips(result.matched_terms, "green", 12))
                if result.missing_terms:
                    st.markdown("Missing: " + _chips(result.missing_terms, "orange", 12))
                st.divider()
            st.write(asset.get("description") or "")
            if asset.get("layout"):
                st.markdown(f"**Layout:** {asset['layout']}")
            if asset.get("typography"):
                st.markdown(f"**Typography:** {asset['typography']}")
            if asset.get("ui_elements"):
                st.markdown("**UI elements:** " + _chips(asset["ui_elements"], "gray", 15))
            if asset.get("color_palette"):
                st.markdown(_swatches(asset["color_palette"]), unsafe_allow_html=True)
            for s in shots:
                path = settings.library_dir / s["file_path"]
                if path.exists():
                    cap = s.get("page_name") or path.name
                    if s.get("route"):
                        cap += f" ({s['route']})"
                    st.image(str(path), caption=cap, width="stretch")
                    if s.get("description"):
                        st.caption(s["description"])
            meta = [f"Folder: `{settings.library_dir.name}/{asset['folder_path']}`"]
            if asset.get("license"):
                meta.append(f"License: {asset['license']}")
            if asset.get("repository_url"):
                meta.append(f"[Repository]({asset['repository_url']})")
            st.caption(" · ".join(meta))

        c1, c2 = st.columns(2)
        with c1:
            if asset.get("preview_url"):
                st.link_button("Live preview", asset["preview_url"], width="stretch")
            else:
                st.button("No preview link", key=f"pv-{key}", disabled=True, width="stretch",
                          help="Add \"preview_url\" to this item's metadata.json, then run python ingest.py")
        with c2:
            st.button("Select", key=f"sel-{key}", type="primary", width="stretch",
                      on_click=_select, args=(asset, query, requirements or {}))


def grid(cards: list[Any], render, cols: int = 3) -> None:
    for i in range(0, len(cards), cols):
        columns = st.columns(cols)
        for col, card in zip(columns, cards[i:i + cols]):
            with col:
                render(card)


def _set_query(text: str) -> None:
    st.session_state.query = text


def _search_all_types() -> None:
    st.session_state.type_filter = TYPE_ORDER
    st.session_state.rerun_search = True


# --------------------------------------------------------------------------- sidebar

engine = get_engine()
with db.session() as conn:
    counts = db.counts_by_type(conn)

with st.sidebar:
    st.title("🧩 Template Finder")
    st.caption("SQLite metadata · FAISS embeddings · Claude on Amazon Bedrock")
    st.text_input("Project ID", key="project_id", help="Selections are stored per project session.")

    st.subheader("Filters")
    type_filter = st.multiselect("Asset types", TYPE_ORDER, key="type_filter", format_func=ASSET_TYPE_LABELS.get)
    top_k = st.slider("Results to show", 6, 10, 8)
    use_llm = st.toggle("Understand query with Claude", value=True,
                        help="Stage 1: Claude extracts structured requirements. Off = rule-based parsing (no AWS call).")

    st.subheader("Library")
    for folder, (asset_type, level) in ASSET_FOLDERS.items():
        st.markdown(f"L{level} · {folder}: **{counts.get(asset_type, 0)}**")
    st.caption(f"{engine.store().size} vectors in FAISS · `{settings.embedding_model}`")
    _ep = endpoint_info()
    st.caption(f"LLM: `{_ep['model']}` · {_ep['client']} · {_ep['region']}")

tab_search, tab_browse, tab_ingest, tab_sel = st.tabs(["🔎 Search", "📚 Browse library", "⚙️ Ingest", "✅ Selections"])

# --------------------------------------------------------------------------- search

with tab_search:
    with st.form("search_form", border=False):
        query = st.text_area("Describe what you need", key="query", height=90, placeholder=EXAMPLES[0])
        submitted = st.form_submit_button("Search", type="primary")
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        col.button(ex, on_click=_set_query, args=(ex,), type="tertiary", key=f"ex-{ex[:20]}")

    if st.session_state.pop("rerun_search", False):
        submitted = True
    if submitted:
        if not query.strip():
            st.warning("Type what you are looking for first.")
        elif not type_filter:
            st.warning("Select at least one asset type in the sidebar.")
        else:
            with st.spinner("Stage 1: understanding your request · Stage 2: hybrid retrieval · Stage 3: ranking ..."):
                st.session_state.last_response = engine.search(
                    query.strip(), asset_types=type_filter, top_k=top_k, use_llm=use_llm)

    resp = st.session_state.get("last_response")
    if resp is not None:
        for note in resp.notes:
            st.warning(note)
        if resp.hidden_better:
            st.button("Include all asset types and search again", type="primary", on_click=_search_all_types)
        source = "Claude" if resp.used_llm else "rule-based"
        with st.expander(f"Stage 1 · Extracted requirements ({source})"):
            st.json(resp.requirements)
        if resp.results:
            scope = ("all asset types" if not resp.type_filter or set(resp.type_filter) == set(TYPE_ORDER)
                     else "only " + ", ".join(ASSET_TYPE_LABELS[t] for t in resp.type_filter))
            st.caption(
                f"Searched {scope}. Stage 2 retrieved {resp.n_candidates} candidates, Stage 3 reranked them; "
                f"showing the top {len(resp.results)} for “{resp.query}”.  "
                "Score = 0.35·R + 0.25·D + 0.20·F + 0.10·Q + 0.10·C (factors you didn't ask about are left out)"
            )
            grid(resp.results, lambda r: render_card(
                r.asset, r.screenshots, key=f"s-{r.asset['id']}", result=r,
                query=resp.query, requirements=resp.requirements))
        elif not resp.notes:
            st.info("No results.")

# --------------------------------------------------------------------------- browse

with tab_browse:
    c1, c2 = st.columns([3, 1])
    browse_type = c1.selectbox("Asset type", ["all", *TYPE_ORDER],
                               format_func=lambda t: "All types" if t == "all" else ASSET_TYPE_LABELS[t])
    with db.session() as conn:
        assets = db.get_assets(conn, asset_type=None if browse_type == "all" else browse_type)
        n_pages = max(1, -(-len(assets) // PAGE_SIZE))
        page_no = c2.number_input("Page", 1, n_pages, 1)
        page_assets = assets[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]
        shots = db.get_screenshots(conn, [a["id"] for a in page_assets])
    if not assets:
        st.info("The library is empty. Add item folders with screenshots and run ingestion (⚙️ Ingest tab).")
    else:
        st.caption(f"{len(assets)} asset(s)")
        grid(page_assets, lambda a: render_card(a, shots[a["id"]], key=f"b-{a['id']}"))

# --------------------------------------------------------------------------- ingest

with tab_ingest:
    st.markdown(
        f"Put each item in **its own subfolder** inside one of the four category folders of "
        f"`{settings.library_dir}`, then run the pipeline. Unchanged items are skipped; deleted folders are "
        "removed from SQLite and FAISS."
    )
    st.code(
        "library/\n"
        "  Web Templates/   aurora-saas/   01-home.png  02-dashboard.png  notes.txt  metadata.json\n"
        "  Page Templates/  neon-login/    login.png\n"
        "  UI Blocks/       glass-pricing/ pricing.png\n"
        "  UI Components/   gradient-button/ states.png",
        language="text",
    )
    items, scan_warnings = scan_library()
    with db.session() as conn:
        indexed = db.asset_index(conn)
    rows = []
    for it in items:
        row = indexed.get(it.rel_folder)
        state = "new" if row is None else ("indexed" if row["content_hash"] == it.content_hash else "changed")
        rows.append({"type": ASSET_TYPE_LABELS[it.asset_type], "folder": it.rel_folder,
                     "screenshots": len(it.images), "notes": bool(it.notes),
                     "metadata.json": bool(it.metadata), "status": state})
    for w in scan_warnings:
        st.warning(w)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No items found yet.")

    force = st.checkbox("Re-analyse everything with Claude (ignore unchanged check)")
    reembed = st.checkbox("Rebuild FAISS from stored text only (no Claude calls)")
    if st.button("Run ingestion pipeline", type="primary"):
        with st.status("Running ingestion pipeline ...", expanded=True) as status:
            report = run_ingestion(force=force, reembed=reembed, embedder=get_embedder(), progress=st.write)
            status.update(label=report.summary(), state="error" if report.failed else "complete")
        for folder, err in report.failed:
            st.error(f"{folder}: {err}")

# --------------------------------------------------------------------------- selections

with tab_sel:
    with db.session() as conn:
        selections = db.list_selections(conn)
    pid = st.session_state.project_id
    current = next((s for s in selections if s["project_id"] == pid), None)
    if current:
        st.markdown(f"**Current project session** (`{pid}`), ready to hand to the coding agent:")
        st.json({
            "project_id": current["project_id"],
            "selected_template_id": current["selected_asset_id"],
            "selected_template_name": current["asset_name"],
            "asset_type": current["asset_type"],
            "source_folder": f"{settings.library_dir.name}/{current['folder_path']}" if current["folder_path"] else None,
            "query": current["query"],
            "user_requirements": json.loads(current["user_requirements"] or "{}"),
            "status": current["status"],
        })
    else:
        st.info(f"No template selected yet for `{pid}`. Use **Select** on a result card.")
    if selections:
        st.markdown("**All project sessions**")
        st.dataframe(
            [{k: s[k] for k in ("project_id", "asset_name", "asset_type", "query", "status", "updated_at")}
             for s in selections],
            hide_index=True, width="stretch",
        )
