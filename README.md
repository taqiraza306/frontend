# Template Finder (demo)

Demo of the template retrieval system: you drop screenshots into folders, a pipeline has
**Claude Sonnet 5 on Amazon Bedrock** analyse them, stores the metadata in **SQLite**, embeds a rich
description with an **open-source embedding model**, and indexes it in **FAISS**. A **Streamlit** app takes
a natural-language request and shows the closest templates, pages, blocks and components.

| Production plan     | This demo                                                         |
|---------------------|-------------------------------------------------------------------|
| PostgreSQL          | SQLite (`storage/catalog.db`) + FTS5 for keyword search          |
| Qdrant / pgvector   | FAISS (`storage/faiss.index`), exact cosine search                |
| S3 / MinIO          | Local `library/` folder                                           |
| LLM                 | Claude Sonnet 5 via Bedrock (`anthropic.claude-sonnet-5`)         |
| Embeddings          | `BAAI/bge-small-en-v1.5` (sentence-transformers, runs locally)    |

## 1. Setup after cloning

You need **Python 3.11 or newer** (tested on 3.12; check with `python --version`) and git.
`.env`, `storage/` and `.venv/` are not in the repo, so every clone creates its own.

**Windows (Git Bash)**
```bash
git clone <repo-url> frontendagent && cd frontendagent
py -3.12 -m venv .venv                     # or: python -m venv .venv
source .venv/Scripts/activate              # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt            # a few minutes (PyTorch is large)
python -m playwright install chromium      # browser for capture.py, once per machine
cp .env.example .env                       # then put your Bedrock key in .env
```

**macOS / Linux**
```bash
git clone <repo-url> frontendagent && cd frontendagent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
```

Then build the search index and start the app:
```bash
python ingest.py --check     # one tiny Claude request: confirms .env works
python ingest.py             # analyses the screenshots already in library/ (about 15 s each)
streamlit run app.py         # http://localhost:8501
```

Notes for a fresh machine:
- `storage/` is not committed, so the index starts empty. `python ingest.py` rebuilds it from `library/`.
  It needs working Bedrock credentials, costs a few cents for the demo library, and stops with one clear
  message if the credentials are missing.
- Without credentials the app still opens, and search falls back to rule-based parsing, but there is
  nothing to find until ingestion has run once.
- The embedding model (~130 MB) downloads from Hugging Face on first use, so the first start needs internet.
- On Linux, `pip` installs the CUDA build of PyTorch (several GB). For a CPU-only install, run
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -r requirements.txt`.
- Share the Bedrock key privately, never through git: `.env` is in `.gitignore`.

AWS: in the Bedrock console, enable model access for **Claude Sonnet 5** in your region. In `.env`, set:

- `BEDROCK_MODEL_ID`, e.g. `anthropic.claude-sonnet-5`, or an inference profile such as `us.anthropic.claude-sonnet-5`.
- `BEDROCK_BASE_URL` (optional). The client is picked from it: `bedrock-runtime.<region>.amazonaws.com`
  uses the InvokeModel API (`AnthropicBedrock`), and `bedrock-mantle.<region>.api.aws/anthropic` (the default)
  uses the Messages API (`AnthropicBedrockMantle`). The region is read from the URL.
- Credentials: a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`), IAM keys, or `AWS_PROFILE`.

Then verify with `python ingest.py --check`, which sends one tiny request and prints the endpoint, model and
auth mode (never the key). To try Sonnet 5.5, change only the model ID (`...claude-sonnet-5-5`).

The embedding model (~130 MB) downloads from Hugging Face on first use.

## 2. Add your assets

Each item gets **its own subfolder** inside one of the four level folders:

```
library/
  Web Templates/        (level 1, complete project)
    aurora-saas/
      01-home.png       screenshots at the top level or in screenshots/
      02-dashboard.png
      03-pricing.png
      notes.txt         optional: free-text hints for Claude
      metadata.json     optional: authoritative overrides (see below)
      package.json      optional: tech stack is detected from it
  Page Templates/       (level 2, single page)
  UI Blocks/            (level 3, page section)
  UI Components/        (level 4, single element)
```

- Supported images: png, jpg, jpeg, webp, gif. Name the screenshot you want on the card `cover.png`;
  otherwise the first file in natural sort order is used.
- Tall full-page screenshots are automatically split into tiles so Claude can read them.
- `metadata.json` example (every field is optional; lists are merged, other values override Claude):

```json
{
  "license": "MIT",
  "repository_url": "https://github.com/you/aurora-saas",
  "preview_url": "https://aurora-saas.vercel.app",
  "framework": "Next.js",
  "styling": "Tailwind CSS",
  "ui_library": "shadcn/ui",
  "status": "verified",
  "features": ["stripe_billing"]
}
```

### Capture screenshots automatically

`capture.py` opens live demos in headless Chromium and saves full-page or element screenshots, plus a
`metadata.json` (repository, preview URL, license), straight into the right library folder.

```powershell
python -m playwright install chromium          # one-time browser download
copy capture_list.example.json capture_list.json
python capture.py capture_list.json            # edit the list first; re-runs skip existing items
```

Each list item has a `type` (web, page, block or component), a `name`, one or more `urls`, an optional
`selector` that captures only that element (`iframe >> nth=1` captures the second iframe preview), plus
`mobile`, `dark`, `click`, `hide`, `repository_url` and `license`. For a one-off capture, use flags instead:

```powershell
python capture.py --type block --name my-hero --url https://demo.site --selector "main section" --license MIT
```

The script scrolls each page so lazy images and scroll animations load, hides common cookie banners,
refuses HTTP error pages (404, a paused deployment), and keeps going when one page fails.

## 3. Run the pipeline

```powershell
python ingest.py --scan      # preview what will be indexed
python ingest.py             # analyse new/changed items, embed, index
python ingest.py --force     # re-analyse everything with Claude
python ingest.py --reembed   # rebuild FAISS from stored text (no Claude calls)
```

You can also run it from the app's **Ingest** tab. Runs are incremental: unchanged folders are
skipped (content hash), deleted folders are removed from SQLite and FAISS, and if FAISS ever drifts
from SQLite it is rebuilt from the stored `searchable_text`.

## 4. Start the app

```powershell
streamlit run app.py
```

Tabs: **Search** (query → ranked cards with match score, score breakdown, screenshots, Select button),
**Browse library**, **Ingest**, **Selections** (the stored project session JSON for the coding agent).

## How it works

**Ingestion** (`template_finder/pipeline.py`)
1. Scan folders → `ScannedItem` (screenshots, notes, metadata.json, package.json stack, hash).
2. Claude Sonnet 5 sees all screenshots and returns structured JSON (`schemas.AssetAnalysis`):
   name, descriptions, category, theme, design style, features, UI elements, palette, layout,
   per-screenshot page names, quality score.
3. Merge sources (Claude < package.json < metadata.json) and write to the `assets` and
   `asset_screenshots` tables.
4. Build `searchable_text` from the description, design, features, pages and stack (not just the name).
   Write one asset-level record plus one per page for multi-screenshot items to `asset_embeddings`.
5. Embed with sentence-transformers; FAISS vector id = `asset_embeddings.id`.

**Search** (`template_finder/search.py`)
1. **Requirement extraction**: Claude turns the prompt into `QueryRequirements` (types, theme,
   styles, features, UI elements, framework, keywords, rewritten semantic query). If Bedrock is
   unreachable or the toggle is off, a rule-based parser is used instead.
2. **Hybrid retrieval**: FAISS similarity (best of asset- and page-level vectors) + FTS5 BM25 keyword
   search + asset-type filter → top 20 candidates.
3. **Ranking**: `Score = 0.35·R + 0.25·D + 0.20·F + 0.10·Q + 0.10·C`
   (semantic relevance, design similarity, feature coverage, quality, compatibility).
   - R is absolute: raw cosine is mapped through a per-model range (`config.similarity_range`, override
     with `EMBEDDING_SIM_RANGE=low,high`). A poor result never looks good just because it is the best
     of a bad pool.
   - Factors you didn't ask about (no theme, style or features given) are left out and the weights are
     renormalised, instead of counting as perfect matches.
   - Each result is labelled Strong (≥70%), Partial (≥45%) or Weak. If the sidebar's asset-type filter
     hides a better match, the app says so and offers to search all types.
   - Weights live in `search.WEIGHTS`. `search_logs` and `project_sessions` record queries and selections
     so you can calibrate them later.

## SQLite tables

| Table               | Purpose                                                             |
|---------------------|---------------------------------------------------------------------|
| `assets`            | One row per item, all four levels (`asset_type`, `level` 1-4)       |
| `asset_screenshots` | Screenshots with page name, route and description                   |
| `asset_embeddings`  | Searchable text + model + dimension; `id` is the FAISS vector id    |
| `assets_fts`        | FTS5 index for keyword search                                       |
| `project_sessions`  | Selected template per project + extracted requirements              |
| `search_logs`       | Query, requirements and returned ids (for weight calibration)       |
| `meta`              | Embedding model in use (a change triggers an automatic re-embed)    |
