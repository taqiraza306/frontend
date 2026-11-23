"""Pydantic models used as structured-output schemas for Claude."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

AssetType = Literal["website_template", "page_template", "ui_block", "ui_component"]


class ScreenshotInfo(BaseModel):
    file_name: str = Field(description="Exact file name of the screenshot, as given in the prompt.")
    page_name: str = Field(description="What this screenshot shows, e.g. 'Dashboard', 'Login', 'Hero section', 'Primary button states'.")
    route: str = Field(description="Likely URL route such as '/dashboard'; empty string for blocks/components.")
    description: str = Field(description="2-3 sentences describing layout, content and notable UI on this screenshot.")


class AssetAnalysis(BaseModel):
    name: str = Field(description="Short, catchy catalog name (2-4 words). Reuse the folder name if it is already meaningful.")
    short_description: str = Field(description="One sentence (max ~25 words) for a gallery card.")
    description: str = Field(description="Detailed description (80-150 words): purpose, layout, sections, visual style, notable interactions.")
    category: str = Field(description="Domain / industry in snake_case, e.g. saas, ai_tool, ecommerce, portfolio, agency, finance, education, healthcare, real_estate, blog, crm, admin, generic.")
    subcategory: str = Field(description="Kind of asset in snake_case. Pages: landing, dashboard, login, pricing, settings ... Blocks: hero, pricing, testimonials, footer, navbar, features, cta, faq ... Components: button, input, dropdown, modal, sidebar, card, table, tabs, toast ... Web templates: multi_page_site, web_app, admin_panel ...")
    theme: Literal["dark", "light", "both"]
    design_style: List[str] = Field(description="3-6 lowercase style descriptors, e.g. minimal, modern, premium, playful, corporate, elegant, bold, futuristic, glassmorphism, neumorphism, brutalist, retro, gradient, colorful, clean, aesthetic.")
    features: List[str] = Field(description="Functional features in snake_case, e.g. authentication, analytics_charts, collapsible_sidebar, data_table, pricing_tiers, dark_mode_toggle, search, notifications, file_upload, checkout.")
    ui_elements: List[str] = Field(description="Visible UI elements in snake_case, e.g. navbar, sidebar, cards, line_chart, modal, tabs, avatar, badge, form, footer.")
    color_palette: List[str] = Field(description="3-6 dominant colors as hex codes like #0F172A.")
    typography: str = Field(description="Short description of the typography, e.g. 'geometric sans-serif headings, generous line height'.")
    layout: str = Field(description="Short description of the layout structure, e.g. 'fixed left sidebar + top bar + 12-col card grid'.")
    responsive_hint: bool = Field(description="True if the design looks responsive / mobile-ready (mobile screenshots, fluid grid).")
    likely_framework: str = Field(description="Best guess of the frontend framework (Next.js, React, Vue ...) or 'unknown'. Screenshots rarely prove this - prefer 'unknown' over guessing.")
    likely_styling: str = Field(description="Best guess of the styling approach (Tailwind CSS, CSS ...) or 'unknown'.")
    likely_ui_library: str = Field(description="Best guess of the UI library (shadcn/ui, MUI ...) or 'unknown'.")
    use_cases: List[str] = Field(description="2-5 short phrases describing who/what this is good for.")
    keywords: List[str] = Field(description="8-15 lowercase search keywords a user might type to find this asset.")
    quality_score: float = Field(description="Visual polish / production readiness from 0.0 (rough) to 1.0 (premium, consistent, pixel-perfect).")
    screenshots: List[ScreenshotInfo] = Field(description="One entry per distinct screenshot file, in the order given.")


class QueryRequirements(BaseModel):
    asset_types: List[AssetType] = Field(description="Asset levels the user is asking for. Empty list if unclear. website_template = complete multi-page project, page_template = a single page, ui_block = a page section, ui_component = a single element.")
    project_type: str = Field(description="Domain / kind in snake_case, e.g. saas_dashboard, ecommerce_store, portfolio, landing_page; empty string if not stated.")
    theme: Literal["dark", "light", "both", "any"]
    design_style: List[str] = Field(description="Lowercase style descriptors the user wants, e.g. minimal, premium, modern.")
    features: List[str] = Field(description="Requested functional features in snake_case, e.g. analytics_charts, authentication.")
    ui_elements: List[str] = Field(description="Requested UI elements in snake_case, e.g. collapsible_sidebar, data_table, navbar.")
    framework: List[str] = Field(description="Requested frameworks in lowercase (nextjs, react, vue ...); empty if not stated.")
    keywords: List[str] = Field(description="5-12 important lowercase keywords and close synonyms for keyword search.")
    semantic_query: str = Field(description="A rich 1-3 sentence description of the ideal asset, written like a catalog entry, for vector similarity search.")
