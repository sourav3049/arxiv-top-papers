# arxiv_top_papers_app.py
# Streamlit app: Filter & rank top arXiv papers in Reinforcement Learning (RL) and NLP
# Features
# - Query arXiv API (Atom) with keyword/category presets for RL & NLP
# - Optional OpenAlex lookup for citation counts
# - Scoring = recency + citations + keyword relevance (user-tunable weights)
# - Time window (last N months), max results, sort order
# - Nice paper cards with PDF/code links when available; export to CSV
# - Caching, error handling, rate limiting-friendly

import math
import time
import csv
import io
from datetime import datetime, timedelta, timezone

import requests
import feedparser
import streamlit as st

APP_TITLE = "arXiv Top Papers Finder — RL & NLP"
ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"

# -------------------------------
# Helpers
# -------------------------------

def _now_utc():
    return datetime.now(timezone.utc)

@st.cache_data(show_spinner=False)
def fetch_arxiv(query: str, categories: list[str], start: int, max_results: int, sort_by: str = "submittedDate"):
    # Build arXiv query string
    # We combine:
    #   (all:query) AND (cat:cat1 OR cat:cat2 ...)
    terms = []
    if query:
        # arXiv expects spaces as "+", colon separated field qualifiers
        # We'll use 'all:' to search everywhere
        q = query.replace(" ", "+")
        terms.append(f"all:{q}")
    if categories:
        cat_clause = "+OR+".join([f"cat:{c}" for c in categories])
        terms.append(f"({cat_clause})")

    search_query = "+AND+".join(terms) if terms else "all:machine+learning"

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,           # submittedDate | lastUpdatedDate | relevance
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    papers = []
    for e in feed.entries:
        # Prefer the PDF link
        pdf_link = None
        alt_links = []
        for l in e.links:
            if l.rel == 'alternate':
                alt_links.append(l.href)
            if l.type == 'application/pdf':
                pdf_link = l.href
        link = pdf_link or (alt_links[0] if alt_links else e.link)

        # Extract arXiv id
        arxiv_id = e.id.split('/abs/')[-1] if '/abs/' in e.id else e.id

        papers.append({
            "title": e.title.strip(),
            "authors": [a.name for a in getattr(e, 'authors', [])],
            "summary": getattr(e, 'summary', '').strip(),
            "published": getattr(e, 'published', ''),
            "updated": getattr(e, 'updated', ''),
            "primary_category": getattr(e, 'arxiv_primary_category', {}).get('term', ''),
            "categories": [t['term'] for t in getattr(e, 'tags', []) if 'term' in t],
            "link": link,
            "arxiv_id": arxiv_id,
        })
    return papers

@st.cache_data(show_spinner=False)
def openalex_citations(title: str, first_author: str | None = None):
    """Lookup citation count via OpenAlex; return dict or None.
    We use a conservative search and pick the top match.
    """
    if not title:
        return None
    q = title
    params = {
        "search": q,
        "per_page": 1,
        "mailto": "example@example.com",
    }
    try:
        r = requests.get(OPENALEX_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get('results'):
            w = data['results'][0]
            return {
                "openalex_id": w.get('id'),
                "cited_by_count": w.get('cited_by_count', 0),
                "publication_year": w.get('publication_year'),
                "host_venue": (w.get('host_venue') or {}).get('display_name'),
                "doi": w.get('doi'),
                "oa_url": (w.get('open_access') or {}).get('oa_url'),
            }
    except Exception:
        return None
    return None


def parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def age_days(dt: datetime | None) -> float:
    if not dt:
        return 3650
    return max(0.0, (_now_utc() - dt).total_seconds() / 86400.0)


def relevance_score(text: str, keywords: list[str]) -> float:
    if not text or not keywords:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw in keywords:
        k = kw.lower().strip()
        if not k:
            continue
        # count occurrences in title + abstract (simple heuristic)
        score += t.count(k) * 1.0
    return score


def compute_score(p, w_recency: float, w_citations: float, w_relevance: float, half_life_days: float, keywords: list[str]):
    # Recency via exponential decay
    pub_dt = parse_date(p.get('updated') or p.get('published'))
    a = age_days(pub_dt)
    rec = math.exp(-a / max(1e-6, half_life_days))

    # Citations
    c = (p.get('citations') or 0)
    cit = math.log1p(max(0, c))

    # Relevance from title + abstract
    rel = relevance_score((p.get('title') or '') + ' ' + (p.get('summary') or ''), keywords)

    return w_recency * rec + w_citations * cit + w_relevance * rel


def to_csv(rows: list[dict]):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return buf.getvalue()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title(APP_TITLE)
st.caption("Search, rank, and export the freshest & most-cited arXiv papers in RL and NLP. ✨")

with st.sidebar:
    st.header("Search settings")

    preset = st.selectbox(
        "Preset",
        ["Reinforcement Learning", "NLP (Natural Language Processing)", "Custom"],
    )

    preset_categories = {
        "Reinforcement Learning": ["cs.LG", "cs.AI", "stat.ML"],
        "NLP (Natural Language Processing)": ["cs.CL", "cs.AI", "cs.LG"],
    }

    categories = st.multiselect(
        "arXiv categories",
        options=["cs.AI", "cs.CL", "cs.LG", "stat.ML", "cs.CV", "cs.IR", "cs.RO"],
        default=preset_categories.get(preset, ["cs.AI", "cs.LG"]) if preset != "Custom" else ["cs.AI", "cs.LG"],
        help="Use arXiv category codes. RL: cs.LG/stat.ML/cs.AI; NLP: cs.CL.",
    )

    default_query = "reinforcement learning" if preset == "Reinforcement Learning" else ("natural language processing" if preset.startswith("NLP") else "")
    query = st.text_input("Keywords (space-separated)", value=default_query)

    months = st.slider("Lookback window (months)", 1, 24, 6)

    max_fetch = st.slider("Fetch count from arXiv (per page)", 10, 200, 100, step=10, help="App will over-fetch and filter by date.")
    pages = st.slider("Pages to fetch", 1, 5, 2, help="Each page fetches 'max' results. Increase for broader coverage.")

    st.divider()
    st.subheader("Ranking weights")
    w_recency = st.slider("Recency weight", 0.0, 3.0, 1.0, 0.1)
    w_citations = st.slider("Citations weight", 0.0, 3.0, 1.0, 0.1)
    w_relevance = st.slider("Keyword relevance weight", 0.0, 3.0, 0.5, 0.1)
    half_life_days = st.slider("Recency half-life (days)", 7, 365, 90)

    st.divider()
    use_openalex = st.toggle("Enrich with OpenAlex (citations)", value=True, help="Adds 1 API call per paper (rate limits may apply).")
    pause = st.slider("Pause between OpenAlex calls (seconds)", 0.0, 2.0, 0.2, 0.1)

    st.divider()
    top_k = st.slider("Show top K", 5, 100, 20)

# Action button
run_search = st.button("🔎 Search & Rank", type="primary")

if run_search:
    lookback_cutoff = _now_utc() - timedelta(days=months * 30)

    with st.spinner("Querying arXiv…"):
        all_papers = []
        for p in range(pages):
            batch = fetch_arxiv(query=query, categories=categories, start=p * max_fetch, max_results=max_fetch)
            all_papers.extend(batch)
            time.sleep(0.5)  # be kind to arXiv

    # Filter by date window
    filtered = []
    for p in all_papers:
        dt = parse_date(p.get('updated') or p.get('published'))
        if not dt or dt < lookback_cutoff:
            continue
        filtered.append(p)

    st.success(f"Fetched {len(all_papers)}; kept {len(filtered)} within the last {months} months.")

    # Optional: enrich with OpenAlex citations
    if use_openalex:
        st.caption("Enriching with OpenAlex for citations…")
        prog = st.progress(0.0, text="Looking up citations…")
        enriched = []
        for i, p in enumerate(filtered):
            info = openalex_citations(p.get('title', ''), (p.get('authors') or [None])[0])
            if info:
                p.update({
                    "citations": info.get('cited_by_count', 0),
                    "openalex_id": info.get('openalex_id'),
                    "doi": info.get('doi'),
                    "oa_url": info.get('oa_url'),
                    "venue": info.get('host_venue'),
                })
            else:
                p.update({"citations": 0})
            enriched.append(p)
            prog.progress((i + 1) / max(1, len(filtered)))
            time.sleep(pause)
        filtered = enriched

    # Scoring & sorting
    keywords = [k.strip() for k in query.split()] if query else []
    for p in filtered:
        p["score"] = compute_score(p, w_recency, w_citations, w_relevance, half_life_days, keywords)
    filtered.sort(key=lambda x: x["score"], reverse=True)

    top = filtered[:top_k]

    # Summary row
    st.subheader("Results")
    st.write(f"Showing **{len(top)}** of **{len(filtered)}** ranked papers.")

    # Export CSV
    if top:
        csv_bytes = to_csv([
            {
                "title": p.get('title'),
                "authors": "; ".join(p.get('authors') or []),
                "published": p.get('published'),
                "updated": p.get('updated'),
                "primary_category": p.get('primary_category'),
                "categories": ", ".join(p.get('categories') or []),
                "citations": p.get('citations', 0),
                "score": round(p.get('score', 0), 4),
                "pdf_or_link": p.get('link'),
                "arxiv_id": p.get('arxiv_id'),
                "doi": p.get('doi'),
                "venue": p.get('venue'),
                "oa_url": p.get('oa_url'),
            }
            for p in top
        ])
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="arxiv_top_papers.csv", mime="text/csv")

    # Pretty cards
    for p in top:
        with st.container(border=True):
            left, right = st.columns([0.75, 0.25])
            with left:
                st.markdown(f"### {p.get('title')}")
                st.caption(
                    f"{'; '.join(p.get('authors') or [])} · {p.get('primary_category')} · Updated {p.get('updated')[:10]}"
                )
                st.write(p.get('summary'))
                chips = [c for c in (p.get('categories') or [])][:8]
                if chips:
                    st.write("**Categories:** ", ", ".join(chips))
            with right:
                st.metric("Score", f"{p.get('score'):.2f}")
                st.metric("Citations", f"{p.get('citations', 0)}")
                st.markdown(f"[arXiv link]({p.get('link')})")
                if p.get('oa_url'):
                    st.markdown(f"[Open Access]({p.get('oa_url')})")
                if p.get('doi'):
                    st.markdown(f"DOI: {p.get('doi')}")
                st.code(p.get('arxiv_id') or "")

else:
    st.info("Configure your query in the sidebar, then click **Search & Rank**.")

st.divider()
st.caption("Tip: Tweak weights to balance fresh vs. influential papers. OpenAlex enrichment is optional if you want faster searches.")
