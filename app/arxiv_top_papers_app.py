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
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
HTTP_HEADERS = {
    "User-Agent": "arxiv-top-papers/1.0 (+sourav3049@gmail.com)"
}

CONF_ALIASES = {
    "NeurIPS": ["neurips", "nips"],
    "ICLR": ["iclr"],
    "ICML": ["icml"],
    "CVPR": ["cvpr"],
    "ICCV": ["iccv"],
    "ECCV": ["eccv"],
    "ACL": ["acl"],
    "EMNLP": ["emnlp"],
    "NAACL": ["naacl"],
    "COLING": ["coling"],
    "AAAI": ["aaai"],
    "IJCAI": ["ijcai"],
    "KDD": ["kdd"],
    "SIGIR": ["sigir"],
}

# -------------------------------
# Helpers
# -------------------------------
def matches_conference(paper: dict, selected: list[str]) -> bool:
    """Return True if paper matches any selected conference by:
       - OpenAlex host_venue (if present)
       - arXiv comments
       - title/summary
    """
    if not selected:
        return True  # no filtering

    hay = " ".join([
        (paper.get("venue") or ""),                # from OpenAlex enrichment
        (paper.get("arxiv_comment") or ""),
        (paper.get("title") or ""),
        (paper.get("summary") or ""),
    ]).casefold()

    for conf in selected:
        for alias in CONF_ALIASES.get(conf, []):
            if alias in hay:
                return True
    return False

def _now_utc():
    return datetime.now(timezone.utc)

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def fetch_arxiv(query: str, categories: list[str], start: int, max_results: int, sort_by: str = "submittedDate"):
    """
    Returns a list of paper dicts from arXiv. Uses HTTPS + User-Agent.
    Also returns a debug dict for UI diagnostics.
    """
    # Build arXiv query string
    terms = []
    if query:
        # Let requests encode spaces; arXiv accepts '+' or encoded spaces
        terms.append(f"all:{query}")
    if categories:
        # (cat:cs.CL OR cat:cs.AI ...)
        cat_clause = " OR ".join([f"cat:{c}" for c in categories])
        terms.append(f"({cat_clause})")

    # If nothing specified, default to broad ML
    search_query = " AND ".join(terms) if terms else "all:machine learning"

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,            # submittedDate | lastUpdatedDate | relevance
        "sortOrder": "descending",
    }

    debug = {"url": ARXIV_API, "params": params, "raw_count": 0}

    try:
        resp = requests.get(ARXIV_API, params=params, timeout=30, headers=HTTP_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Surface the error in UI
        st.error(f"arXiv request failed: {e}")
        return [], debug

    feed = feedparser.parse(resp.text)
    if hasattr(feed, "bozo_exception") and feed.bozo:
        st.warning(f"Feed parse warning: {feed.bozo_exception}")
    entries = feed.entries or []
    debug["raw_count"] = len(entries)

    papers = []
    for e in entries:
        pdf_link = None
        alt_links = []
        for l in getattr(e, "links", []):
            if getattr(l, "rel", "") == "alternate":
                alt_links.append(l.href)
            if getattr(l, "type", "") == "application/pdf":
                pdf_link = l.href
        link = pdf_link or (alt_links[0] if alt_links else getattr(e, "link", None))

        # Extract arXiv id
        eid = getattr(e, "id", "")
        arxiv_id = eid.split("/abs/")[-1] if "/abs/" in eid else eid

        papers.append({
            "title": getattr(e, "title", "").strip(),
            "authors": [a.name for a in getattr(e, 'authors', [])] if hasattr(e, "authors") else [],
            "summary": getattr(e, 'summary', '').strip(),
            "published": getattr(e, 'published', ''),
            "updated": getattr(e, 'updated', ''),
            "primary_category": getattr(getattr(e, 'arxiv_primary_category', {}), 'get', lambda k, d=None: d)('term', '') if hasattr(e, 'arxiv_primary_category') else '',
            "categories": [t['term'] for t in getattr(e, 'tags', []) if isinstance(t, dict) and 'term' in t],
            "link": link,
            "arxiv_id": arxiv_id,
            "arxiv_comment": getattr(e, "arxiv_comment", ""),
        })
    return papers, debug


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
    CONF_OPTIONS = [
        "NeurIPS", "ICLR", "ICML", "CVPR", "ICCV", "ECCV",
        "ACL", "EMNLP", "NAACL", "COLING",
        "AAAI", "IJCAI", "KDD", "SIGIR"
    ]
    conf_filter = st.multiselect("Conference filter (optional)", options=CONF_OPTIONS, default=[])
    match_year = st.toggle("Only show if a match is found (strict)", value=False,
                        help="If ON, papers must match at least one selected conference via comments/venue/title.")

    st.divider()
    top_k = st.slider("Show top K", 5, 100, 20)

# Action button
run_search = st.button("🔎 Search & Rank", type="primary")

if run_search:
    lookback_cutoff = _now_utc() - timedelta(days=months * 30)

    with st.spinner("Querying arXiv…"):
        all_papers = []
        debug_info = []
        for p in range(pages):
            batch, dbg = fetch_arxiv(query=query, categories=categories, start=p * max_fetch, max_results=max_fetch)
            all_papers.extend(batch)
            debug_info.append(dbg)
            time.sleep(0.5)  # be kind to arXiv

    # Diagnostics
    #with st.expander("Debug: arXiv API calls"):
    #    for i, d in enumerate(debug_info):
    #        st.write(f"Page {i+1}: params={d['params']}")
    #        st.write(f"Raw entries: {d['raw_count']}")

    # Filter by date window (be lenient if dates are missing)
    filtered = []
    dropped_old = 0
    for p in all_papers:
        dt = parse_date(p.get('updated') or p.get('published'))
        if dt is None:
            # keep items with missing dates instead of dropping silently
            filtered.append(p)
            continue
        if dt >= lookback_cutoff:
            filtered.append(p)
        else:
            dropped_old += 1

    st.success(f"Fetched {len(all_papers)}; kept {len(filtered)} within the last {months} months (dropped {dropped_old} as too old).")

    if not filtered:
        st.warning("No results after filtering. Try: reduce lookback months, remove some categories, or broaden keywords.")
        st.stop()

    # 📌 Conference filter
    if conf_filter:
        before = len(filtered)
        filtered = [p for p in filtered if matches_conference(p, conf_filter)]
        st.info(f"Conference filter kept {len(filtered)} of {before} papers.")

    if match_year and not filtered:
        st.warning("No papers matched the selected conferences. Try disabling Strict mode, broadening keywords, or increasing the lookback window.")
        st.stop()

    # Scoring & sorting
    keywords = [k.strip() for k in query.split()] if query else []
    for p in filtered:
        p["score"] = compute_score(p, w_recency, w_citations, w_relevance, half_life_days, keywords)
    filtered.sort(key=lambda x: x["score"], reverse=True)

    top = filtered[:top_k]

    st.subheader("Results")
    st.write(f"Showing **{len(top)}** of **{len(filtered)}** ranked papers.")

    # Export button
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
