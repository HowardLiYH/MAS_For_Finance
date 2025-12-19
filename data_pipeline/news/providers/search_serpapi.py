"""Google News search via SerpAPI.
See comments inside for leakage-safe handling and strict post-filtering."""
# src/news/providers/search_serpapi.py
import os, re, requests, datetime as dt
from dateutil import parser as dtparser

SERP_ENDPOINT = "https://serpapi.com/search"

def _parse_serp_relative_date(s: str, now: dt.datetime) -> dt.datetime | None:
    if not s:
        return None
    t = s.strip().lower()
    if t == "yesterday":
        return now - dt.timedelta(days=1)
    m = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "minute":
            return now - dt.timedelta(minutes=n)
        if unit == "hour":
            return now - dt.timedelta(hours=n)
        if unit == "day":
            return now - dt.timedelta(days=n)
        if unit == "week":
            return now - dt.timedelta(weeks=n)
        if unit == "month":
            return now - dt.timedelta(days=30*n)
        if unit == "year":
            return now - dt.timedelta(days=365*n)
    # Absolute forms like "Sep 18, 2025"
    try:
        t = dtparser.parse(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None

def search_news_serpapi(query: str, from_dt: dt.datetime, to_dt: dt.datetime, limit: int = 50):
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY not set")

    # Google News with a *custom date range* matching [from_dt, to_dt]
    tbs = f"cdr:1,cd_min:{from_dt.strftime('%m/%d/%Y')},cd_max:{to_dt.strftime('%m/%d/%Y')}"
    params = {
        "engine": "google",
        "q": query,
        "tbm": "nws",                      # news vertical
        "tbs": tbs,                        # custom date range
        "num": min(100, max(10, limit)),   # 10..100
        "api_key": api_key,
        "hl": "en",
    }

    r = requests.get(SERP_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    rows = []
    now = dt.datetime.now(dt.timezone.utc)
    for it in js.get("news_results", []):
        url = it.get("link") or it.get("source")
        title = it.get("title", "")
        snippet = it.get("snippet") or " ".join(it.get("snippet_highlighted_words", []) or [])

        # SerpAPI gives 'date' (relative like "2 hours ago" or absolute like "Sep 18, 2025").
        dstr = it.get("date") or it.get("date_utc")
        ts = _parse_serp_relative_date(dstr, now)

        if not url or not ts:
            continue
        if from_dt <= ts <= to_dt:
            rows.append({"url": url, "title": title, "snippet": snippet, "published_at": ts})

        if len(rows) >= limit:
            break

    return rows
