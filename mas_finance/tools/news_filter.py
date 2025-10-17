
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone

def basic_filter(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    return [it for it in items if it.get("title")]

def enhanced_filter(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    seen = set()
    for it in items:
        url = it.get("url"); title = it.get("title","").strip()
        if not url or url in seen: continue
        seen.add(url)
        if len(title) < 5: continue
        out.append(it)
    return out

def integrated_filter(items: List[Dict[str,Any]], *, from_dt: datetime, to_dt: datetime) -> List[Dict[str,Any]]:
    out=[]
    for it in items:
        ts = it.get("published_at")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except Exception:
                ts = None
        if not ts or ts.tzinfo is None:
            continue
        ts = ts.astimezone(timezone.utc)
        if from_dt <= ts <= to_dt:
            out.append(it)
    return out

def filter_news_3_stage(items: List[Dict[str,Any]], *, from_dt: datetime, to_dt: datetime) -> List[Dict[str,Any]]:
    return integrated_filter(enhanced_filter(basic_filter(items)), from_dt=from_dt, to_dt=to_dt)
