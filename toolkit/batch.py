"""Batch processing helpers."""

import pandas as pd
import requests

from toolkit.builtin import invoke, should_use_builtin
from toolkit.config import API_BASE
from toolkit.fallback import mock_response


def batch_sentiment(texts: list[str], *, offline: bool = False) -> pd.DataFrame:
    rows = []
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        if should_use_builtin():
            t0 = __import__("time").perf_counter()
            r = invoke("sentiment", {"text": text})
            ms = (__import__("time").perf_counter() - t0) * 1000
            rows.append({
                "row": i + 1,
                "text": text[:80],
                "label": r["label"],
                "score": r["score"],
                "latency_ms": round(ms, 1),
            })
            continue
        if offline:
            r = mock_response("sentiment")
            rows.append({"row": i + 1, "text": text[:80], "label": r["label"], "score": r["score"], "latency_ms": 0})
            continue
        try:
            import time
            t0 = time.perf_counter()
            resp = requests.post(f"{API_BASE}/sentiment/analyze", json={"text": text}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "row": i + 1,
                "text": text[:80],
                "label": data.get("label"),
                "score": data.get("score"),
                "latency_ms": round(ms, 1),
            })
        except Exception as exc:
            rows.append({"row": i + 1, "text": text[:80], "label": "ERROR", "score": 0, "latency_ms": str(exc)})
    return pd.DataFrame(rows)
