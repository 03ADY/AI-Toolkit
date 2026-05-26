"""Unified API helper: built-in (Streamlit), offline mocks, or external FastAPI."""

import time
from typing import Any, Callable

import requests

from toolkit.builtin import invoke, should_use_builtin
from toolkit.config import API_BASE
from toolkit.fallback import mock_response

_ENDPOINT_FALLBACK = {
    "/sentiment/analyze": "sentiment",
    "/summarization/summarize": "summarization",
    "/generation/generate": "generation",
    "/translation/translate": "translation",
    "/qa/answer": "qa",
    "/image/caption": "caption",
    "caption": "caption",
}


def call_api(
    endpoint: str,
    payload: dict,
    *,
    offline: bool = False,
    timeout: int = 90,
    method: str = "POST",
    file_bytes: bytes | None = None,
) -> tuple[bool, Any, str | None, float]:
    fid = next((v for k, v in _ENDPOINT_FALLBACK.items() if k in endpoint), None)

    if fid and should_use_builtin():
        try:
            t0 = time.perf_counter()
            data = invoke(fid, payload, file_bytes=file_bytes)
            ms = (time.perf_counter() - t0) * 1000
            return True, data, None, ms
        except Exception as exc:
            return False, None, str(exc), 0.0

    if offline and fid:
        return True, mock_response(fid, payload), None, 0.0

    try:
        t0 = time.perf_counter()
        if method.upper() == "POST":
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        else:
            r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            data = r.content
        return True, data, None, ms
    except Exception as exc:
        return False, None, str(exc), 0.0


def log_if_present(log_fn: Callable | None, service: str, inp: str, out: str, ok: bool, ms: float):
    if log_fn:
        log_fn(service, inp, out, ok, latency_ms=ms)
