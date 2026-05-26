"""HTTP client for FastAPI backend."""

import time
from typing import Any

import requests

from toolkit.config import API_BASE, API_ROOT, SERVICES


def check_status(timeout: int = 5) -> tuple[bool, str | None, dict]:
    try:
        r = requests.get(f"{API_BASE}/status", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("models_loaded")), None, data
    except requests.exceptions.ConnectionError:
        return False, "Backend unreachable — enable Offline demo mode (Streamlit Cloud) or deploy the FastAPI API.", {}
    except requests.exceptions.Timeout:
        return False, "Backend timeout — models may still be loading.", {}
    except requests.exceptions.RequestException as e:
        return False, str(e), {}


def check_health(timeout: int = 3) -> dict:
    try:
        r = requests.get(f"{API_ROOT}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def get_metrics(timeout: int = 3) -> dict:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def post_with_timing(endpoint: str, payload: dict, timeout: int = 90) -> tuple[bool, Any, str | None, float]:
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return True, r.json(), None, ms
    except Exception as exc:
        return False, None, str(exc), 0.0


def list_models(timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{API_BASE}/models", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"models": [], "loaded": False}


def run_demo_call(service_id: str, timeout: int = 60) -> tuple[bool, Any, str | None]:
    from toolkit.builtin import invoke, should_use_builtin
    from toolkit.scenarios import DEMO_SAMPLES

    if should_use_builtin():
        try:
            payload = {}
            if service_id == "qa":
                payload = DEMO_SAMPLES["qa"]
            elif service_id in DEMO_SAMPLES and isinstance(DEMO_SAMPLES[service_id], str):
                payload = {"text": DEMO_SAMPLES[service_id]}
            return True, invoke(service_id, payload), None
        except Exception as exc:
            return False, None, str(exc)

    svc = next((s for s in SERVICES if s["id"] == service_id), None)
    if not svc:
        return False, None, "Unknown service"

    try:
        if service_id == "sentiment":
            payload = {"text": DEMO_SAMPLES["sentiment"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        elif service_id == "summarization":
            payload = {"text": DEMO_SAMPLES["summarization"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        elif service_id == "generation":
            payload = {"text": DEMO_SAMPLES["generation"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        elif service_id == "translation":
            payload = {"text": DEMO_SAMPLES["translation"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        elif service_id == "chatbot":
            payload = {"text": DEMO_SAMPLES["chatbot"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        elif service_id == "qa":
            q = DEMO_SAMPLES["qa"]
            payload = {"question": q["question"], "context": q["context"]}
            r = requests.post(f"{API_BASE}{svc['endpoint']}", json=payload, timeout=timeout)
        else:
            return False, None, f"Quick demo not wired for {service_id} in sidebar — use the service page."

        r.raise_for_status()
        return True, r.json(), None
    except Exception as exc:
        return False, None, str(exc)
