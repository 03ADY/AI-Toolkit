"""Detect Streamlit Community Cloud."""

import os


def is_streamlit_cloud() -> bool:
    if os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT") == "cloud":
        return True
    blob = " ".join(
        os.environ.get(k, "")
        for k in ("HOSTNAME", "STREAMLIT_SERVER_ADDRESS", "STREAMLIT_SHARING_MODE")
    ).lower()
    if "streamlit.app" in blob:
        return True
    return os.path.isdir("/mount/src")


def use_builtin_runtime() -> bool:
    """Streamlit Cloud runs in-app AI — no separate FastAPI process."""
    if is_streamlit_cloud():
        return True
    return os.environ.get("AI_TOOLKIT_BUILTIN", "").lower() in ("1", "true", "yes")
