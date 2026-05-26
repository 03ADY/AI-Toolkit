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
