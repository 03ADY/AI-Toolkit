"""Enterprise UI: sidebar demo controls, home, dashboard."""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from toolkit.analytics import history_to_usage, success_rate, timeline
from toolkit.api_client import check_health, check_status, get_metrics, list_models, run_demo_call
from toolkit.builtin import should_use_builtin
from toolkit.cloud import is_streamlit_cloud, use_builtin_runtime
from toolkit.config import API_BASE, API_ROOT, APP_NAME, SERVICES
from toolkit.cost import estimate_session_cost
from toolkit.scenarios import DEMO_SAMPLES, MENU_TO_SAMPLE, get_sample
from toolkit.theme_ui import inject_theme


def _service_health_grid(models: dict, loaded: bool):
    names = [s["name"] for s in SERVICES]
    registered = set(models.get("models", [])) if loaded else set()
    cols = st.columns(3)
    for i, s in enumerate(SERVICES):
        # Map friendly names to pipeline keys loosely
        online = loaded and (not registered or any(k in str(registered).lower() for k in [s["id"], s["name"][:4].lower()]))
        if not loaded:
            online = False
        color = "#22c55e" if online else "#94a3b8"
        with cols[i % 3]:
            st.markdown(
                f'<div style="padding:0.5rem;border-left:4px solid {color};margin:0.2rem 0;">'
                f'{s["icon"]} <b>{s["name"]}</b><br><span style="font-size:0.75rem;color:{color};">'
                f'{"Online" if online else "Standby"}</span></div>',
                unsafe_allow_html=True,
            )


def render_demo_sidebar() -> dict:
    inject_theme()
    st.markdown("### 🎬 Demo")
    present = st.toggle("Present mode", value=st.session_state.get("present_mode", True))
    st.session_state.present_mode = present

    api_base = API_BASE
    loaded = False
    offline = bool(st.session_state.get("offline_demo", False))

    if use_builtin_runtime():
        st.session_state.builtin_mode = True
        st.session_state.offline_demo = True
        st.success("✅ Built-in AI — runs in this app")
        st.caption("No API server or URL needed on Streamlit Cloud.")
        loaded = True
        offline = True
    else:
        offline = st.toggle(
            "Offline demo mode (mocks)",
            value=offline,
            help="Use static mocks without starting the FastAPI backend.",
        )
        st.session_state.offline_demo = offline
        st.session_state.builtin_mode = False

        with st.expander("Advanced: external FastAPI API"):
            api_base = st.text_input(
                "API base",
                API_BASE,
                help="Only if you run merged_backend.py separately (local Docker, Render, etc.)",
            )
            if api_base != API_BASE:
                os.environ["AI_TOOLKIT_API_BASE"] = api_base
                st.caption("Restart app if URL changed mid-session.")

        if offline:
            st.info("Offline mocks — not real inference")
            loaded = True
        else:
            loaded, err, _ = check_status()
            if loaded:
                st.success("Backend ready")
            else:
                st.warning(err or "Start API with scripts/start-demo.ps1 or enable offline mocks.")

    est = estimate_session_cost(st.session_state.get("history", []))
    st.caption(f"Session est. cost: ${est['total_usd']:.4f}")

    with st.expander("⚡ Quick API test"):
        sid = st.selectbox("Service", [s["id"] for s in SERVICES], format_func=lambda x: next(s["name"] for s in SERVICES if s["id"] == x))
        if st.button("Run demo call", use_container_width=True):
            if should_use_builtin():
                from toolkit.builtin import invoke
                from toolkit.scenarios import DEMO_SAMPLES

                if sid == "qa":
                    data = invoke(sid, DEMO_SAMPLES["qa"])
                else:
                    data = invoke(sid, {"text": DEMO_SAMPLES.get(sid, "Hello")})
                st.json(data)
                st.session_state.last_demo_result = {"service": sid, "data": data}
            elif offline:
                from toolkit.fallback import mock_response
                data = mock_response(sid)
                st.json(data)
                st.session_state.last_demo_result = {"service": sid, "data": data}
            else:
                ok, data, e = run_demo_call(sid)
                if ok:
                    st.json(data)
                    st.session_state.last_demo_result = {"service": sid, "data": data}
                else:
                    st.error(e)

    sample_menu = st.session_state.get("main_menu_selector", "Sentiment Analysis")
    sk = MENU_TO_SAMPLE.get(sample_menu)
    if sk and st.button("📋 Load sample for current page", use_container_width=True):
        st.session_state[f"demo_fill_{sk}"] = get_sample(sk)
        st.rerun()

    return {"present": present, "api": api_base, "loaded": loaded, "offline": offline}


def render_enterprise_home():
    builtin = should_use_builtin()
    offline = st.session_state.get("offline_demo", False) or builtin
    loaded, err, _ = check_status() if not offline else (True, None, {})
    health = check_health() if not offline and not builtin else {"status": "builtin"}
    models = list_models() if not offline and not builtin else {"models": list(MOCK_PIPELINE_NAMES()), "loaded": True}

    st.header("Welcome to AI Toolkit Enterprise")
    if builtin:
        st.success(
            "**Built-in engine** — sentiment, summary, translation, Q&A, and more run **inside this Streamlit app**. "
            "Open **Demo Runner** or any service from the sidebar and click the action button."
        )
    elif not loaded:
        st.warning(
            "Backend at `127.0.0.1:8000` is not reachable. Enable **Offline demo mode** in the sidebar, "
            "or deploy `merged_backend.py` and set the API base URL."
        )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("API status", "Online" if loaded or offline else "Offline")
    c2.metric("Models", len(models.get("models", [])))
    metrics = get_metrics() if not offline else {}
    c3.metric("Requests served", metrics.get("requests_served", "—"))
    c4.metric("Session calls", st.session_state.get("api_calls_count", 0))
    c5.metric("Success rate", f"{success_rate(st.session_state.get('history', [])):.0f}%")

    if not loaded and not offline and not builtin:
        st.error(err or "Backend unreachable — enable **Offline demo mode** or run `scripts/start-demo.ps1`.")

    st.subheader("Service health")
    _service_health_grid(models, loaded or offline or builtin)

    st.subheader("Quick launch")
    links = [
        ("Demo Runner", "🎬"),
        ("Document Analyzer", "📑"),
        ("Batch Lab", "📦"),
        ("API Playground", "🧪"),
        ("Integrations Hub", "🔗"),
        ("Cost Center", "💰"),
    ]
    cols = st.columns(3)
    for i, (name, icon) in enumerate(links):
        with cols[i % 3]:
            st.info(f"{icon} **{name}** — select in sidebar")

    with st.expander("✅ Presenter checklist (2 min)", expanded=st.session_state.get("present_mode", True)):
        st.markdown("""
1. **Backend ready** or **Offline demo mode**  
2. **Demo Runner** → full suite + latency chart  
3. **Document Analyzer** → multi-service pipeline  
4. **Service Compare** → sentiment vs translation  
5. **Session Report** → HTML export  
6. **Integrations Hub** → webhooks + governance  
        """)

    if st.session_state.get("last_demo_result"):
        st.subheader("Last quick demo")
        st.json(st.session_state.last_demo_result)


def MOCK_PIPELINE_NAMES():
    return ["sentiment", "summarizer", "translator", "captioner", "generator", "tts", "stt", "qa_model"]


def render_enterprise_dashboard():
    st.header("🚀 System Dashboard")
    builtin = should_use_builtin()
    offline = st.session_state.get("offline_demo", False)
    loaded, err, _ = check_status() if not offline and not builtin else (True, None, {})
    health = check_health() if not offline and not builtin else {"status": "builtin"}
    models = list_models() if not offline and not builtin else {"models": MOCK_PIPELINE_NAMES()}

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Health", health.get("status", "—"))
    h2.metric("Models loaded", "Yes" if (loaded or offline or builtin) else "No")
    h3.metric("API calls (session)", st.session_state.get("api_calls_count", 0))
    est = estimate_session_cost(st.session_state.get("history", []))
    h4.metric("Est. cost", f"${est['total_usd']:.4f}")

    if not loaded and not offline and not builtin:
        st.error(err or "Backend not ready")
        st.info(f"API `{API_ROOT}` · docs `/docs`")
    else:
        label = "built-in engine" if builtin else ("offline mocks" if offline else "FastAPI backend")
        st.success(f"All services operational ({label})")

    st.subheader("Pipeline registry")
    if models.get("models"):
        df = pd.DataFrame({"pipeline": models["models"], "status": "ready"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    hist = st.session_state.get("history", [])
    usage = history_to_usage(hist)
    if not usage.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(usage, x="service", y="count"), use_container_width=True)
        with c2:
            st.metric("Success rate", f"{success_rate(hist):.1f}%")
        lat_df = pd.DataFrame([h for h in hist if h.get("latency_ms")])
        if not lat_df.empty:
            st.plotly_chart(px.box(lat_df, x="service", y="latency_ms"), use_container_width=True)

    metrics = get_metrics() if not offline else {}
    if metrics:
        st.json(metrics)
