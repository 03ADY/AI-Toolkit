"""Extra UI: playground, batch, demo runner, session report."""

import json
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from toolkit.analytics import history_to_usage, success_rate, timeline
from toolkit.api_client import check_status, get_metrics, post_with_timing, run_demo_call
from toolkit.batch import batch_sentiment
from toolkit.config import API_BASE, API_ROOT, APP_NAME, SERVICES
from toolkit.builtin import invoke, should_use_builtin
from toolkit.fallback import mock_response
from toolkit.reports import session_report_html, session_report_markdown
from toolkit.scenarios import DEMO_SAMPLES


DEMO_SUITE = ["sentiment", "summarization", "translation", "qa"]


def render_api_playground():
    st.header("🧪 API Playground")
    st.caption("Send raw JSON to any toolkit endpoint.")

    offline = st.session_state.get("offline_demo", False)
    builtin = should_use_builtin()
    svc = st.selectbox("Service", SERVICES, format_func=lambda s: f"{s['icon']} {s['name']}")
    method = st.radio("Method", ["POST"], horizontal=True)
    if svc["id"] == "qa":
        default_payload = DEMO_SAMPLES["qa"]
    elif svc["id"] in DEMO_SAMPLES and isinstance(DEMO_SAMPLES[svc["id"]], str):
        default_payload = {"text": DEMO_SAMPLES[svc["id"]]}
    else:
        default_payload = {"text": "Hello"}

    body = st.text_area("JSON body", value=json.dumps(default_payload, indent=2), height=200)
    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Send request", type="primary", use_container_width=True)
    with col2:
        st.code(f"{method} {API_BASE}{svc['endpoint']}", language="text")

    if run:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        if builtin:
            data = invoke(svc["id"], payload)
            st.success("Built-in engine — ran in this app")
            st.json(data)
            return
        if offline:
            data = mock_response(svc["id"], payload)
            st.success("Offline demo mode — mock response")
            st.json(data)
            return
        ok, data, err, ms = post_with_timing(svc["endpoint"], payload)
        if ok:
            st.metric("Latency", f"{ms:.0f} ms")
            st.json(data)
        else:
            st.error(err)


def render_batch_lab():
    st.header("📦 Batch Lab")
    st.write("Analyze sentiment for many lines at once.")

    offline = st.session_state.get("offline_demo", False)
    sample = "Great product\nTerrible support\nAverage experience\nOutstanding quality"
    raw = st.text_area("One text per line", value=sample, height=160)
    if st.button("Run batch sentiment", type="primary"):
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        with st.spinner(f"Processing {len(lines)} rows…"):
            df = batch_sentiment(lines, offline=offline)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty and "label" in df.columns:
            st.plotly_chart(px.bar(df["label"].value_counts().reset_index(), x="label", y="count"), use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False).encode(), "batch_sentiment.csv")


def _display_demo_suite_results(results: list[dict], *, offline: bool, builtin: bool):
    """Show suite output prominently (Cloud demos often stop at the success banner)."""
    if not results:
        return
    ok_n = sum(1 for r in results if r.get("status") in ("ok", "mock", "builtin"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Services run", len(results))
    c2.metric("Successful", ok_n)
    c3.metric("Mode", "Built-in" if builtin else ("Offline mocks" if offline else "Live API"))

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    cols = st.columns(min(len(results), 4))
    for col, row in zip(cols, results):
        with col:
            icon = "✅" if row.get("status") in ("ok", "mock", "builtin") else "❌"
            st.markdown(f"**{icon} {row['service']}**")
            st.caption(row.get("preview", "")[:80])

    for row in results:
        with st.expander(f"{row['service']} — full response"):
            st.json(row.get("data") or row.get("preview"))

    if "latency_ms" in df.columns and df["latency_ms"].sum() > 0:
        st.plotly_chart(
            px.bar(df, x="service", y="latency_ms", title="Latency by service"),
            use_container_width=True,
        )
    elif offline:
        st.caption("Offline mode uses instant mock responses (0 ms).")


def render_demo_runner():
    st.header("🎬 Demo Runner")
    st.write("Runs a scripted tour across core NLP services.")

    offline = st.session_state.get("offline_demo", False)
    builtin = should_use_builtin()
    if builtin:
        st.info("**Built-in engine** — all services run inside Streamlit (no API URL required).")
    elif offline:
        st.info("Offline mocks — enable built-in mode on Cloud or run `start-demo.ps1` locally for full models.")

    if st.button("▶️ Run full demo suite", type="primary"):
        results = []
        progress = st.progress(0)
        for i, sid in enumerate(DEMO_SUITE):
            progress.progress((i + 1) / len(DEMO_SUITE), text=f"Running {sid}…")
            if builtin:
                t0 = time.perf_counter()
                from toolkit.scenarios import DEMO_SAMPLES

                if sid == "qa":
                    data = invoke(sid, DEMO_SAMPLES["qa"])
                else:
                    data = invoke(sid, {"text": DEMO_SAMPLES.get(sid, "Hello")})
                ms = (time.perf_counter() - t0) * 1000
                results.append({
                    "service": sid,
                    "status": "builtin",
                    "latency_ms": round(ms, 1),
                    "preview": str(data)[:120],
                    "data": data,
                })
            elif offline:
                data = mock_response(sid)
                results.append({
                    "service": sid,
                    "status": "mock",
                    "latency_ms": 0,
                    "preview": str(data)[:120],
                    "data": data,
                })
            else:
                t0 = time.perf_counter()
                ok, data, err = run_demo_call(sid)
                ms = (time.perf_counter() - t0) * 1000
                results.append({
                    "service": sid,
                    "status": "ok" if ok else "error",
                    "latency_ms": round(ms, 1),
                    "preview": str(data if ok else err)[:120],
                    "data": data if ok else {"error": err},
                })
        st.session_state.demo_suite_results = results
        progress.empty()
        st.success("Demo suite complete — results below.")
        _display_demo_suite_results(results, offline=offline, builtin=builtin)

    elif st.session_state.get("demo_suite_results"):
        st.caption("Last run (re-run anytime with the button above).")
        _display_demo_suite_results(
            st.session_state.demo_suite_results,
            offline=offline,
            builtin=builtin,
        )


def render_session_report():
    st.header("📑 Session Report")
    hist = st.session_state.get("history", [])
    calls = st.session_state.get("api_calls_count", 0)

    md = session_report_markdown(hist, calls)
    st.markdown(md)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download report (.md)", md.encode(), "ai_toolkit_session.md")
    with col2:
        st.download_button("Download report (.html)", session_report_html(md).encode(), "ai_toolkit_session.html")

    usage = history_to_usage(hist)
    if not usage.empty:
        st.plotly_chart(px.pie(usage, names="service", values="count", hole=0.35), use_container_width=True)


def render_document_analyzer():
    st.header("📑 Document Analyzer")
    st.write("One-click sentiment + summary + keywords on any text.")

    from toolkit.document import analyze_document

    offline = st.session_state.get("offline_demo", False)
    text = st.text_area("Document", value=DEMO_SAMPLES["summarization"], height=220)
    if st.button("Analyze document", type="primary"):
        with st.spinner("Running multi-service pipeline…"):
            out = analyze_document(text, offline=offline)
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", out["word_count"])
        c2.metric("Sentiment latency", f"{out.get('sentiment_ms', 0):.0f} ms")
        c3.metric("Summary latency", f"{out.get('summary_ms', 0):.0f} ms")
        st.subheader("Sentiment")
        st.json(out["sentiment"])
        st.subheader("Summary")
        st.json(out["summary"])
        st.subheader("Keywords")
        st.write(", ".join(out.get("keywords", [])))


def render_service_compare():
    st.header("⚖️ Service Compare")
    st.write("Run the same text through sentiment and translation.")

    offline = st.session_state.get("offline_demo", False)
    text = st.text_input("Text", value=DEMO_SAMPLES["translation"])
    if st.button("Compare", type="primary"):
        from toolkit.helpers import call_api
        c1, c2 = st.columns(2)
        ok1, s, e1, ms1 = call_api("/sentiment/analyze", {"text": text}, offline=offline)
        ok2, t, e2, ms2 = call_api("/translation/translate", {"text": text}, offline=offline)
        with c1:
            st.metric("Sentiment", f"{ms1:.0f} ms")
            st.json(s if ok1 else {"error": e1})
        with c2:
            st.metric("Translation", f"{ms2:.0f} ms")
            st.json(t if ok2 else {"error": e2})


def render_integrations_hub():
    st.header("🔗 Integrations Hub")
    tab1, tab2, tab3 = st.tabs(["API keys", "Webhooks", "Governance"])

    with tab1:
        st.text_input("Demo API key", value="atk_live_demo_8f3a9c2e", disabled=True)
        st.caption("Keys are mocked for portfolio demo — wire to Vault in production.")
        if st.button("Rotate key (mock)"):
            st.success("New key: atk_live_demo_" + __import__("secrets").token_hex(4))

    with tab2:
        url = st.text_input("Webhook URL", "https://hooks.example.com/ai-toolkit")
        events = st.multiselect("Events", ["inference.completed", "model.loaded", "error"], default=["inference.completed"])
        if st.button("Test webhook"):
            st.json({"status": "delivered", "url": url, "events": events, "latency_ms": 42})

    with tab3:
        st.markdown("""
        **Data handling (demo)**  
        - Text/audio processed in-memory · not persisted to disk  
        - Session history stored only in browser session state  
        - Export history via API Call History → CSV  
        """)
        if st.button("Export audit log"):
            hist = st.session_state.get("history", [])
            st.download_button("Download audit JSON", json.dumps(hist, indent=2).encode(), "audit_log.json")


def render_cost_center():
    st.header("💰 Cost Center")
    from toolkit.cost import RATE_CARD, estimate_session_cost

    hist = st.session_state.get("history", [])
    est = estimate_session_cost(hist)
    c1, c2, c3 = st.columns(3)
    c1.metric("Session calls", est["calls"])
    c2.metric("Est. cost (USD)", f"${est['total_usd']:.4f}")
    c3.metric("Rate card", f"{len(RATE_CARD)} services")

    st.dataframe(
        pd.DataFrame([{"service": k, "usd_per_1k": v} for k, v in RATE_CARD.items()]),
        use_container_width=True,
        hide_index=True,
    )
    if est["breakdown"]:
        df = pd.DataFrame([{"service": k, "calls": v} for k, v in est["breakdown"].items()])
        st.plotly_chart(px.bar(df, x="service", y="calls", title="Calls by service"), use_container_width=True)


def render_service_health():
    """Compact per-pipeline status for sidebar or dashboard."""
    loaded, err, _ = check_status()
    metrics = get_metrics()
    if loaded:
        st.caption(f"Pipelines: {metrics.get('pipelines', '—')} · API {metrics.get('version', '')}")
