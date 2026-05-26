"""AI Toolkit — dark enterprise Streamlit theme."""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ACCENT = ("#667eea", "#764ba2")
PRIMARY = "#818cf8"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
.block-container { padding-top: 1.25rem; max-width: 1400px; }
.ep-hero, .main-header {
  background: linear-gradient(135deg, ACCENT_A 0%, ACCENT_B 100%) !important;
  padding: 1.75rem 2rem !important; border-radius: 16px !important; color: #fff !important;
  box-shadow: 0 16px 48px rgba(102, 126, 234, 0.35); border: 1px solid rgba(255,255,255,0.08);
  text-align: center; margin-bottom: 1.25rem;
}
.ep-hero h1, .main-header h1 { margin: 0 !important; font-size: 1.85rem !important; color: #fff !important; }
.ep-hero p, .main-header p { margin: 0.4rem 0 0 !important; opacity: 0.92; color: #fff !important; }
div[data-testid="stMetric"] {
  background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 12px;
}
section[data-testid="stSidebar"], div[data-testid="stSidebar"], [data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"], [data-testid="stSidebarNav"] {
  background-color: #0f172a !important;
  background-image: linear-gradient(180deg, #0f172a 0%, #312e81 100%) !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] .stCaption,
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span { color: #e2e8f0 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: #1e293b !important; color: #f8fafc !important;
}
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, ACCENT_A, ACCENT_B) !important; color: #fff !important;
}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stRadio label p { color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; background: rgba(30,41,59,0.6); color: #e2e8f0; }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, ACCENT_A, ACCENT_B) !important; color: #fff !important;
}
.feature-card, .sidebar-card, .content-display, .generated-content {
  background: rgba(30, 41, 59, 0.9) !important; color: #e2e8f0 !important;
  border: 1px solid rgba(148, 163, 184, 0.15) !important; border-radius: 12px !important;
}
.sentiment-positive { background: rgba(6,78,59,0.5) !important; color: #a7f3d0 !important; border: 2px solid #22c55e !important; border-radius: 12px; padding: 1rem; text-align: center; }
.sentiment-negative { background: rgba(127,29,29,0.45) !important; color: #fecaca !important; border: 2px solid #ef4444 !important; border-radius: 12px; padding: 1rem; text-align: center; }
.original-text { background: rgba(120,53,15,0.35) !important; color: #fde68a !important; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 10px; }
.translated-text { background: rgba(12,74,110,0.4) !important; color: #bae6fd !important; border-left: 4px solid #0ea5e9; padding: 1rem; border-radius: 10px; }
.transcription-result { background: rgba(30,58,138,0.4) !important; color: #bfdbfe !important; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 10px; }
</style>
""".replace("ACCENT_A", ACCENT[0]).replace("ACCENT_B", ACCENT[1])


def inject_theme() -> None:
    pio.templates.default = "plotly_dark"
    st.markdown(_CSS, unsafe_allow_html=True)


def hero_html(title: str, subtitle: str, icon: str = "") -> str:
    return f'<div class="ep-hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>'


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(15,23,42,0.5)", font=dict(color="#e2e8f0"))
    return fig
