"""Session report export."""

from datetime import datetime

from toolkit.analytics import history_to_usage, success_rate
from toolkit.config import APP_NAME


def session_report_markdown(history: list[dict], api_calls: int) -> str:
    usage = history_to_usage(history)
    lines = [
        f"# {APP_NAME} — Session Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Product: {APP_NAME}",
        "",
        "## Summary",
        f"- Total API calls: **{api_calls}**",
        f"- Success rate: **{success_rate(history):.1f}%**",
        "",
        "## Usage by service",
    ]
    if not usage.empty:
        for _, row in usage.iterrows():
            lines.append(f"- {row['service']}: {int(row['count'])} calls")
    else:
        lines.append("- No calls logged this session.")
    lines.extend(["", "## Recent activity", ""])
    for h in history[-15:]:
        status = "OK" if h.get("success", True) else "FAIL"
        lat = f" ({h['latency_ms']} ms)" if h.get("latency_ms") else ""
        lines.append(f"- [{h.get('timestamp')}] **{h.get('service')}** {status}{lat}")
    lines.append("\n---\n*AI Toolkit Enterprise session export*")
    return "\n".join(lines)


def session_report_html(md: str, title: str = APP_NAME) -> str:
    body = md.replace("\n", "<br>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui;max-width:800px;margin:2rem auto;padding:1rem;}}</style>
</head><body>{body}</body></html>"""
