"""Usage analytics from session history (real data, not mock)."""

import pandas as pd


def history_to_usage(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["service", "count"])
    df = pd.DataFrame(history)
    return df.groupby("service").size().reset_index(name="count").sort_values("count", ascending=False)


def success_rate(history: list[dict]) -> float:
    if not history:
        return 100.0
    df = pd.DataFrame(history)
    if "success" not in df.columns:
        return 100.0
    return float(df["success"].mean() * 100)


def timeline(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["timestamp", "count"])
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df.groupby("date").size().reset_index(name="count")
