from typing import Any


_latest_metrics: dict[str, Any] = {}


def save_latest_metrics(metrics: dict[str, Any]) -> None:
    global _latest_metrics
    _latest_metrics = metrics


def get_latest_metrics() -> dict[str, Any]:
    return _latest_metrics or {
        "message": "No metrics recorded yet. Call /ask, /compare, /lit-review, or /extract-insights first."
    }