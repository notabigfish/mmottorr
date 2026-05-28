from __future__ import annotations


def run_posebusters_if_available(prediction_csv: str, output_csv: str) -> dict:
    try:
        import posebusters  # noqa: F401
    except Exception:
        return {"available": False, "reason": "posebusters not installed"}
    return {
        "available": True,
        "reason": "posebusters installed; full molecular export not implemented in milestone 3",
        "prediction_csv": prediction_csv,
        "output_csv": output_csv,
    }
