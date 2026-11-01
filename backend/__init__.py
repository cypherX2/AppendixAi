"""Backend package.

- backend.lite.LitePredictor      → NumPy-only runtime (used by the HTML website / Vercel)
- backend.predictor.AppendicitisPredictor → full CatBoost runtime (used by the Streamlit app)

The full predictor is imported lazily so the light website never loads CatBoost/pandas.
"""
from .lite import UNKNOWN, LitePredictor, pretty

__all__ = ["AppendicitisPredictor", "LitePredictor", "UNKNOWN", "pretty"]


def __getattr__(name):
    if name == "AppendicitisPredictor":
        from .predictor import AppendicitisPredictor
        return AppendicitisPredictor
    raise AttributeError(name)
