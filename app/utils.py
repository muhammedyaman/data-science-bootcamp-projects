from functools import lru_cache
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=None)
def load(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def slider(st, label, stats, step=None, fmt="%.1f"):
    lo, hi, med = stats["min"], stats["max"], stats["median"]
    if step is None:
        step = max((hi - lo) / 100, 0.01)
    return st.slider(label, float(lo), float(hi), float(med), step=float(step), format=fmt)
