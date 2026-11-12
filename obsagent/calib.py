"""Load bench calibration (series per metric group, spans per request)."""
import json
import os

from .config import DEFAULT_SERIES, DEFAULT_SPANS_PER_REQ

RESULTS = os.environ.get("RESULTS_DIR", "/work/results")


def load():
    path = os.path.join(RESULTS, "calibration.json")
    if os.path.exists(path):
        with open(path) as f:
            c = json.load(f)
        return c["series"], c["spans_per_req"], True
    return dict(DEFAULT_SERIES), DEFAULT_SPANS_PER_REQ, False


def load_noise():
    """Real reading errors measured by noise_fit.py, or None."""
    path = os.path.join(RESULTS, "obs_noise.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
