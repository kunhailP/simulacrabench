"""v7 trained on 50% of the visible rows (learning-curve probe, lab only)."""
import importlib.util, os
import numpy as np
_p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sub", "v7", "main.py")
_s = importlib.util.spec_from_file_location("v7lc50", _p)
V7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V7)


def predict(frame, schema):
    tg = [k for k, r in schema["items"].items() if r["class"] == "PREDICT"]
    hid = frame[tg].isna().any(axis=1).to_numpy()
    rng = np.random.default_rng(7)
    keep = hid | (rng.random(len(frame)) < 50 / 100)
    out = V7.predict(frame[keep].reset_index(drop=True), schema)
    predict.last_info = getattr(V7.predict, "last_info", None)
    return out
