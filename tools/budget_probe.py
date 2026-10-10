"""What does a submission actually run inside the Dev budget? (sandbox shapes, real sizes)

Imports the submission once and calls predict() on the three sandbox
instruments in leaderboard order, as the grader does, printing predict.last_info
(items reached by TabICL, gate factorisation, pool weights, seconds). The H100
is faster than our RTX 3090 by an unknown factor; --speedup s runs with a
budget of s x 900 s x SAFETY so that what a 3090 does here is roughly what an
H100 s times faster would do in 900 s.

    .venv/bin/python tools/budget_probe.py sub/v8 --speedup 1
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))

import pandas as pd  # noqa: E402
from make_sandbox import load_config, load_schema  # noqa: E402

ORDER = ["unicef", "world_bank", "unhcr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--phase", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    budget = (900 if a.phase == 1 else 3600) * 0.80 * a.speedup
    os.environ["V8_BUDGET"] = str(budget)
    t0 = time.time()
    sys.path.insert(0, os.path.abspath(a.sub))
    import main as sub  # noqa: E402
    print(f"import {time.time() - t0:.0f}s, budget {budget:.0f}s (speedup {a.speedup})", flush=True)
    for inst in ORDER:
        d = os.path.join(ROOT, "data", "sandbox", inst)
        schema = load_schema(os.path.join(ROOT, "raw", "repo", "data", f"{inst}.json"), cfg)
        files = [f for f in os.listdir(d) if f.endswith((".parquet", ".csv"))]
        f = os.path.join(d, files[0])
        frame = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        roles = frame.get("role")
        if roles is not None:
            vis = roles.isin(["TRAIN"] + (["DEV"] if a.phase == 2 else []))
            hid = roles == ("DEV" if a.phase == 1 else "TEST")
            frame = frame[vis | hid].copy()
            pred = [k for k, r in schema["items"].items() if r["class"] == "PREDICT"]
            frame.loc[hid[vis | hid].to_numpy(), pred] = None
            frame = frame[["respondent_id"] + [k for k, r in schema["items"].items()
                                                if r["class"] in ("GIVEN", "PREDICT")]]
        t = time.time()
        sub.predict(frame, json.loads(json.dumps(schema)))
        info = getattr(sub.predict, "last_info", {})
        print(f"{inst:10s} {time.time() - t:6.0f}s  total {time.time() - t0:6.0f}s  "
              f"{json.dumps(info, default=str)[:400]}", flush=True)


if __name__ == "__main__":
    main()
