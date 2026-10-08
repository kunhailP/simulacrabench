"""Split oracle headroom into estimation vs approximation error.

Same world, same hidden DEV rows, but the visible TRAIN rows are re-drawn at
growing sizes. If the excess KL(P* || Q) falls toward zero as n grows, the gap
is estimation error (sample size; shrinkage and priors can win it back). If it
stays flat, the method's model class cannot represent P*.

    .venv/bin/python tools/learning_curve.py sub/v2 --inst world_bank --scenario base --sizes 5195,20000,80000
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluate import load_sub  # noqa: E402
from make_sandbox import load_config, load_schema  # noqa: E402
from score import floored, load_frames, sample_rows  # noqa: E402
from world import SCENARIOS, World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--inst", default="world_bank")
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sizes", default="5195,20000,80000")
    a = ap.parse_args()
    config = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    floor = config["scoring"]["floor"]
    schema = load_schema(os.path.join(ROOT, "raw", "repo", "data", a.inst + ".json"), config)
    d = os.path.join(ROOT, "data", "worlds", a.scenario, a.inst)
    masked, cells, _ = sample_rows(schema, load_frames(d, schema), 1)
    items = schema["items"]
    targets = [k for k, r in items.items() if r["class"] == "PREDICT"]
    hidden = masked[masked[targets].isna().any(axis=1)].reset_index(drop=True)
    O = np.load(os.path.join(d, "oracle.npz"))
    orc = floored([O["DEV::" + t][r] for r in range(len(hidden)) for t in targets],
                  schema, cells, floor)
    W = World(schema, a.seed, **SCENARIOS[a.scenario])
    mod = load_sub(os.path.abspath(a.submission))
    for n in [int(x) for x in a.sizes.split(",")]:
        rng = np.random.default_rng([a.seed, 100 + n])
        codes, cols = W.sample_given(n, rng)
        ans, _ = W.sample_predict(codes, cols, rng)
        vis = pd.DataFrame({k: (cols[k] if k in cols else ans[k]) for k in W.names}).astype(object)
        vis.insert(0, "respondent_id", ["V%07d" % i for i in range(n)])
        frame = pd.concat([vis, hidden[vis.columns]], ignore_index=True)
        t = time.time()
        vec = mod.predict(frame, json.loads(json.dumps(schema)))
        vec = floored(vec, schema, cells, floor)
        kl = np.mean([(p * (np.log(p) - np.log(q))).sum() for p, q in zip(orc, vec)])
        info = getattr(mod.predict, "last_info", {}) or {}
        print("n=%6d  excess KL %.4f nats/cell  %.0fs  w=%s" % (
            n, kl, time.time() - t, info.get("glob_w")), flush=True)


if __name__ == "__main__":
    main()
