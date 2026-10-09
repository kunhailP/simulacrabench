"""Hidden-row skill of TabICL after instrument-level adaptation on the visible rows.

    .venv/bin/python research/exp_instft.py --data data/proxy/gss --rep 0 --steps 100,300
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

import candidates as C  # noqa: E402
from blend import skill  # noqa: E402
from exp_tabicl import features, run  # noqa: E402
from instft import adapt  # noqa: E402
from oof import build, cache_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--steps", default="100,300")
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    from tabicl import TabICLClassifier
    from tabicl._sklearn.preprocessing import TransformToNumerical
    C._patch_tabicl()
    schema, ins = build(a.data, a.rep)
    t = np.load(os.path.join(cache_dir(a.data, a.rep), "_truth.npz"))
    X = features(ins, "cat")
    Xnum = np.asarray(TransformToNumerical().fit_transform(X), dtype=np.float32)
    probe = TabICLClassifier(device="cuda", n_estimators=1)
    probe._load_model()
    key = next(iter(C._TABICL))
    base = C._TABICL[key]
    for steps in map(int, a.steps.split(",")):
        t0 = time.time()
        m, losses = adapt(base[0], Xnum, ins.Y, ins.vis, steps=steps, lr=a.lr)
        C._TABICL[key] = (m, base[1], base[2])
        H = run(ins, 1, "cat", 0.9)
        C._TABICL[key] = base
        s, _ = skill(H, t["Yh"], t["K"])
        print(f"steps {steps} lr {a.lr}: hidden raw {s:.4f}  (plain full1 = 0.4811 on gss rep0) "
              f"loss first/last 20: {np.mean(losses[:20]):.3f}/{np.mean(losses[-20:]):.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
