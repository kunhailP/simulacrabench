"""TabICL configurations, fitted on all visible rows and scored on the hidden rows.

    .venv/bin/python research/exp_tabicl.py --data data/proxy/gss --rep 0 --configs full4,full8
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blend import skill  # noqa: E402
from oof import build, cache_dir  # noqa: E402


def features(ins, enc):
    G = np.stack([ins.gcodes[g] for g in ins.given], 1)
    if enc == "cat":
        return pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    if enc == "ord":
        return pd.DataFrame(G.astype(float), columns=ins.given)
    if enc == "both":  # ordinal code plus categorical copy
        df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
        for i, g in enumerate(ins.given):
            df[g + "_o"] = G[:, i].astype(float)
        return df
    raise ValueError(enc)


def run(ins, n_est, enc, temp):
    from tabicl import TabICLClassifier
    from candidates import _patch_tabicl
    _patch_tabicl()
    X = features(ins, enc)
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[ins.vis, j]
        ok = y >= 0
        K = ins.K[j]
        cl = np.unique(y[ok])
        P = np.full((len(ins.hid), K), 1e-6)
        if len(cl) < 2:
            P[:, cl] = 1
        else:
            m = TabICLClassifier(device="cuda", n_estimators=n_est, random_state=0,
                                 softmax_temperature=temp)
            m.fit(X.iloc[ins.vis[ok]], y[ok])
            P[:, m.classes_] = m.predict_proba(X.iloc[ins.hid])
        out.append(P / P.sum(1, keepdims=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--configs", default="full4")
    a = ap.parse_args()
    schema, ins = build(a.data, a.rep)
    t = np.load(os.path.join(cache_dir(a.data, a.rep), "_truth.npz"))
    for cfg in a.configs.split(","):
        # full<n>[_enc][_t<temp>]
        parts = cfg.split("_")
        n_est = int(parts[0][4:])
        enc = next((p for p in parts[1:] if p in ("cat", "ord", "both")), "cat")
        temp = next((float(p[1:]) for p in parts[1:] if p.startswith("t")), 0.9)
        t0 = time.time()
        H = run(ins, n_est, enc, temp)
        s, _ = skill(H, t["Yh"], t["K"])
        np.savez_compressed(os.path.join(cache_dir(a.data, a.rep), f"_hid_tabicl_{cfg}.npz"),
                            **{f"h{j}": H[j].astype(np.float32) for j in range(len(H))})
        print(f"{cfg:16s} hidden raw {s:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
