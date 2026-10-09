"""Does fine-tuning TabICL on the visible rows help? Per-item check on a few items.

    .venv/bin/python research/exp_finetune.py --data data/proxy/gss --rep 0 --items 12
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

from candidates import _patch_tabicl  # noqa: E402
from oof import build, cache_dir  # noqa: E402


def ll(P, y, K):
    p = 1e-3 + (1 - K * 1e-3) * P[np.arange(len(y)), y]
    return np.log(p).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    from tabicl import FinetunedTabICLClassifier, TabICLClassifier
    _patch_tabicl()
    schema, ins = build(a.data, a.rep)
    t = np.load(os.path.join(cache_dir(a.data, a.rep), "_truth.npz"))
    Yh = t["Yh"]
    G = np.stack([ins.gcodes[g] for g in ins.given], 1)
    X = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    nobs = (ins.Y[ins.vis] >= 0).sum(0)
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(nobs > 2000), a.items, replace=False)
    tot = []
    for j in pick:
        y = ins.Y[ins.vis, j]
        ok = y >= 0
        K = ins.K[j]
        Xtr, ytr = X.iloc[ins.vis[ok]], y[ok]
        Xte, yte = X.iloc[ins.hid], Yh[:, j]
        res = {}
        for name in ("plain", "ft"):
            t0 = time.time()
            if name == "plain":
                m = TabICLClassifier(device="cuda", n_estimators=1, random_state=0)
            else:
                m = FinetunedTabICLClassifier(device="cuda", epochs=a.epochs, learning_rate=a.lr,
                                              eval_metric="log_loss", n_estimators_inference=1,
                                              random_state=0, save_interval=10 ** 6)
            m.fit(Xtr, ytr)
            P = np.full((len(Xte), K), 1e-6)
            P[:, m.classes_] = m.predict_proba(Xte)
            P /= P.sum(1, keepdims=True)
            res[name] = (ll(P, yte, K), time.time() - t0)
        tot.append(res["ft"][0] - res["plain"][0])
        print(f"{ins.targets[j]:12s} K={K:2d} plain {res['plain'][0]:.4f} ft {res['ft'][0]:.4f} "
              f"(diff {tot[-1]:+.4f} nats, ft {res['ft'][1]:.0f}s)", flush=True)
    print(f"mean diff {np.mean(tot):+.4f} nats/cell (positive = fine-tuning helps)")


if __name__ == "__main__":
    main()
