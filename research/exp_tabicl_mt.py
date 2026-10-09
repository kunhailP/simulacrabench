"""TabICL with a multi-task representation: PCA of a cross-fitted multi-task model's
log-probabilities over all items, appended to the GIVEN columns.

OOF tables for visible rows never used that row's labels, so the appended
features carry no row-level target leakage into the TabICL context.

    .venv/bin/python research/exp_tabicl_mt.py --data data/proxy/gss --rep 0 --rep-model mlp:20 --dims 16
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

from blend import load, skill  # noqa: E402
from oof import build, cache_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--rep-model", default="mlp:20")
    ap.add_argument("--dims", default="16")
    ap.add_argument("--n-est", type=int, default=4)
    a = ap.parse_args()
    from tabicl import TabICLClassifier
    schema, ins = build(a.data, a.rep)
    d = cache_dir(a.data, a.rep)
    t = np.load(os.path.join(d, "_truth.npz"))
    J = len(ins.targets)
    o, h = load(d, a.rep_model, J)
    Lv = np.concatenate([np.log(np.maximum(x, 1e-6)) for x in o], 1)
    Lh = np.concatenate([np.log(np.maximum(x, 1e-6)) for x in h], 1)
    mu, sd = Lv.mean(0), Lv.std(0) + 1e-6
    Zv, Zh = (Lv - mu) / sd, (Lh - mu) / sd
    U, S, Vt = np.linalg.svd(Zv, full_matrices=False)
    G = np.stack([ins.gcodes[g] for g in ins.given], 1)
    for dims in map(int, a.dims.split(",")):
        Pv, Ph = Zv @ Vt[:dims].T, Zh @ Vt[:dims].T
        F = np.zeros((len(ins.Y), dims))
        F[ins.vis], F[ins.hid] = Pv, Ph
        X = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
        for k in range(dims):
            X[f"z{k}"] = F[:, k]
        t0 = time.time()
        out = []
        for j in range(J):
            y = ins.Y[ins.vis, j]
            ok = y >= 0
            K = ins.K[j]
            cl = np.unique(y[ok])
            P = np.full((len(ins.hid), K), 1e-6)
            if len(cl) < 2:
                P[:, cl] = 1
            else:
                m = TabICLClassifier(device="cuda", n_estimators=a.n_est, random_state=0)
                m.fit(X.iloc[ins.vis[ok]], y[ok])
                P[:, m.classes_] = m.predict_proba(X.iloc[ins.hid])
            out.append(P / P.sum(1, keepdims=True))
        s, _ = skill(out, t["Yh"], t["K"])
        np.savez_compressed(os.path.join(d, f"_hid_tabicl_mt{dims}.npz"),
                            **{f"h{j}": out[j].astype(np.float32) for j in range(J)})
        print(f"tabicl+mt{dims} ({a.rep_model}) hidden raw {s:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
