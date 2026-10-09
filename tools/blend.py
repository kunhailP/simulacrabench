"""Score cached base models and blends on the hidden rows of a lab split.

    .venv/bin/python tools/blend.py --data data/proxy/gss --rep 0
    .venv/bin/python tools/blend.py --data data/proxy/gss --rep 0 --sets "eb,lin:0.001,mlp:40" "eb,lgbm"

With no --sets, prints each cached model alone. A set is blended the v7 way:
per-item convex weights fitted on OOF log loss, shrunk to instrument-wide
weights, then a per-item temperature; everything fitted on visible rows only.
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from oof import V7, cache_dir  # noqa: E402

FLOOR = 1e-3


def load(d, key, J):
    z = np.load(os.path.join(d, key + ".npz"))
    return [z[f"o{j}"].astype(np.float64) for j in range(J)], [z[f"h{j}"].astype(np.float64) for j in range(J)]


def skill(H, Yh, K):
    lp, U = [], np.mean(np.log(K))
    for j in range(len(K)):
        p = H[j] / H[j].sum(1, keepdims=True)
        p = FLOOR + (1 - K[j] * FLOOR) * p
        lp.append(np.log(p[np.arange(len(Yh)), Yh[:, j]]))
    lp = np.stack(lp, 1)
    return 1 + lp.mean() / U, lp


def blend(models, Yv, K, temper=True):
    """models: list of (oof, hid). Returns hidden tables."""
    J = len(K)
    M = len(models)

    def picked(T, j):
        ok = Yv[:, j] >= 0
        return V7._mix(T[ok][np.arange(ok.sum()), Yv[ok, j]], K[j])

    stacks = [np.stack([picked(m[0][j], j) for m in models], 1) for j in range(J)]
    glob_w = V7._eg(np.concatenate(stacks, 0), np.full(M, 1.0 / M), 0.0)
    out_o, out_h = [], []
    for j in range(J):
        w = V7._eg(stacks[j], glob_w, V7.BLEND_SHRINK) if len(stacks[j]) else glob_w
        O = sum(w[m] * models[m][0][j] for m in range(M))
        Hh = sum(w[m] * models[m][1][j] for m in range(M))
        out_o.append(O / O.sum(1, keepdims=True))
        out_h.append(Hh / Hh.sum(1, keepdims=True))
    if temper:
        for j in range(J):
            ok = Yv[:, j] >= 0
            P, y = out_o[j][ok], Yv[ok, j]
            bestT, bv = 1.0, -np.inf
            for T in V7.TEMPS:
                ll = np.log(V7._mix(V7._temper(P, T)[np.arange(len(y)), y], K[j])).sum()
                ll -= V7.TEMP_SHRINK * len(y) / 100.0 * np.log(T) ** 2
                if ll > bv:
                    bestT, bv = T, ll
            out_h[j] = V7._temper(out_h[j], bestT)
    return out_h, glob_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    d = cache_dir(a.data, a.rep)
    t = np.load(os.path.join(d, "_truth.npz"))
    Yh, Yv, K = t["Yh"], t["Yv"], t["K"]
    J = len(K)
    keys = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.npz"))
                  if not os.path.basename(f).startswith("_"))
    if not a.sets:
        for k in keys:
            o, h = load(d, k, J)
            s_raw, _ = skill(h, Yh, K)
            hb, _ = blend([(o, h)], Yv, K)
            s_t, _ = skill(hb, Yh, K)
            print(f"{k:28s} raw {s_raw:.4f}   tempered {s_t:.4f}")
        return
    for s in a.sets:
        names = s.split(",")
        ms = [load(d, n, J) for n in names]
        hb, w = blend(ms, Yv, K)
        sk, _ = skill(hb, Yh, K)
        print(f"{s:50s} {sk:.4f}  w={np.round(w, 3).tolist()}")


if __name__ == "__main__":
    main()
