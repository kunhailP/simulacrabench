"""v7 blend vs TabICL vs log-linear pool of the two, over repeats, with paired SEs.

    .venv/bin/python research/exp_confirm.py --data data/proxy/gss --reps 0,1,2
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

from blend import blend, load, skill  # noqa: E402
from exp_pool import geo_apply, geo_fit  # noqa: E402
from oof import cache_dir  # noqa: E402

V7SET = ["marg", "eb", "lin:3e-05", "mlp:15", "sint:0.3_0.03_3.0"]


def tempered_pair(ms, Yv, K):
    """Blend (no temper) -> OOF & hidden tables of the blend."""
    from blend import V7
    J = len(K)
    M = len(ms)

    def picked(T, j):
        ok = Yv[:, j] >= 0
        return V7._mix(T[ok][np.arange(ok.sum()), Yv[ok, j]], K[j])
    stacks = [np.stack([picked(m[0][j], j) for m in ms], 1) for j in range(J)]
    g = V7._eg(np.concatenate(stacks, 0), np.full(M, 1.0 / M), 0.0)
    O, H = [], []
    for j in range(J):
        w = V7._eg(stacks[j], g, V7.BLEND_SHRINK)
        o = sum(w[m] * ms[m][0][j] for m in range(M))
        h = sum(w[m] * ms[m][1][j] for m in range(M))
        O.append(o / o.sum(1, keepdims=True))
        H.append(h / h.sum(1, keepdims=True))
    return O, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", default="0,1,2")
    a = ap.parse_args()
    rows = []
    for rep in map(int, a.reps.split(",")):
        d = cache_dir(a.data, rep)
        t = np.load(os.path.join(d, "_truth.npz"))
        Yh, Yv, K = t["Yh"], t["Yv"], t["K"]
        J = len(K)
        v7 = tempered_pair([load(d, n, J) for n in V7SET], Yv, K)
        tab = load(d, "tabicl", J)
        res = {}
        for name, (o, h) in {"v7blend": v7, "tabicl": tab}.items():
            hb, _ = blend([(o, h)], Yv, K)
            res[name] = skill(hb, Yh, K)[1]
        w = geo_fit([v7, tab], Yv, K)
        go, gh = geo_apply([v7, tab], w, 0), geo_apply([v7, tab], w, 1)
        hb, _ = blend([(go, gh)], Yv, K)
        res["geo(v7,tab)"] = skill(hb, Yh, K)[1]
        U = np.mean(np.log(K))
        base = res["v7blend"]
        line = [f"rep{rep}"]
        for k, lp in res.items():
            sk = 1 + lp.mean() / U
            dper = (lp - base).mean(1) / U  # per respondent
            se = dper.std() / np.sqrt(len(dper))
            line.append(f"{k} {sk:.4f} ({dper.mean():+.4f}±{se:.4f})")
            rows.append((rep, k, dper.mean(), se))
        print("  ".join(line), f"w={np.round(w, 3).tolist()}", flush=True)
    for k in ["tabicl", "geo(v7,tab)"]:
        ds = [r[2] for r in rows if r[1] == k]
        ses = [r[3] for r in rows if r[1] == k]
        print(f"{k} - v7blend: mean {np.mean(ds):+.4f}, se {np.sqrt(np.sum(np.square(ses))) / len(ses):.4f}")


if __name__ == "__main__":
    main()
