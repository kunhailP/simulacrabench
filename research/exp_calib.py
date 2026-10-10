"""Calibration beyond one temperature: TabICL under-predicts small probabilities
by 1.5-1.9x on every lab (research note 2026-10-10). Per item, fitted on OOF
tables with the official floor, evaluated on the hidden rows:

  T      p^(1/T) renormalised                       (v8's per-item temperature)
  T+m    (1-e) p^(1/T) + e * marginal_j             (shrink toward the item marginal)
  T+u    (1-e) p^(1/T) + e * uniform
  shared the T+m family with (T, e) shared across items of the instrument

    .venv/bin/python research/exp_calib.py --data data/proxy/gss --model tabicl1
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from blend import load  # noqa: E402
from oof import cache_dir  # noqa: E402

FLOOR = 1e-3
TS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7)
ES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)


def apply(P, T, e, base):
    Q = np.power(np.maximum(P, 1e-12), 1.0 / T)
    Q /= Q.sum(1, keepdims=True)
    return (1 - e) * Q + e * base


def ll(P, y, K):
    p = P / P.sum(1, keepdims=True)
    p = FLOOR + (1 - K * FLOOR) * p
    return np.log(p[np.arange(len(y)), y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--model", default="tabicl1")
    a = ap.parse_args()
    d = cache_dir(a.data, a.rep)
    t = np.load(os.path.join(d, "_truth.npz"))
    Yh, Yv, K = t["Yh"], t["Yv"], t["K"]
    J = len(K)
    O, H = load(d, a.model, J)
    U = np.mean(np.log(K))
    res = {k: 0.0 for k in ("raw", "T", "T+m", "T+u", "shared")}
    grid_sh = np.zeros((len(TS), len(ES)))
    keep = []
    for j in range(J):
        ok = Yv[:, j] >= 0
        y, Oj = Yv[ok, j], O[j][ok]
        marg = (np.bincount(y, minlength=K[j]) + 0.5) / (len(y) + 0.5 * K[j])
        uni = np.full(K[j], 1.0 / K[j])
        res["raw"] += ll(H[j], Yh[:, j], K[j]).mean()
        bestT = max(TS, key=lambda T: ll(apply(Oj, T, 0, uni), y, K[j]).sum())
        res["T"] += ll(apply(H[j], bestT, 0, uni), Yh[:, j], K[j]).mean()
        for name, base in (("T+m", marg), ("T+u", uni)):
            T, e = max(((T, e) for T in TS for e in ES),
                       key=lambda te: ll(apply(Oj, te[0], te[1], base), y, K[j]).sum())
            res[name] += ll(apply(H[j], T, e, base), Yh[:, j], K[j]).mean()
        for a_, T in enumerate(TS):
            for b_, e in enumerate(ES):
                grid_sh[a_, b_] += ll(apply(Oj, T, e, marg), y, K[j]).mean()
        keep.append(marg)
    a_, b_ = np.unravel_index(np.argmax(grid_sh), grid_sh.shape)
    for j in range(J):
        res["shared"] += ll(apply(H[j], TS[a_], ES[b_], keep[j]), Yh[:, j], K[j]).mean()
    base = 1 + res["raw"] / J / U
    print(f"{os.path.basename(a.data)} {a.model}: " + "  ".join(
        f"{k} {1 + v / J / U:.4f} ({1 + v / J / U - base:+.4f})" for k, v in res.items())
        + f"   shared T={TS[a_]}, e={ES[b_]}")


if __name__ == "__main__":
    main()
