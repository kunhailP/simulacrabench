"""E6: does ordinal smoothing of a model's table help ordered items?

For an item whose substantive levels are coded "1. ...", "2. ...", ... (or
banded numbers), mix each probability with its neighbours along that order:
q = (1 - h) p + h (K_ord p), K_ord a row-normalised tridiagonal kernel over the
substantive levels only; non-substantive levels (sentinel, "Not answered",
DK/refused, "Not recorded") keep their mass and stay out of the order. h is
chosen per item from a grid including 0 by the floored OOF log score, then the
hidden tables get the same h. Reported against the unsmoothed table.

    .venv/bin/python research/exp_ordinal.py --data data/proxy/gss --rep 0 --model tabicl1
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

from blend import load  # noqa: E402
from oof import cache_dir  # noqa: E402

FLOOR = 1e-3
H_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)
NONSUB = re.compile(r"(NA_GATED|not answered|don.?t know|refused|no answer|not recorded|"
                    r"not asked|skipped|missing|prefer not|other missing|\(dk\)|^i$)", re.I)


def ordered_levels(opts):
    """Indices of substantive levels in their order, or None if not ordinal-coded."""
    idx, keys = [], []
    for k, v in enumerate(opts):
        if NONSUB.search(v):
            continue
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)", v.replace("<", "").replace(">", ""))
        if not m:
            return None
        idx.append(k)
        keys.append(float(m.group(1)))
    if len(idx) < 3 or len(set(keys)) != len(keys):
        return None
    return [i for _, i in sorted(zip(keys, idx))]


def smooth(P, order, h):
    if h == 0:
        return P
    Q = P.copy()
    S = P[:, order]
    n = S.shape[1]
    T = np.zeros((n, n))
    for i in range(n):
        nb = [x for x in (i - 1, i + 1) if 0 <= x < n]
        for x in nb:
            T[i, x] = 1 / len(nb)
    Q[:, order] = (1 - h) * S + h * S @ T   # mass leaving i is spread to its neighbours
    return Q


def floored_ll(P, y, K):
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
    Yh, Yv, K, T = t["Yh"], t["Yv"], t["K"], list(t["targets"])
    J = len(K)
    O, H = load(d, a.model, J)
    sch = json.load(open(os.path.join(a.data, "schema.json")))["items"]
    U = np.mean(np.log(K))
    base, new, n_ord, picked = 0.0, 0.0, 0, []
    for j in range(J):
        opts = sch[T[j]]["values"] + (["NA_GATED"] if sch[T[j]].get("gate") else [])
        b = floored_ll(H[j], Yh[:, j], K[j]).mean()
        base += b
        order = ordered_levels(opts)
        if order is None:
            new += b
            continue
        n_ord += 1
        ok = Yv[:, j] >= 0
        scores = [floored_ll(smooth(O[j][ok], order, h), Yv[ok, j], K[j]).sum() for h in H_GRID]
        h = H_GRID[int(np.argmax(scores))]
        picked.append(h)
        new += floored_ll(smooth(H[j], order, h), Yh[:, j], K[j]).mean()
    sb, sn = 1 + base / J / U, 1 + new / J / U
    print(f"{os.path.basename(a.data)} {a.model}: ordinal items {n_ord}/{J}, h picked "
          f"{dict(zip(*np.unique(picked, return_counts=True))) if picked else {}}")
    print(f"  skill (raw, untempered) {sb:.4f} -> {sn:.4f} ({sn - sb:+.4f})")


if __name__ == "__main__":
    main()
