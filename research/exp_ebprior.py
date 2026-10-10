"""Does shrinking cell counts toward a model help, and at which partition?

    .venv/bin/python research/exp_ebprior.py --data data/proxy/gss --rep 0 --prior eb
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

from blend import blend, load, skill  # noqa: E402
from ebprior import correct  # noqa: E402
from oof import V7, build, cache_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--prior", required=True, help="comma list: blended to form the prior")
    ap.add_argument("--modes", default="mult")
    ap.add_argument("--fixed", default="", help="';'-separated fixed partitions, e.g. 'country;country,relig'")
    a = ap.parse_args()
    schema, ins = build(a.data, a.rep)
    d = cache_dir(a.data, a.rep)
    t = np.load(os.path.join(d, "_truth.npz"))
    Yh, Yv, K = t["Yh"], t["Yv"], t["K"]
    J = len(K)
    rng = np.random.default_rng(V7.SEED)
    perm = rng.permutation(len(ins.vis))
    folds = [perm[i::V7.N_FOLDS] for i in range(V7.N_FOLDS)]

    ms = [load(d, n, J) for n in a.prior.split(",")]
    if len(ms) == 1:
        po, ph = ms[0]
    else:  # blend without temperature: OOF of a blend fitted on OOF is mildly optimistic
        po_h, _ = blend(ms, Yv, K, temper=False)
        raise SystemExit("multi-model prior: TODO")
    base, _ = skill(blend([(po, ph)], Yv, K)[0], Yh, K)
    print(f"prior {a.prior}: tempered {base:.4f}")

    yv = ins.Y[ins.vis]
    rank = []
    for j in range(J):
        ok = yv[:, j] >= 0
        rank.append(sorted(ins.given, reverse=True, key=lambda g: V7._mi(
            ins.gcodes[g][ins.vis][ok], yv[ok, j], ins.gcard[g], K[j])))
    parts = {f"top{k}": (lambda j, k=k: rank[j][:k]) for k in (1, 2, 3, 4, 5)}
    parts["full"] = lambda j: ins.given
    for spec in filter(None, a.fixed.split(";")):
        cols = spec.split(",")
        parts[spec] = (lambda j, cols=cols: cols)
    for name, f in parts.items():
        for mode in a.modes.split(","):
            o, h, al = correct(ins, po, ph, f, folds, mode)
            s, _ = skill(blend([(o, h)], Yv, K)[0], Yh, K)
            print(f"  +EB[{name:14s},{mode}] {s:.4f} ({s - base:+.4f})  alpha median {np.median(al):.1f}", flush=True)


if __name__ == "__main__":
    main()
