"""Linear vs log-linear (geometric) pooling of cached models, weights fitted on OOF only.

    .venv/bin/python research/exp_pool.py --data data/proxy/gss --rep 0 --models eb,lgbm,mlp:20,tabicl
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from blend import blend, load, skill  # noqa: E402
from oof import cache_dir  # noqa: E402

FLOOR = 1e-3


def geo_fit(ms, Yv, K, steps=300):
    """Global log-linear weights (unconstrained, so they also set a temperature)."""
    J = len(K)
    L, Y, M = [], [], len(ms)
    for j in range(J):
        ok = Yv[:, j] >= 0
        lg = np.stack([np.log(np.maximum(m[0][j][ok], 1e-9)) for m in ms], 0)  # M x n x K
        L.append(torch.tensor(lg))
        Y.append(torch.tensor(Yv[ok, j]))
    w = torch.full((M,), 1.0 / M, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tot, n = 0.0, 0
        for lg, y in zip(L, Y):
            z = torch.einsum("m,mnk->nk", w, lg)
            tot = tot - torch.log_softmax(z, -1).gather(1, y[:, None]).sum()
            n += len(y)
        loss = tot / n
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def geo_apply(ms, w, which):
    out = []
    for j in range(len(ms[0][which])):
        z = sum(w[m] * np.log(np.maximum(ms[m][which][j], 1e-9)) for m in range(len(ms)))
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        out.append(p / p.sum(1, keepdims=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--models", required=True)
    a = ap.parse_args()
    d = cache_dir(a.data, a.rep)
    t = np.load(os.path.join(d, "_truth.npz"))
    Yh, Yv, K = t["Yh"], t["Yv"], t["K"]
    names = a.models.split(",")
    ms = [load(d, n, len(K)) for n in names]
    lin, _ = blend(ms, Yv, K)
    print(f"linear+temper   {skill(lin, Yh, K)[0]:.4f}")
    w = geo_fit(ms, Yv, K)
    geo = geo_apply(ms, w, 1)
    print(f"geometric       {skill(geo, Yh, K)[0]:.4f}  w={np.round(w, 3).tolist()}")
    g_o = geo_apply(ms, w, 0)
    gt, _ = blend([(g_o, geo)], Yv, K)
    print(f"geometric+temper {skill(gt, Yh, K)[0]:.4f}")


if __name__ == "__main__":
    main()
