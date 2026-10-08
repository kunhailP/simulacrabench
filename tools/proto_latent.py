"""H3 prototype: respondent-level latent factor model vs per-item MLP.

    y_j | x, z ~ softmax(eta_j(x) + Lambda_j z),   z | x ~ N(mu(x), I_D)

Fitted by importance-weighted ELBO with an amortised encoder q(z | x, y);
the encoder is a training device only. A hidden row has x alone, so its
forecast is the prior marginal E_{z ~ N(mu(x), I)} softmax(...), by Monte Carlo.
Scored against the world oracle on the DEV rows: excess KL(P* || Q).

    .venv/bin/python tools/proto_latent.py --inst world_bank --scenario base --dims 0,1,2,4
"""

import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from make_sandbox import load_config, load_schema  # noqa: E402
from score import floored, load_frames, sample_rows  # noqa: E402

spec = importlib.util.spec_from_file_location("v4", os.path.join(ROOT, "sub", "v4", "main.py"))
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
DEV = v4.DEVICE


class Latent(torch.nn.Module):
    def __init__(self, dx, J, Km, D, H=256, drop=0.3):
        super().__init__()
        self.J, self.Km, self.D = J, Km, D
        self.trunk = torch.nn.Sequential(torch.nn.Linear(dx, H), torch.nn.GELU(),
                                         torch.nn.Dropout(drop))
        self.eta = torch.nn.Linear(H, J * Km)
        if D:
            self.mu = torch.nn.Linear(H, D)
            self.lam = torch.nn.Parameter(0.1 * torch.randn(J, Km, D))
            self.enc = torch.nn.Sequential(torch.nn.Linear(dx + J * Km, H), torch.nn.GELU(),
                                           torch.nn.Linear(H, 2 * D))

    def logits(self, h, z):              # h: [B, H], z: [S, B, D] -> [S, B, J, Km]
        eta = self.eta(h).view(1, -1, self.J, self.Km)
        if not self.D:
            return eta
        return eta + torch.einsum("sbd,jkd->sbjk", z, self.lam)


def logp_y(logits, Y, mask):
    """sum_j log p(y_j) per sample and row: [S, B]."""
    lp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
    ok = (Y >= 0)
    pk = lp.gather(-1, Y.clamp(min=0)[None, ..., None].expand(lp.shape[0], -1, -1, 1)).squeeze(-1)
    return (pk * ok).sum(-1)


def fit_predict(ins, tr, rows, D, epochs, k_iw=8, S_pred=256, eval_rows=None, eval_y=None):
    torch.manual_seed(0)
    J, Km = len(ins.targets), ins.Kmax
    X = torch.tensor(ins.X, device=DEV)
    Y = torch.tensor(ins.Y, device=DEV)
    mask = v4._heads(ins)
    Yoh = torch.zeros(len(Y), J, Km, device=DEV)
    ok = Y >= 0
    Yoh[ok] = torch.nn.functional.one_hot(Y[ok], Km).float()
    Yoh = Yoh.view(len(Y), -1)
    m = Latent(X.shape[1], J, Km, D).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    tr_t = torch.tensor(tr, device=DEV)
    out = {}

    def marginal(idx):
        m.eval()
        with torch.no_grad():
            h = m.trunk(X[idx])
            if not D:
                return torch.softmax(m.logits(h, None)[0].masked_fill(~mask, -1e9), -1)
            acc = 0
            for _ in range(S_pred // 64):
                z = m.mu(h)[None] + torch.randn(64, len(idx), D, device=DEV)
                acc = acc + torch.softmax(m.logits(h, z).masked_fill(~mask, -1e9), -1).mean(0)
            return acc / (S_pred // 64)

    for ep in range(1, epochs + 1):
        m.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            h = m.trunk(X[idx])
            if D:
                e = m.enc(torch.cat([X[idx], Yoh[idx]], 1))
                qm, qs = e[:, :D], torch.nn.functional.softplus(e[:, D:]) + 1e-3
                z = qm[None] + qs[None] * torch.randn(k_iw, len(idx), D, device=DEV)
                pm = m.mu(h)[None]
                lpz = -0.5 * ((z - pm) ** 2).sum(-1)
                lqz = -0.5 * (((z - qm[None]) / qs[None]) ** 2).sum(-1) - torch.log(qs).sum(-1)[None]
                lw = logp_y(m.logits(h, z), Y[idx], mask) + lpz - lqz
                loss = -(torch.logsumexp(lw, 0) - np.log(k_iw)).sum() / ok[idx].sum()
            else:
                loss = -logp_y(m.logits(h, None), Y[idx], mask).sum() / ok[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0:
            P = marginal(torch.tensor(rows, device=DEV)).cpu().numpy()
            out[ep] = [P[:, j, :k] for j, k in enumerate(ins.K)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="world_bank")
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--dims", default="0,1,2,4")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    config = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    floor = config["scoring"]["floor"]
    schema = load_schema(os.path.join(ROOT, "raw", "repo", "data", a.inst + ".json"), config)
    d = os.path.join(ROOT, "data", "worlds", a.scenario, a.inst)
    masked, cells, _ = sample_rows(schema, load_frames(d, schema), 1)
    ins = v4.Instrument(masked.reset_index(drop=True), json.loads(json.dumps(schema)))
    O = np.load(os.path.join(d, "oracle.npz"))
    orc = [O["DEV::" + t] for t in ins.targets]

    def kl(tabs):
        tot = 0.0
        for j, P in enumerate(tabs):
            p = floor + (1 - ins.K[j] * floor) * orc[j]
            q = floor + (1 - ins.K[j] * floor) * (P / P.sum(1, keepdims=True))
            tot += (p * (np.log(p) - np.log(q))).sum()
        return tot / (len(ins.hid) * len(ins.targets))

    for D in [int(x) for x in a.dims.split(",")]:
        t = time.time()
        ck = fit_predict(ins, ins.vis, ins.hid, D, a.epochs)
        res = {ep: kl(tabs) for ep, tabs in ck.items()}
        best = min(res, key=res.get)
        print("D=%d  best ep %d  excess KL %.4f   curve %s  %.0fs" % (
            D, best, res[best], " ".join("%.3f" % res[e] for e in sorted(res)), time.time() - t),
            flush=True)


if __name__ == "__main__":
    main()
