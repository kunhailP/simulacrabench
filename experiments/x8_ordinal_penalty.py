# Adjacent-category difference penalty on ordinal items: -9% ordinal KL in the ordinal world, harmful elsewhere.
# Run from the repo root: .venv/bin/python experiments/x8_ordinal_penalty.py <args>  (see docs/EXPERIMENTS.md)
"""Adjacent-category difference penalty on a multinomial logistic (main effects).
Ordinal items: options sorted by value (world.split_options); penalty on
W[:, j, k_(i+1)] - W[:, j, k_(i)] over substantive options in that order."""
import sys, json, numpy as np, torch
sys.path.insert(0, "tools")
from proto_latent import v4
from world import split_options
from make_sandbox import load_config, load_schema
from score import load_frames, sample_rows
DEV = v4.DEVICE
inst, sc = sys.argv[1], sys.argv[2]
cfg = load_config("raw/repo/config.yml"); floor = cfg["scoring"]["floor"]
schema = load_schema("raw/repo/data/%s.json" % inst, cfg)
d = "data/worlds/%s/%s" % (sc, inst)
masked, cells, _ = sample_rows(schema, load_frames(d, schema), 1)
ins = v4.Instrument(masked.reset_index(drop=True), json.loads(json.dumps(schema)))
O = np.load(d + "/oracle.npz"); orc = [O["DEV::" + t] for t in ins.targets]
J, Km = len(ins.targets), ins.Kmax
ordj = []
pairs_a, pairs_b, pairs_j = [], [], []
for j, t in enumerate(ins.targets):
    sub, nr, ordered = split_options(schema["items"][t]["values"])
    if ordered:
        ordj.append(j)
        for a, b in zip(sub[:-1], sub[1:]):
            pairs_a.append(a); pairs_b.append(b); pairs_j.append(j)
pa, pb, pj = (torch.tensor(x, device=DEV) for x in (pairs_a, pairs_b, pairs_j))
print("ordinal items", len(ordj), "of", J)
def kl(tabs, js):
    tot = 0.0; n = 0
    for j in js:
        Pj = tabs[j]
        p = floor + (1 - ins.K[j] * floor) * orc[j]
        q = floor + (1 - ins.K[j] * floor) * (Pj / Pj.sum(1, keepdims=True))
        tot += (p * (np.log(p) - np.log(q))).sum(); n += len(p)
    return tot / n
X = torch.tensor(ins.X, device=DEV); Y = torch.tensor(ins.Y, device=DEV); mask = v4._heads(ins)
vis = ins.vis; n = len(vis)
def fit(l2, lo):
    W = torch.zeros(X.shape[1], J, Km, device=DEV, requires_grad=True)
    b = torch.zeros(J, Km, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = v4._nll(torch.einsum("nf,fjk->njk", X[vis], W) + b, Y[vis], mask) + l2 * (W ** 2).sum()
        if lo:
            dW = W[:, pj, pb] - W[:, pj, pa]
            loss = loss + lo * (dW ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return v4._probs(torch.einsum("nf,fjk->njk", X[ins.hid], W) + b, mask, ins)
nom = [j for j in range(J) if j not in ordj]
for l2 in (1e-5, 3e-5, 1e-4):
    row = []
    for lo in (0, 1e-5, 1e-4, 1e-3, 1e-2):
        T = fit(l2, lo)
        row.append("lo=%g ord %.4f nom %.4f" % (lo, kl(T, ordj), kl(T, nom)))
    print("l2=%g | " % l2 + " | ".join(row), flush=True)
