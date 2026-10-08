# MAP-EM variance components (collapse), block-norm AUC 0.91, and top-k select-then-refit vs oracle set.
# Run from the repo root: .venv/bin/python experiments/x4_ard_topk_selection.py <args>  (see docs/EXPERIMENTS.md)
"""Per-(item, block) variance components for a multinomial logistic model.
MAP with W_bj ~ N(0, tau_bj^2 I); tau updated by the EM fixed point
tau^2 = (||W_bj||^2 + c) / (d_b + c/t0)  (inverse-gamma hyperprior keeps it off 0).
"""
import sys, json, itertools, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "tools")
from proto_latent import v4
from world import World, SCENARIOS
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
def kl(tabs):
    tot = 0.0
    for j, Pj in enumerate(tabs):
        p = floor + (1 - ins.K[j] * floor) * orc[j]
        q = floor + (1 - ins.K[j] * floor) * (Pj / Pj.sum(1, keepdims=True))
        tot += (p * (np.log(p) - np.log(q))).sum()
    return tot / (len(ins.hid) * len(ins.targets))

# design with blocks
vis = ins.vis
mats, blk = [], []
for g in ins.given:
    oh = np.eye(ins.gcard[g], dtype=np.float32)[ins.gcodes[g]]
    oh = oh[:, oh[vis].sum(0) > 0]
    mats.append(oh); blk += [len(mats) - 1] * oh.shape[1]
nmain = len(mats)
for a, b in itertools.combinations(ins.given, 2):
    k = ins.gcodes[a] * ins.gcard[b] + ins.gcodes[b]
    oh = np.eye(ins.gcard[a] * ins.gcard[b], dtype=np.float32)[k]
    oh = oh[:, oh[vis].sum(0) >= 5]
    if oh.shape[1]:
        mats.append(oh); blk += [len(mats) - 1] * oh.shape[1]
X = torch.tensor(np.concatenate(mats, 1), device=DEV)
blk = torch.tensor(blk, device=DEV)
B = len(mats)
J, Km = len(ins.targets), ins.Kmax
Y = torch.tensor(ins.Y, device=DEV); mask = v4._heads(ins)
dims = torch.bincount(blk).float()[:, None] * torch.tensor(ins.K, device=DEV).float()[None]  # B x J
n = len(vis)
print("blocks", B, "features", X.shape[1], "n", n)

def fit(lam, W0=None, b0=None, iters=150):
    W = (W0.clone() if W0 is not None else torch.zeros(X.shape[1], J, Km, device=DEV)).requires_grad_(True)
    b = (b0.clone() if b0 is not None else torch.zeros(J, Km, device=DEV)).requires_grad_(True)
    Xt, Yt = X[vis], Y[vis]
    L = lam[blk]  # F x J
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        logits = torch.einsum("nf,fjk->njk", Xt, W) + b
        loss = v4._nll(logits, Yt, mask) + ((W ** 2).sum(-1) * L).sum() / n
        loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()

def pred(W, b):
    with torch.no_grad():
        return v4._probs(torch.einsum("nf,fjk->njk", X[ins.hid], W) + b, mask, ins)

hyp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
for lam0 in ():
    lam = torch.full((B, J), lam0 * 1.0, device=DEV)   # lambda = 1/(2 tau^2) in per-row units * n
    W, b = fit(lam)
    out = ["%.4f" % kl(pred(W, b))]
    for it in range(6):
        # EM fixed point for tau^2 per (block,item) with IG(c, c*t0) prior around 1/(2 lam0)
        with torch.no_grad():
            pr = torch.softmax((torch.einsum("nf,fjk->njk", X[vis], W) + b).masked_fill(~mask, -1e9), -1)
            okm = (Y[vis] >= 0).float()[..., None]
            H = torch.einsum("nf,njk->fjk", X[vis] ** 2, pr * (1 - pr) * okm)   # diag Hessian of n*NLL
            var = 1.0 / (H + 2 * lam[blk][..., None])
            var = var * mask[None].float()
        sq = torch.zeros(B, J, device=DEV).index_add_(0, blk, (W ** 2 + var).sum(-1))
        t0 = 1 / (2 * lam0); c = hyp * dims.mean()
        tau2 = (sq + c * t0) / (dims + c)
        lam = 1 / (2 * tau2)
        W, b = fit(lam, W, b, 80)
        out.append("%.4f" % kl(pred(W, b)))
    act = (lam[nmain:] < lam0).float().mean().item()
    print("lam0 %.1f  KL by EM iter: %s   pair blocks loosened %.2f" % (lam0, " ".join(out), act), flush=True)

# diagnostic: are true interaction blocks separable by fitted block norm?
W_ = World(schema, 7, **SCENARIOS[sc])
pairs = list(itertools.combinations(ins.given, 2))
lam = torch.full((B, J), 0.3, device=DEV); Wf, bf = fit(lam)
sq = torch.zeros(B, J, device=DEV).index_add_(0, blk, (Wf ** 2).sum(-1)) / dims
sq = sq.cpu().numpy()
act, ina = [], []
pidx = {}
bi = nmain
for a, b2 in pairs:
    pidx[(a, b2)] = bi; bi += 1   # assumes no pair block dropped
for j, t in enumerate(ins.targets):
    truep = {tuple(sorted((ga, gb), key=ins.given.index)) for ga, gb, _ in W_.par[t]["inter"]}
    for pr in pairs:
        (act if pr in truep else ina).append(sq[pidx[pr], j])
act, ina = np.array(act), np.array(ina)
from scipy.stats import mannwhitneyu
print("active blocks %d median %.4g | inactive %d median %.4g | AUC %.3f" % (
    len(act), np.median(act), len(ina), np.median(ina),
    mannwhitneyu(act, ina).statistic / (len(act) * len(ina))))

def run(sel, la, li, lm=0.3):
    lam = torch.full((B, J), lm, device=DEV)
    lam[nmain:] = li
    lam[nmain:][torch.tensor(sel, device=DEV)] = la
    W, b = fit(lam)
    return kl(pred(W, b))

truth = np.zeros((B - nmain, J), bool)
for j, t in enumerate(ins.targets):
    truep = {tuple(sorted((ga, gb), key=ins.given.index)) for ga, gb, _ in W_.par[t]["inter"]}
    for pr in truep:
        truth[pidx[pr] - nmain, j] = True
ps = sq[nmain:]
for lm, la, li in [(0.3, 0.03, 3.0), (0.3, 0.01, 3.0), (0.1, 0.03, 3.0), (0.1, 0.1, 3.0), (0.1, 0.1, 1.0)]:
    row = ["ORACLE %.4f" % run(truth, la, li, lm)]
    for k in (4, 8, 16, 36):
        s_ = np.zeros_like(truth)
        top = np.argsort(-ps, 0)[:k]
        for j in range(J): s_[top[:, j], j] = True
        row.append("top%d %.4f" % (k, run(s_, la, li, lm)))
    print("lm %.1f la %.2f li %.0f  " % (lm, la, li) + "  ".join(row), flush=True)
