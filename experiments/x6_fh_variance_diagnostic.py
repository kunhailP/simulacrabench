# Diagnostic behind x5: direct-estimate W^2 vs sampling variance per weight.
# Run from the repo root: .venv/bin/python experiments/x6_fh_variance_diagnostic.py <args>  (see docs/EXPERIMENTS.md)
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


W_ = World(schema, 7, **SCENARIOS[sc])
pairs = list(itertools.combinations(ins.given, 2))

def direct(lp):
    """Pilot (direct) estimates and their sampling variances per weight."""
    lam = torch.full((B, J), lp, device=DEV)
    W, b = fit(lam)
    with torch.no_grad():
        pr = torch.softmax((torch.einsum("nf,fjk->njk", X[vis], W) + b).masked_fill(~mask, -1e9), -1)
        okm = (Y[vis] >= 0).float()[..., None]
        H = torch.einsum("nf,njk->fjk", X[vis] ** 2, pr * (1 - pr) * okm)
        H = H + 1e-4
        Wd = W * (H + 2 * lp) / H        # undo the ridge: direct estimate
        v = 1.0 / H                      # its sampling variance
    return W, b, Wd, v

def fh_mix(W, v, share=True, iters=50):
    """Two-groups scale mixture per (pair block, item), Gaussian FH marginal:
    W_f ~ N(0, tau_c^2 + v_f) for every weight f in the block, c in {S, L}.
    pi per pair shared across items (share=True) or global."""
    m = mask[None].float()
    Wp, vp = W[blk >= nmain], v[blk >= nmain]
    bp = blk[blk >= nmain] - nmain
    P = B - nmain
    tS, tL, pi = 1e-3, 0.3, torch.full((P, 1), 0.1, device=DEV)
    for _ in range(iters):
        def ll(t):
            q = -0.5 * (torch.log(t + vp) + Wp ** 2 / (t + vp)) * m
            return torch.zeros(P, J, device=DEV).index_add_(0, bp, q.sum(-1))
        lS, lL = ll(tS), ll(tL)
        r = torch.sigmoid(lL - lS + torch.log(pi) - torch.log1p(-pi))
        pi = (r.mean(1, keepdim=True) if share else r.mean().expand(P, 1)).clamp(1e-4, 1 - 1e-4)
        # tau updates: moment matching (FH-style, truncated at a floor)
        rf = r[bp]                                        # F x J
        num = ((Wp ** 2 - vp) * m).sum(-1)                 # F x J
        den = m.sum(-1).expand_as(num)
        tL = ((rf * num).sum() / (rf * den).sum()).clamp(min=1e-3)
        tS = (((1 - rf) * num).sum() / ((1 - rf) * den).sum()).clamp(min=1e-5, max=float(tL) * 0.5)
    return r, float(tS), float(tL), pi

truth = np.zeros((B - nmain, J), bool)
pidx = {pr: nmain + i for i, pr in enumerate(pairs)}
for j, t in enumerate(ins.targets):
    for ga, gb, _ in W_.par[t]["inter"]:
        truth[pidx[tuple(sorted((ga, gb), key=ins.given.index))] - nmain, j] = True

for lp in (0.03, 0.3):
    W0, b0, Wd, v = direct(lp)
    m = mask[None].float()
    sel = blk >= nmain
    bp = (blk[sel] - nmain).cpu().numpy()
    w2 = ((Wd[sel] ** 2) * m).sum(-1).cpu().numpy(); vv = (v[sel] * m).sum(-1).cpu().numpy()
    w0 = ((W0[sel] ** 2) * m).sum(-1).cpu().numpy()
    act = truth[bp]          # F x J
    for name, a in (("active", act), ("inactive", ~act)):
        print("lp %.2f %-8s  mean W^2 %.4g  mean Wd^2 %.4g  mean v %.4g  median v %.4g" % (
            lp, name, w0[a].mean(), w2[a].mean(), vv[a].mean(), np.median(vv[a])))
