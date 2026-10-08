"""SimulacraBench submission v4 (v2 synthetic + gate factorisation).

Every scored cell is graded on its own, so the target is the per-item
conditional distribution P(y_j | GIVEN). Real answers never leave the grader,
so everything that depends on the data is chosen inside predict() by K-fold
cross-validation on the visible rows.

Base models, each a probability table per item:
  marg  smoothed crowd marginal
  eb    empirical-Bayes Dirichlet-multinomial backoff over the most
        informative GIVEN columns; the shrinkage strength of every level of
        every item is estimated by marginal likelihood (Fay-Herriot in spirit)
  lin   multi-head multinomial logistic regression, L2 tuned by CV
  mlp   multi-task MLP sharing one hidden layer across all items
Fold models are bagged for the hidden rows (no refit). Blend: per-item convex
weights on out-of-fold log loss, shrunk toward instrument-wide weights, then a
per-item temperature, also shrunk.
"""

import time

import numpy as np
from scipy.special import gammaln

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # torch missing: count models only
    torch = None
    DEVICE = "cpu"

FLOOR = 1e-3
N_FOLDS = 5
SEED = 0
MIN_VISIBLE = 50

EB_DEPTH = 4
EB_ALPHAS = np.exp(np.linspace(np.log(0.3), np.log(3000.0), 40))
L2_GRID = (3e-5, 1e-4, 1e-3)
MLP_HIDDEN = 256
MLP_EPOCHS = 60
MLP_EVAL_EVERY = 5
BLEND_SHRINK = 200.0          # pseudo-cells pulling item weights to global
TEMPS = np.linspace(0.7, 1.5, 17)
TEMP_SHRINK = 2.0             # nats of penalty per unit (log T)^2 per 100 cells


# ---------------------------------------------------------------- encoding --

class Instrument:
    def __init__(self, frame, schema):
        items = schema["items"]
        gv = schema["gated_value"]
        self.names = [k for k, r in items.items()
                      if r["class"] in ("GIVEN", "PREDICT")]
        self.given = [k for k in self.names if items[k]["class"] == "GIVEN"]
        self.targets = [k for k in self.names if items[k]["class"] == "PREDICT"]
        self.opts = {k: list(items[k]["values"]) + ([gv] if items[k].get("gate") else [])
                     for k in self.names}
        self.K = np.array([len(self.opts[k]) for k in self.targets])
        self.Kmax = int(self.K.max())

        hidden = frame[self.targets].isna().any(axis=1).to_numpy()
        self.vis = np.flatnonzero(~hidden)
        self.hid = np.flatnonzero(hidden)

        # GIVEN codes; an unexpected value gets its own extra level.
        self.gcodes, self.gcard = {}, {}
        for g in self.given:
            lut = {v: i for i, v in enumerate(self.opts[g])}
            col = frame[g].to_numpy(dtype=object)
            self.gcodes[g] = np.array([lut.get(v, len(lut)) for v in col], np.int64)
            self.gcard[g] = len(lut) + 1

        # Targets as indices into opts; -1 = held out / unknown.
        Y = np.full((len(frame), len(self.targets)), -1, dtype=np.int64)
        for j, t in enumerate(self.targets):
            lut = {v: i for i, v in enumerate(self.opts[t])}
            col = frame[t].to_numpy(dtype=object)
            Y[:, j] = [lut.get(v, -1) if not _isna(v) else -1 for v in col]
        self.Y = Y

        blocks = [np.eye(self.gcard[g], dtype=np.float32)[self.gcodes[g]]
                  for g in self.given]
        X = (np.concatenate(blocks, 1) if blocks
             else np.ones((len(frame), 1), np.float32))
        if len(self.vis):
            X = X[:, X[self.vis].std(0) > 0]
        self.X = X if X.shape[1] else np.ones((len(frame), 1), np.float32)


def _isna(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


# ------------------------------------------------------------ count models --

def m_marginal(ins, tr, rows):
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        c = np.bincount(y[y >= 0], minlength=ins.K[j]).astype(float) + 0.5
        out.append(np.tile(c / c.sum(), (len(rows), 1)))
    return out


def _mi(x, y, nx, ny):
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    joint /= max(joint.sum(), 1)
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _eb_alpha(counts, base):
    """Dirichlet-multinomial marginal likelihood of cell counts around the
    parent-level means `base`, maximised over the precision alpha.

    counts, base: [cells, K] for the occupied cells. Large alpha means the
    cells look like their parent (shrink hard), small alpha means real
    between-cell variation (trust the cell).
    """
    N = counts.sum(1)
    a = EB_ALPHAS[:, None, None]                       # A x 1 x 1
    am = a * np.maximum(base[None], 1e-9)              # A x C x K
    ll = (gammaln(a[:, :, 0]) - gammaln(a[:, :, 0] + N[None])
          + (gammaln(am + counts[None]) - gammaln(am)).sum(-1)).sum(1)
    return float(EB_ALPHAS[int(np.argmax(ll))])


def m_eb(ins, tr, rows):
    """Empirical-Bayes backoff: marginal -> +g1 -> +g1 x g2 -> ...,
    each level's precision estimated from the training counts."""
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        yk, K = y[ok], ins.K[j]
        ranked = sorted(ins.given, reverse=True, key=lambda g: _mi(
            ins.gcodes[g][tr][ok], yk, ins.gcard[g], K))
        base = np.bincount(yk, minlength=K).astype(float) + 0.5
        base /= base.sum()
        run_tr = np.tile(base, (len(yk), 1))          # parent probs for train rows
        run = np.tile(base, (len(rows), 1))
        kf = np.zeros(len(yk), np.int64)
        ka = np.zeros(len(rows), np.int64)
        for g in ranked[:EB_DEPTH]:
            kf = kf * ins.gcard[g] + ins.gcodes[g][tr][ok]
            ka = ka * ins.gcard[g] + ins.gcodes[g][rows]
            # compact cell ids over occupied cells
            cells, inv = np.unique(kf, return_inverse=True)
            if len(cells) > 50_000:
                break
            tab = np.zeros((len(cells), K))
            np.add.at(tab, (inv, yk), 1.0)
            # parent mean per cell (all rows in a cell share the parent)
            first = np.zeros(len(cells), np.int64)
            first[inv[::-1]] = np.arange(len(inv))[::-1]
            parent = run_tr[first]
            alpha = _eb_alpha(tab, parent)
            post = (tab + alpha * parent) / (tab.sum(1, keepdims=True) + alpha)
            run_tr = post[inv]
            pos = np.searchsorted(cells, ka)
            pos = np.minimum(pos, len(cells) - 1)
            hit = cells[pos] == ka
            run = np.where(hit[:, None], post[pos], run)   # unseen cell: keep parent
        out.append(run)
    return out


# ------------------------------------------------------------ torch models --

def _heads(ins):
    mask = np.zeros((len(ins.targets), ins.Kmax), bool)
    for j, k in enumerate(ins.K):
        mask[j, :k] = True
    return torch.tensor(mask, device=DEVICE)


def _nll(logits, Y, mask):
    logits = logits.masked_fill(~mask, -1e9)
    logp = torch.log_softmax(logits, -1)
    ok = Y >= 0
    picked = logp.gather(-1, Y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -(picked * ok).sum() / ok.sum().clamp(min=1)


def _probs(logits, mask, ins):
    p = torch.softmax(logits.masked_fill(~mask, -1e9), -1).cpu().numpy()
    return [p[:, j, :k] for j, k in enumerate(ins.K)]


def m_linear(ins, tr, rows, l2):
    torch.manual_seed(SEED)
    J, Km = len(ins.targets), ins.Kmax
    X = torch.tensor(ins.X, device=DEVICE)
    Y = torch.tensor(ins.Y, device=DEVICE)
    mask = _heads(ins)
    W = torch.zeros(X.shape[1], J * Km, device=DEVICE, requires_grad=True)
    b = torch.zeros(J * Km, device=DEVICE, requires_grad=True)
    Xt, Yt = X[tr], Y[tr]
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=150, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _nll((Xt @ W + b).view(-1, J, Km), Yt, mask) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return _probs((X[rows] @ W + b).view(-1, J, Km), mask, ins)


def m_mlp(ins, tr, rows, epochs):
    """Predictions for `rows` at every MLP_EVAL_EVERY-th epoch."""
    torch.manual_seed(SEED)
    J, Km = len(ins.targets), ins.Kmax
    X = torch.tensor(ins.X, device=DEVICE)
    Y = torch.tensor(ins.Y, device=DEVICE)
    mask = _heads(ins)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], MLP_HIDDEN), torch.nn.GELU(),
        torch.nn.Dropout(0.3), torch.nn.Linear(MLP_HIDDEN, J * Km)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    tr_t = torch.tensor(tr, device=DEVICE)
    rows_t = torch.tensor(rows, device=DEVICE)
    ckpts = {}
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = _nll(net(X[idx]).view(-1, J, Km), Y[idx], mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % MLP_EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                ckpts[ep] = _probs(net(X[rows_t]).view(-1, J, Km), mask, ins)
    return ckpts


# ------------------------------------------------------------------- blend --

def _mix(p, K):
    return FLOOR + (1 - K * FLOOR) * p


def _eg(S, prior, shrink, steps=300):
    """Convex weights maximising sum log(S @ w) - shrink * KL(prior || w)."""
    w = prior.copy()
    n = len(S)
    for _ in range(steps):
        mix = S @ w
        grad = (S / mix[:, None]).sum(0) + shrink * prior / np.maximum(w, 1e-12)
        w = np.maximum(w * np.exp(0.5 * grad / (n + shrink)), 1e-12)
        w /= w.sum()
    return w


def _temper(P, T):
    Q = np.power(np.maximum(P, 1e-12), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


# --------------------------------------------------------------- synthetic --

def synthetic(ins):
    """v2 blend, returned as cross-fitted tables.

    Returns (S_vis, S_hid, info): S_vis[j] is [len(vis), K_j], each row
    predicted by models that never saw that row; S_hid[j] is the bag of the
    fold models on the hidden rows.
    """
    vis, hid = ins.vis, ins.hid
    J = len(ins.targets)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(vis))
    folds = [perm[i::N_FOLDS] for i in range(N_FOLDS)]
    nh = len(hid)
    oof, bag = {}, {}

    def put(name, part, tables):
        if name not in oof:
            oof[name] = [np.zeros((len(vis), ins.K[j])) for j in range(J)]
            bag[name] = [np.zeros((nh, ins.K[j])) for j in range(J)]
        nt = len(part)
        for j in range(J):
            oof[name][j][part] = tables[j][:nt]
            bag[name][j] += tables[j][nt:] / N_FOLDS

    for part in folds:
        tr = vis[np.setdiff1d(np.arange(len(vis)), part)]
        rows = np.concatenate([vis[part], hid])
        put("marg", part, m_marginal(ins, tr, rows))
        put("eb", part, m_eb(ins, tr, rows))
        for l2 in L2_GRID:
            put(("lin", l2), part, m_linear(ins, tr, rows, l2))
        for ep, tabs in m_mlp(ins, tr, rows, MLP_EPOCHS).items():
            put(("mlp", ep), part, tabs)

    yv = ins.Y[vis]
    oks = [yv[:, j] >= 0 for j in range(J)]

    def picked(name, j):
        ok = oks[j]
        return _mix(oof[name][j][ok][np.arange(ok.sum()), yv[ok, j]], ins.K[j])

    def total(name):
        return sum(np.log(picked(name, j)).sum() for j in range(J))

    lin_best = max([k for k in oof if isinstance(k, tuple) and k[0] == "lin"], key=total)
    mlp_best = max([k for k in oof if isinstance(k, tuple) and k[0] == "mlp"], key=total)
    chosen = ["marg", "eb", lin_best, mlp_best]
    M = len(chosen)

    stacks = [np.stack([picked(c, j) for c in chosen], 1) for j in range(J)]
    glob = _eg(np.concatenate(stacks, 0), np.full(M, 1.0 / M), 0.0)
    W = np.array([_eg(stacks[j], glob, BLEND_SHRINK) if len(stacks[j]) else glob
                  for j in range(J)])

    S_vis, S_hid = [], []
    for j in range(J):
        P = sum(W[j, m] * oof[c][j] for m, c in enumerate(chosen))
        S_vis.append(P / P.sum(1, keepdims=True))
        P = sum(W[j, m] * bag[c][j] for m, c in enumerate(chosen))
        S_hid.append(P / P.sum(1, keepdims=True))
    return S_vis, S_hid, {"chosen": [str(c) for c in chosen],
                          "glob_w": np.round(glob, 3).tolist()}


# ------------------------------------------------------- gate factorisation --
# For an item whose gate parent is itself a PREDICT item, the parent is seen in
# training rows but hidden for scored ones. The child given the parent is far
# sharper (near-deterministic NA_GATED, routing violations included) than the
# child given GIVEN alone, so
#     P(child | x) = sum_v P(parent = v | x) * P(child | parent = v, x)
# borrows the parent's (shared) estimate instead of re-learning the routing
# for every child. P(child | parent, x) is an empirical-Bayes backoff whose
# first stratum is the parent's answer; P(parent | x) is the cross-fitted
# synthetic, so nothing is fitted twice on the same row.

def _cond_eb(ins, tr, rows, j, p, parent_vals):
    """P(y_j | y_p = parent_vals, GIVEN) for `rows` (parent value supplied)."""
    y = ins.Y[tr, j]
    yp = ins.Y[tr, p]
    ok = (y >= 0) & (yp >= 0)
    yk, K, Kp = y[ok], ins.K[j], ins.K[p]
    base = np.bincount(yk, minlength=K).astype(float) + 0.5
    base /= base.sum()
    # level 1: parent value
    kf = yp[ok].astype(np.int64)
    ka = np.asarray(parent_vals, np.int64)
    run_tr = np.tile(base, (len(yk), 1))
    run = np.tile(base, (len(ka), 1))
    gcols = sorted(ins.given, reverse=True, key=lambda g: _mi(
        ins.gcodes[g][tr][ok], yk, ins.gcard[g], K))[:EB_DEPTH - 1]
    levels = [None] + gcols
    for g in levels:
        if g is not None:
            kf = kf * ins.gcard[g] + ins.gcodes[g][tr][ok]
            ka = ka * ins.gcard[g] + ins.gcodes[g][rows]
        cells, inv = np.unique(kf, return_inverse=True)
        tab = np.zeros((len(cells), K))
        np.add.at(tab, (inv, yk), 1.0)
        first = np.zeros(len(cells), np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        parent = run_tr[first]
        alpha = _eb_alpha(tab, parent)
        post = (tab + alpha * parent) / (tab.sum(1, keepdims=True) + alpha)
        run_tr = post[inv]
        pos = np.minimum(np.searchsorted(cells, ka), len(cells) - 1)
        hit = cells[pos] == ka
        run = np.where(hit[:, None], post[pos], run)
    return run


def gate_factor(ins, S_vis, S_hid, folds):
    """OOF and hidden tables for items gated on a PREDICT parent."""
    vis, hid = ins.vis, ins.hid
    tidx = {t: i for i, t in enumerate(ins.targets)}
    items = ins.schema_items
    out_vis, out_hid = {}, {}
    for j, t in enumerate(ins.targets):
        g = items[t].get("gate")
        if not g or g["parent"] not in tidx:
            continue
        p = tidx[g["parent"]]
        Kp = ins.K[p]
        ov = np.zeros((len(vis), ins.K[j]))
        oh = np.zeros((len(hid), ins.K[j]))
        for part in folds:
            tr = vis[np.setdiff1d(np.arange(len(vis)), part)]
            for v in range(Kp):
                rows = np.concatenate([vis[part], hid])
                cond = _cond_eb(ins, tr, rows, j, p, np.full(len(rows), v))
                w_te = S_vis[p][part, v][:, None]
                w_h = S_hid[p][:, v][:, None]
                ov[part] += w_te * cond[:len(part)]
                oh += w_h * cond[len(part):] / len(folds)
        out_vis[j] = ov / ov.sum(1, keepdims=True)
        out_hid[j] = oh / oh.sum(1, keepdims=True)
    return out_vis, out_hid


# ---------------------------------------------------------------- predict --

def predict(frame, schema):
    t0 = time.time()
    frame = frame.reset_index(drop=True)
    ins = Instrument(frame, schema)
    ins.schema_items = schema["items"]
    vis, hid = ins.vis, ins.hid
    J = len(ins.targets)

    if len(vis) < MIN_VISIBLE or torch is None:
        return _emit(m_marginal(ins, vis, hid), ins)

    S_vis, S_hid, info = synthetic(ins)
    yv = ins.Y[vis]
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(len(vis))
    folds = [perm[i::N_FOLDS] for i in range(N_FOLDS)]
    G_vis, G_hid = gate_factor(ins, S_vis, S_hid, folds)

    def picked(T, j):
        ok = yv[:, j] >= 0
        return _mix(T[ok][np.arange(ok.sum()), yv[ok, j]], ins.K[j])

    # per-item convex weight synthetic vs factorised, shrunk to global
    W = {}
    gain = 0.0
    if G_vis:
        stacks = {j: np.stack([picked(S_vis[j], j), picked(G_vis[j], j)], 1) for j in G_vis}
        glob = _eg(np.concatenate(list(stacks.values()), 0), np.array([0.5, 0.5]), 0.0)
        for j, s in stacks.items():
            W[j] = _eg(s, glob, BLEND_SHRINK)
            gain += np.log(s @ W[j]).sum() - np.log(s[:, 0]).sum()
        info["w_gate"] = round(float(glob[1]), 3)
    oof, tables = [], []
    for j in range(J):
        if j in W:
            oof.append(W[j][0] * S_vis[j] + W[j][1] * G_vis[j])
            P = W[j][0] * S_hid[j] + W[j][1] * G_hid[j]
            tables.append(P / P.sum(1, keepdims=True))
        else:
            oof.append(S_vis[j])
            tables.append(S_hid[j])

    for j in range(J):
        ok = yv[:, j] >= 0
        if ok.sum() == 0:
            continue
        P, y = oof[j][ok], yv[ok, j]
        bestT, bv = 1.0, -np.inf
        for T in TEMPS:
            ll = np.log(_mix(_temper(P, T)[np.arange(len(y)), y], ins.K[j])).sum()
            ll -= TEMP_SHRINK * len(y) / 100.0 * np.log(T) ** 2
            if ll > bv:
                bestT, bv = T, ll
        tables[j] = _temper(tables[j], bestT)

    n_cells = int((yv >= 0).sum())
    info.update({"n_gate_items": len(G_vis),
                 "gate_gain_nats_per_cell": round(float(gain / n_cells), 5),
                 "seconds": round(time.time() - t0, 1)})
    predict.last_info = info
    return _emit(tables, ins)


def _emit(tables, ins):
    """Canonical order: hidden rows top to bottom, targets in schema order."""
    out = []
    for r in range(len(ins.hid)):
        for j in range(len(ins.targets)):
            v = np.asarray(tables[j][r], dtype=float)
            v = np.where(np.isfinite(v) & (v > 0), v, 0) + 1e-12
            out.append((v / v.sum()).tolist())
    return out
