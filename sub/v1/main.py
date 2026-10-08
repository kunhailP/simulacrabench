"""SimulacraBench submission v1.

Every scored cell is graded on its own, so the target is the per-item
conditional distribution P(y_j | GIVEN). Real answers never leave the grader,
so everything that depends on the data -- smoothing strength, model choice,
blend weights -- is chosen inside predict() by K-fold cross-validation on the
visible rows.

Base models, each returning a full probability table per item:
  M0  smoothed crowd marginal
  M1  Dirichlet backoff over the most informative GIVEN columns
  M2  multi-head multinomial logistic regression (L2 tuned by CV)
  M3  multi-task MLP sharing one hidden layer across all items
Blend: per-item convex weights fitted on out-of-fold log loss, shrunk toward
the instrument-wide weights.
"""

import time

import numpy as np
import pandas as pd

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

BACKOFF_GRID = [(s, a) for s in (1, 2, 3) for a in (2.0, 8.0, 32.0)]
L2_GRID = (1e-4, 1e-3, 1e-2)
MLP_HIDDEN = 256
MLP_EPOCHS = 60
MLP_EVAL_EVERY = 5
BLEND_SHRINK = 200.0   # pseudo-cells pulling item weights to global weights


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

        tgt = frame[self.targets]
        hidden = tgt.isna().any(axis=1).to_numpy()
        self.vis = np.flatnonzero(~hidden)
        self.hid = np.flatnonzero(hidden)

        # GIVEN codes; an unexpected value gets its own extra level.
        self.gcodes, self.gcard = {}, {}
        for g in self.given:
            lut = {v: i for i, v in enumerate(self.opts[g])}
            col = frame[g].to_numpy(dtype=object)
            c = np.array([lut.get(v, len(lut)) for v in col], dtype=np.int64)
            self.gcodes[g], self.gcard[g] = c, len(lut) + 1

        # Targets as indices into opts; -1 = held out / unknown.
        Y = np.full((len(frame), len(self.targets)), -1, dtype=np.int64)
        for j, t in enumerate(self.targets):
            lut = {v: i for i, v in enumerate(self.opts[t])}
            col = frame[t].to_numpy(dtype=object)
            Y[:, j] = [lut.get(v, -1) if not _isna(v) else -1 for v in col]
        self.Y = Y

        blocks = [np.eye(self.gcard[g], dtype=np.float32)[self.gcodes[g]]
                  for g in self.given]
        self.X = (np.concatenate(blocks, 1) if blocks
                  else np.ones((len(frame), 1), np.float32))
        self.X = self.X[:, self.X[self.vis].std(0) > 0] if len(self.vis) else self.X
        if self.X.shape[1] == 0:
            self.X = np.ones((len(frame), 1), np.float32)


def _isna(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


# ------------------------------------------------------------------ models --

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


def m_backoff(ins, tr, rows, strata, alpha):
    """List over configs of list over items of [len(rows), K] tables."""
    configs = {cfg: [] for cfg in BACKOFF_GRID} if strata is None else {(strata, alpha): []}
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        yk, K = y[ok], ins.K[j]
        ranked = sorted(ins.given, reverse=True, key=lambda g: _mi(
            ins.gcodes[g][tr][ok], yk, ins.gcard[g], K))
        base = np.bincount(yk, minlength=K).astype(float) + 0.5
        base = np.tile(base / base.sum(), (len(rows), 1))
        for (s, a) in configs:
            run = base.copy()
            kf = np.zeros(len(yk), np.int64)
            ka = np.zeros(len(rows), np.int64)
            size = 1
            for g in ranked[:s]:
                kf = kf * ins.gcard[g] + ins.gcodes[g][tr][ok]
                ka = ka * ins.gcard[g] + ins.gcodes[g][rows]
                size *= ins.gcard[g]
                if size > 2_000_000:
                    break
                tab = np.bincount(kf * K + yk, minlength=size * K).reshape(size, K)
                cnt = tab[ka].astype(float)
                run = (cnt + a * run) / (cnt.sum(1, keepdims=True) + a)
            configs[(s, a)].append(run)
    return configs


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
        z = (Xt @ W + b).view(-1, J, Km)
        loss = _nll(z, Yt, mask) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return _probs((X[rows] @ W + b).view(-1, J, Km), mask, ins)


def m_mlp(ins, tr, rows, epochs, eval_rows=None):
    """Returns predictions for `rows`; with eval_rows also per-checkpoint
    predictions so CV can pick the epoch count."""
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
    bs = 256
    ckpts = {}
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = _nll(net(X[idx]).view(-1, J, Km), Y[idx], mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if eval_rows is not None and ep % MLP_EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                ckpts[ep] = _probs(net(X[eval_rows]).view(-1, J, Km), mask, ins)
    net.eval()
    with torch.no_grad():
        final = _probs(net(X[rows]).view(-1, J, Km), mask, ins)
    return final, ckpts


# ------------------------------------------------------------------- blend --

def _mix(p, K):
    return FLOOR + (1 - K * FLOOR) * p


def _cell_logp(tables, ins, rows):
    """[n_cells_per_item] log-probs per item for observed targets."""
    out = []
    for j, P in enumerate(tables):
        y = ins.Y[rows, j]
        ok = y >= 0
        p = _mix(P[ok], ins.K[j])
        out.append(np.log(p[np.arange(ok.sum()), y[ok]]))
    return out


def _fit_weights(P_list, y, K, prior, shrink, steps=200):
    """Convex weights over models minimising log loss (+ pull to prior)."""
    M = len(P_list)
    stack = np.stack([_mix(P[np.arange(len(y)), y], K) for P in P_list], 1)  # n x M
    w = prior.copy()
    for _ in range(steps):   # exponentiated gradient
        mix = stack @ w
        grad = -(stack / mix[:, None]).sum(0) - shrink * (prior / np.maximum(w, 1e-12))
        w = w * np.exp(-0.5 * grad / max(len(y) + shrink, 1))
        w = np.maximum(w, 1e-9)
        w /= w.sum()
    return w


# ---------------------------------------------------------------- predict --

def _folds(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def predict(frame, schema):
    t0 = time.time()
    frame = frame.reset_index(drop=True)
    ins = Instrument(frame, schema)
    vis, hid = ins.vis, ins.hid

    if len(vis) < MIN_VISIBLE or torch is None:
        tables = m_marginal(ins, vis, hid)
        return _emit(tables, ins)

    # ---- out-of-fold predictions on visible rows -------------------------
    folds = _folds(len(vis), N_FOLDS, SEED)
    oof = {}
    J = len(ins.targets)

    def put(name, part, tables):
        if name not in oof:
            oof[name] = [np.zeros((len(vis), ins.K[j])) for j in range(J)]
        for j in range(J):
            oof[name][j][part] = tables[j]

    mlp_ck = {}
    for f, part in enumerate(folds):
        tr = vis[np.setdiff1d(np.arange(len(vis)), part)]
        te = vis[part]
        put("marg", part, m_marginal(ins, tr, te))
        for cfg, tabs in m_backoff(ins, tr, te, None, None).items():
            put(("bo",) + cfg, part, tabs)
        for l2 in L2_GRID:
            put(("lin", l2), part, m_linear(ins, tr, te, l2))
        _, ck = m_mlp(ins, tr, te, MLP_EPOCHS, eval_rows=te)
        for ep, tabs in ck.items():
            put(("mlp", ep), part, tabs)

    # ---- pick the best config per family (instrument-wide) ---------------
    def total(name):
        return sum(lp.sum() for lp in _cell_logp(oof[name], ins, vis))

    families = {"marg": ["marg"],
                "bo": [k for k in oof if isinstance(k, tuple) and k[0] == "bo"],
                "lin": [k for k in oof if isinstance(k, tuple) and k[0] == "lin"],
                "mlp": [k for k in oof if isinstance(k, tuple) and k[0] == "mlp"]}
    best = {fam: max(keys, key=total) for fam, keys in families.items()}
    order = ["marg", "bo", "lin", "mlp"]
    chosen = [best[f] for f in order]

    # ---- blend weights --------------------------------------------------
    M = len(chosen)
    yv = ins.Y[vis]
    glob = np.full(M, 1.0 / M)
    # instrument-wide weights: pool all cells
    pooled_P, pooled_y, pooled_K = [], [], []
    W = np.zeros((J, M))
    allstack = []
    for j in range(J):
        ok = yv[:, j] >= 0
        st = np.stack([_mix(oof[c][j][ok][np.arange(ok.sum()), yv[ok, j]], ins.K[j])
                       for c in chosen], 1)
        allstack.append(st)
    S = np.concatenate(allstack, 0)
    w = glob.copy()
    for _ in range(300):
        mix = S @ w
        grad = -(S / mix[:, None]).mean(0)
        w = np.maximum(w * np.exp(-0.5 * grad), 1e-9)
        w /= w.sum()
    glob = w
    for j in range(J):
        ok = yv[:, j] >= 0
        if ok.sum() == 0:
            W[j] = glob
            continue
        W[j] = _fit_weights([oof[c][j][ok] for c in chosen], yv[ok, j],
                            ins.K[j], glob, BLEND_SHRINK)

    # ---- refit on all visible rows, predict hidden ------------------------
    final = {}
    final["marg"] = m_marginal(ins, vis, hid)
    s, a = best["bo"][1], best["bo"][2]
    final[best["bo"]] = m_backoff(ins, vis, hid, s, a)[(s, a)]
    final[best["lin"]] = m_linear(ins, vis, hid, best["lin"][1])
    final[best["mlp"]], _ = m_mlp(ins, vis, hid, best["mlp"][1])

    tables = []
    for j in range(J):
        P = sum(W[j, m] * final[c][j] for m, c in enumerate(chosen))
        tables.append(P / P.sum(1, keepdims=True))

    predict.last_info = {"best": best, "glob_w": glob.tolist(),
                         "seconds": time.time() - t0}
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
