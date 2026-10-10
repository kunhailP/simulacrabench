"""Structural-event pooling (SEP): one propensity per shared event, not one per item.

In the two instruments we target, a single respondent-level event sets the
missing-type outcome of many items at once: the hidden s3_11 branch (UNHCR, 73
children, gate agreement 0.29-0.98) and attrition in a single-session
assessment (World Bank, "Not answered" on 69 of 76 items). A per-item model
estimates the probability of that event from scratch for every item. SEP
estimates it once per family and lets each item deviate from it under shrinkage:

  1. s_ij = 1 if item j of respondent i sits on its structural level (the gate
     sentinel, else "Not answered"). Families: items whose indicators move
     together on the training rows (average-linkage clustering of the
     indicator correlations).
  2. z_i = mean of s_ij over a family. g(x) = E[z | GIVEN] from TabICL on binned
     z, cross-fitted inside the training rows so step 3 sees honest values.
  3. Per item: logit pi_j(x) = a_j + b_j logit g(x) + ridge-shrunk GIVEN main
     effects (one-hot), i.e. item deviations shrunk toward the family event.
  4. Content: the per-item TabICL table with the structural level removed and
     renormalised. Output: pi_j on the structural level, (1 - pi_j) * content.

Items with no structural level or no family keep plain TabICL.
"""

import numpy as np
import torch

from candidates import _gmat, _tab_fit_predict  # noqa: F401

MIN_RATE, MAX_RATE = 0.005, 0.995
CORR_CUT = 0.35          # average-linkage cut on indicator correlation
MIN_FAMILY = 3
Z_BINS = 8
INNER = 3
RIDGE = 3.0


def structural_level(ins, j):
    t = ins.targets[j]
    opts = ins.opts[t]
    if ins.schema_items[t].get("gate"):
        return len(opts) - 1                       # the sentinel slot is last
    for k, v in enumerate(opts):
        if v == "Not answered":
            return k
    return None


def families(S, ok):
    """Average-linkage clustering of the indicator columns on correlation."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    cols = [j for j in range(S.shape[1]) if ok[j]]
    if len(cols) < MIN_FAMILY:
        return []
    C = np.corrcoef(S[:, cols].T)
    C = np.nan_to_num(C, nan=0.0)
    D = np.clip(1 - C, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Z, 1 - CORR_CUT, "distance")
    fams = []
    for c in np.unique(lab):
        f = [cols[i] for i in np.where(lab == c)[0]]
        if len(f) >= MIN_FAMILY:
            fams.append(f)
    return fams


def _onehot(G):
    cols = []
    for c in range(G.shape[1]):
        lv = np.unique(G[:, c])
        if len(lv) > 1:
            cols.append((G[:, c][:, None] == lv[None, 1:]).astype(np.float64))
    return np.concatenate(cols, 1) if cols else np.zeros((len(G), 0))


def _g_fit_predict(Xdf, z_tr, tr_idx, pred_idx):
    """E[z|x] by TabICL classification on binned z."""
    q = np.unique(np.quantile(z_tr, np.linspace(0, 1, Z_BINS + 1)[1:-1]))
    b = np.searchsorted(q, z_tr, side="right")
    lv = np.unique(b)
    means = np.array([z_tr[b == k].mean() for k in lv])
    b = np.searchsorted(lv, b)
    P = _tab_fit_predict(Xdf.iloc[tr_idx], b, Xdf.iloc[pred_idx], len(lv))
    return np.clip(P @ means, 1e-4, 1 - 1e-4)


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _ridge_logit(F, s, pen, iters=200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = torch.tensor(F, device=dev)
    s = torch.tensor(s, dtype=torch.float64, device=dev)
    pen = torch.tensor(pen, device=dev)
    w = torch.zeros(F.shape[1], dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = F @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, s, reduction="sum")
        loss = loss + 0.5 * (pen * w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().cpu().numpy()


def m_sep(ins, tr, rows, plain=None):
    import pandas as pd
    G = _gmat(ins)
    Xdf = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    J = len(ins.targets)
    if plain is None:
        plain = []
        for j in range(J):
            y = ins.Y[tr, j]
            ok = y >= 0
            plain.append(_tab_fit_predict(Xdf.iloc[tr[ok]], y[ok], Xdf.iloc[rows], ins.K[j]))
    lev = [structural_level(ins, j) for j in range(J)]
    Ytr = ins.Y[tr]
    S = np.zeros((len(tr), J))
    okc = np.zeros(J, bool)
    for j in range(J):
        if lev[j] is None:
            continue
        S[:, j] = Ytr[:, j] == lev[j]
        r = S[:, j].mean()
        okc[j] = MIN_RATE < r < MAX_RATE
    fams = families(S, okc)
    out = list(plain)
    OH = _onehot(G)
    rng = np.random.default_rng(0)
    m_sep.last = {"families": [len(f) for f in fams]}
    for f in fams:
        z = S[:, f].mean(1)
        # honest g on the training rows (inner cross-fit), full fit for `rows`
        g_tr = np.zeros(len(tr))
        perm = rng.permutation(len(tr))
        for k in range(INNER):
            part = perm[k::INNER]
            rest = np.setdiff1d(perm, part)
            g_tr[part] = _g_fit_predict(Xdf, z[rest], tr[rest], tr[part])
        g_rw = _g_fit_predict(Xdf, z, tr, rows)
        Ftr = np.concatenate([np.ones((len(tr), 1)), _logit(g_tr)[:, None], OH[tr]], 1)
        Frw = np.concatenate([np.ones((len(rows), 1)), _logit(g_rw)[:, None], OH[rows]], 1)
        pen = np.r_[1e-6, 1e-6, np.full(OH.shape[1], RIDGE)]
        for j in f:
            w = _ridge_logit(Ftr, S[:, j], pen)
            pi = 1 / (1 + np.exp(-(Frw @ w)))
            pi = np.clip(pi, 1e-5, 1 - 1e-5)
            c = plain[j].copy()
            c[:, lev[j]] = 0
            c /= np.maximum(c.sum(1, keepdims=True), 1e-12)
            c *= (1 - pi)[:, None]
            c[:, lev[j]] = pi
            out[j] = c
    return out


# ---------------------------------------------------------------------------
# SEP2: the shared event as a covariate, not a replacement.
# SEP (above) replaced TabICL's structural-level probability with a ridge GLM on
# GIVEN main effects + logit g(x). On gss_wb it gained on attrition cells but
# lost on gates that GIVEN combinations decide exactly, which TabICL represents
# and the GLM does not. SEP2 keeps TabICL as the item model and hands it
# logit g_f(x) for every family f as extra numeric columns, so each item uses
# the pooled event signal as much as it helps. Families are formed per state
# type (gate sentinel, "Not answered") so attrition inside gated items counts.


def _states(ins, j):
    t = ins.targets[j]
    opts = ins.opts[t]
    out = {}
    if ins.schema_items[t].get("gate"):
        out["gate"] = len(opts) - 1
    for k, v in enumerate(opts):
        if v == "Not answered":
            out["na"] = k
    return out


def family_scores(ins, tr, rows, Xdf, rng):
    """Honest (inner cross-fitted) logit g_f on tr, full-fit on rows, per family."""
    J = len(ins.targets)
    Ytr = ins.Y[tr]
    fams = []
    for kind in ("gate", "na"):
        S = np.zeros((len(tr), J))
        okc = np.zeros(J, bool)
        for j in range(J):
            k = _states(ins, j).get(kind)
            if k is None:
                continue
            S[:, j] = Ytr[:, j] == k
            okc[j] = MIN_RATE < S[:, j].mean() < MAX_RATE
        fams += [(kind, f, S[:, f].mean(1)) for f in families(S, okc)]
    G_tr = np.zeros((len(tr), len(fams)))
    G_rw = np.zeros((len(rows), len(fams)))
    perm = rng.permutation(len(tr))
    for c, (_, f, z) in enumerate(fams):
        for k in range(INNER):
            part = perm[k::INNER]
            rest = np.setdiff1d(perm, part)
            G_tr[part, c] = _logit(_g_fit_predict(Xdf, z[rest], tr[rest], tr[part]))
        G_rw[:, c] = _logit(_g_fit_predict(Xdf, z, tr, rows))
    return fams, G_tr, G_rw


def m_sep2(ins, tr, rows):
    import pandas as pd
    G = _gmat(ins)
    Xdf = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    rng = np.random.default_rng(0)
    fams, G_tr, G_rw = family_scores(ins, tr, rows, Xdf, rng)
    m_sep2.last = {"families": [(k, len(f)) for k, f, _ in fams]}
    Xtr, Xrw = Xdf.iloc[tr].reset_index(drop=True), Xdf.iloc[rows].reset_index(drop=True)
    for c in range(len(fams)):
        Xtr[f"_g{c}"] = G_tr[:, c]
        Xrw[f"_g{c}"] = G_rw[:, c]
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        out.append(_tab_fit_predict(Xtr[ok], y[ok], Xrw, ins.K[j]))
    return out


# ---------------------------------------------------------------------------
# NSP: non-substantive propensity pooling, as a minimal correction of TabICL.
# The loss decomposition (research/decompose.py) shows DK / refused / not-answered
# cells are where per-item estimation is weakest (half-data penalty 0.02-0.09
# nat/cell vs <0.01 elsewhere), and the propensity to give them is a respondent
# trait shared across items. NSP keeps TabICL's table and moves only the
# probability of each non-substantive level along one shared, honestly
# estimated propensity: logit p'_jk = logit p_jk + a_jk + beta_t (o_t(x) - mean),
# beta_t one coefficient per type, a_jk ridge-shrunk to 0; other levels rescale.

import re as _re

NS_TYPES = {"dk": _re.compile(r"don.?t know|\(dk\)|^9?8\.", _re.I),
            "ref": _re.compile(r"refus|prefer not|no response", _re.I),
            "na": _re.compile(r"not answered|no answer|not recorded|missing", _re.I)}
NS_RIDGE = 20.0
NS_INNER = 2


def ns_levels(ins, j):
    out = {}
    for k, v in enumerate(ins.opts[ins.targets[j]]):
        if v == "NA_GATED":
            continue
        for t, rx in NS_TYPES.items():
            if rx.search(v) and t not in out:
                out[t] = k
                break
    return out


def _tab_tables(Xdf, ins, tr, rows, inner, rng):
    """Cross-fitted TabICL tables on tr (inner folds) and full-fit tables for rows."""
    J = len(ins.targets)
    O = [np.zeros((len(tr), ins.K[j])) for j in range(J)]
    perm = rng.permutation(len(tr))
    for k in range(inner):
        part = perm[k::inner]
        rest = perm[np.isin(perm, part, invert=True)]
        for j in range(J):
            y = ins.Y[tr[rest], j]
            ok = y >= 0
            O[j][part] = _tab_fit_predict(Xdf.iloc[tr[rest][ok]], y[ok], Xdf.iloc[tr[part]], ins.K[j])
    H = []
    for j in range(J):
        y = ins.Y[tr, j]
        ok = y >= 0
        H.append(_tab_fit_predict(Xdf.iloc[tr[ok]], y[ok], Xdf.iloc[rows], ins.K[j]))
    return O, H


def _shift(P, k, delta):
    p = np.clip(P[:, k], 1e-6, 1 - 1e-6)
    q = 1 / (1 + np.exp(-(np.log(p) - np.log1p(-p) + delta)))
    Q = P * ((1 - q) / (1 - p))[:, None]
    Q[:, k] = q
    return Q


def m_nsp(ins, tr, rows):
    import pandas as pd
    G = _gmat(ins)
    Xdf = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    rng = np.random.default_rng(0)
    J = len(ins.targets)
    O, H = _tab_tables(Xdf, ins, tr, rows, NS_INNER, rng)
    levs = [ns_levels(ins, j) for j in range(J)]
    Ytr = ins.Y[tr]
    out = list(H)
    m_nsp.last = {}
    for t in NS_TYPES:
        pairs = [(j, levs[j][t]) for j in range(J) if t in levs[j]]
        if len(pairs) < 3:
            continue
        # respondent propensity: share of type-t answers among items offering it
        S = np.stack([(Ytr[:, j] == k) for j, k in pairs], 1).astype(float)
        V = np.stack([(Ytr[:, j] >= 0) & (Ytr[:, j] != ins.K[j] - 1 if ins.schema_items[ins.targets[j]].get("gate") else True)
                      for j, _ in pairs], 1)
        z = (S * V).sum(1) / np.maximum(V.sum(1), 1)
        if z.std() == 0:
            continue
        o_tr = np.zeros(len(tr))
        perm = rng.permutation(len(tr))
        for k in range(INNER):
            part = perm[k::INNER]
            rest = perm[np.isin(perm, part, invert=True)]
            o_tr[part] = _logit(_g_fit_predict(Xdf, z[rest], tr[rest], tr[part]))
        o_rw = _logit(_g_fit_predict(Xdf, z, tr, rows))
        mu = o_tr.mean()
        o_tr, o_rw = o_tr - mu, o_rw - mu
        # pooled fit: beta (shared) + a_j (ridge) on the cross-fitted TabICL logits
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        base, ys, ix = [], [], []
        for c, (j, k) in enumerate(pairs):
            ok = Ytr[:, j] >= 0
            p = np.clip(O[j][ok, k], 1e-6, 1 - 1e-6)
            base.append(np.log(p) - np.log1p(-p))
            ys.append((Ytr[ok, j] == k).astype(float))
            ix.append(np.full(ok.sum(), c))
            o_c = o_tr[ok]
            base[-1] = np.stack([base[-1], o_c], 1)
        B = torch.tensor(np.concatenate(base), device=dev)
        Yb = torch.tensor(np.concatenate(ys), device=dev)
        I = torch.tensor(np.concatenate(ix), device=dev)
        a = torch.zeros(len(pairs), dtype=torch.float64, device=dev, requires_grad=True)
        beta = torch.zeros(1, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([a, beta], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z_ = B[:, 0] + a[I] + beta * B[:, 1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z_, Yb, reduction="sum")
            loss = loss + 0.5 * NS_RIDGE * (a * a).sum()
            loss.backward()
            return loss
        opt.step(closure)
        a_, b_ = a.detach().cpu().numpy(), float(beta.item())
        m_nsp.last[t] = {"items": len(pairs), "beta": round(b_, 3),
                         "a_sd": round(float(a_.std()), 3)}
        for c, (j, k) in enumerate(pairs):
            out[j] = _shift(out[j], k, a_[c] + b_ * o_rw)
    return out
