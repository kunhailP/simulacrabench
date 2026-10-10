"""SimulacraBench submission v9: v8 + OOF-guarded post-processing (post.py):
country empirical-Bayes recalibration before the temperature and the
sequential-attrition monotone constraint after it. Everything below the v9
note is v8 unchanged.

v8: v7 + TabICLv2, log-linear pooling, budget guard.

Evidence (docs/LAB.md, real GSS microdata in the competition format): every
model family lands within 0.005 skill; TabICLv2 is the only one ahead of the v7
blend (+0.002), and linear blending adds nothing on top of it while log-linear
pooling adds a little. So:

  1. v7 runs first and its tables are the fallback for everything below.
  2. TabICLv2 (public weights, jingang/TabICL @ 4dcd344, bundled) predicts each
     PREDICT item from the GIVEN columns by in-context learning. It is
     cross-fitted on v7's folds: OOF tables for visible rows, the fold average
     for hidden rows.
  3. With time left, an item gated on a PREDICT parent also gets the gate
     factorisation sum_v P(parent=v|x) P(child|parent=v,x), both from TabICL.
  4. Items with TabICL get q ∝ p_v7^a * p_tab^b * p_gate^c, weights fitted on
     OOF log loss over all such items, then v7's per-item shrunk temperature.
  5. TabICL works item by item inside a time allowance; an item it does not
     reach, or one that raises, keeps v7.

The TabICL deadline: after v7, the time left in the phase budget (80% of 900 s
in development, of 3600 s in the final, told apart by how many visible rows the
schema's split implies) is shared with the instruments not yet called in
proportion to their estimated costs; see _tab_deadline.
"""

import os
import sys
import time

T_IMPORT = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import v7core as V  # noqa: E402
import post as PP  # noqa: E402

CKPT = os.path.join(HERE, "weights", "tabicl-classifier-v2-20260212.ckpt")
TAB_N_EST = 1                   # development budget (900 s)
TAB_N_EST_FINAL = 4             # final budget (3600 s): lab +0.0001..0.0004 over 1, all 6 labs
TAB_COST = {1: 1.0, 4: 3.5}     # TabICL time relative to n_estimators=1 (lab timings)
SAFETY = 0.80                   # share of the phase budget we plan to use
# seconds per (PREDICT item x visible row), measured on an RTX 3090 (an H100
# is faster, so these only make the early instruments leave time for the later)
V7_RATE = 1.5e-4
TAB_RATE = 1.8e-4
# (PREDICT items, rows) of the three competition instruments, to reserve time
# for the ones not yet called; anything else is treated as already seen
KNOWN = {12: 19846 - 3473, 76: 6927 - 1212, 213: 10349 - 1449}
_seen = set()

try:
    import pandas as pd
    import torch
    from tabicl import TabICLClassifier
    _TAB_OK = torch.cuda.is_available() and os.path.exists(CKPT)
except Exception:  # noqa: BLE001 -- any import problem means v7 alone
    _TAB_OK = False

_MODEL = {}
_N_EST = {"n": TAB_N_EST}


def _patch_loader():
    """TabICLClassifier.fit reloads the checkpoint on every call; load it once."""
    orig = TabICLClassifier._load_model

    def cached(self):
        if "m" not in _MODEL:
            orig(self)
            _MODEL["m"] = (self.model_, self.model_config_, self.model_path_)
        self.model_, self.model_config_, self.model_path_ = _MODEL["m"]
    TabICLClassifier._load_model = cached


if _TAB_OK:
    try:
        _patch_loader()
    except Exception:  # noqa: BLE001
        _TAB_OK = False


def _budget(schema, n_vis):
    sp = schema.get("split") or {}
    n_tr, n_dev = sp.get("n_train", 0), sp.get("n_dev", 0)
    final = n_dev > 0 and n_vis >= n_tr + n_dev / 2
    return 3600.0 if final else 900.0


def _tab_deadline(schema, ins):
    """Called after v7 has run on this instrument. The time left in the phase
    budget is shared between this instrument's TabICL and the full (v7 +
    TabICL) runs of the instruments not yet called, in proportion to their
    estimated costs; the last instrument gets everything that is left."""
    J, n = len(ins.targets), len(ins.vis)
    _seen.add(J)
    total = float(os.environ.get("V8_BUDGET", 0)) or _budget(schema, n) * SAFETY
    now = time.time()
    left = total - (now - T_IMPORT)
    tr = TAB_RATE * TAB_COST.get(_N_EST["n"], _N_EST["n"])
    unseen = sum((V7_RATE + tr) * j * r for j, r in KNOWN.items() if j not in _seen)
    mine = tr * J * n
    return now + max(left, 0.0) * mine / (mine + unseen)


def _features(ins):
    G = np.stack([ins.gcodes[g] for g in ins.given], 1)
    return pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})


def _fit_predict(Xtr, ytr, Xte, K):
    cl = np.unique(ytr)
    P = np.full((len(Xte), K), 1e-6)
    if len(cl) < 2:
        P[:, cl] = 1.0
    else:
        m = TabICLClassifier(device="cuda", n_estimators=_N_EST["n"], random_state=0,
                             model_path=CKPT, allow_auto_download=False, use_fa3=False)
        m.fit(Xtr, ytr)
        P[:, m.classes_] = m.predict_proba(Xte)
    return P / P.sum(1, keepdims=True)


def _tabicl(ins, X, folds, deadline):
    """Cross-fitted TabICL tables for as many items as the deadline allows."""
    vis, hid = ins.vis, ins.hid
    oof, hidt = {}, {}
    order = np.argsort(-(ins.Y[vis] >= 0).sum(0), kind="stable")
    for j in order:
        if time.time() > deadline:
            break
        t_item = time.time()
        y = ins.Y[vis, j]
        K = ins.K[j]
        try:
            O = np.zeros((len(vis), K))
            H = np.zeros((len(hid), K))
            for part in folds:
                trm = np.ones(len(vis), bool)
                trm[part] = False
                trm &= y >= 0
                rows = np.concatenate([vis[part], hid])
                P = _fit_predict(X.iloc[vis[trm]], y[trm], X.iloc[rows], K)
                O[part] = P[:len(part)]
                H += P[len(part):] / len(folds)
            if np.isfinite(O).all() and np.isfinite(H).all():
                oof[j], hidt[j] = O, H
        except Exception:  # noqa: BLE001 -- this item keeps v7
            pass
        if time.time() + (time.time() - t_item) > deadline:
            break
    return oof, hidt


def _tabgate(ins, X, folds, tab_o, tab_h, deadline):
    """For an item gated on a PREDICT parent: sum_v P(parent=v|x) P(child|parent=v,x),
    the child model being TabICL with the parent's answer as one more column and
    P(parent|x) the parent's cross-fitted TabICL table. Other items: plain TabICL."""
    vis, hid = ins.vis, ins.hid
    tidx = {t: i for i, t in enumerate(ins.targets)}
    go, gh = {}, {}
    todo = []
    for j, t in enumerate(ins.targets):
        g = ins.schema_items[t].get("gate")
        if g and g["parent"] in tidx and j in tab_o and tidx[g["parent"]] in tab_o:
            todo.append((j, tidx[g["parent"]]))
    for j, p in todo:
        if time.time() > deadline:
            break
        t_item = time.time()
        K, Kp = ins.K[j], ins.K[p]
        y, yp = ins.Y[vis, j], ins.Y[vis, p]
        try:
            O = np.zeros((len(vis), K))
            H = np.zeros((len(hid), K))
            for part in folds:
                trm = np.ones(len(vis), bool)
                trm[part] = False
                trm &= (y >= 0) & (yp >= 0)
                Xtr = X.iloc[vis[trm]].copy()
                Xtr["_parent"] = pd.Categorical(yp[trm], categories=range(Kp))
                rows = np.concatenate([vis[part], hid])
                reps = []
                for v in range(Kp):
                    Xv = X.iloc[rows].copy()
                    Xv["_parent"] = pd.Categorical(np.full(len(rows), v), categories=range(Kp))
                    reps.append(Xv)
                C = _fit_predict(Xtr, y[trm], pd.concat(reps, ignore_index=True), K)
                C = C.reshape(Kp, len(rows), K)
                Pp = np.concatenate([tab_o[p][part], tab_h[p]], 0)
                P = np.einsum("rv,vrk->rk", Pp, C)
                P /= P.sum(1, keepdims=True)
                O[part] = P[:len(part)]
                H += P[len(part):] / len(folds)
            if np.isfinite(O).all() and np.isfinite(H).all():
                go[j], gh[j] = O, H
        except Exception:  # noqa: BLE001
            pass
        if time.time() + (time.time() - t_item) > deadline:
            break
    return go, gh


def _pool_weights(ins, items, comps):
    """Global log-linear weights maximising OOF log likelihood.
    comps: list of dicts j -> OOF table, every one covering `items`."""
    yv = ins.Y[ins.vis]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L, Y = [], []
    for j in items:
        ok = yv[:, j] >= 0
        L.append(torch.tensor(np.stack([np.log(np.maximum(c[j][ok], 1e-9)) for c in comps]),
                              device=dev))
        Y.append(torch.tensor(yv[ok, j], device=dev))
    M = len(comps)
    w = torch.full((M,), 1.0 / M, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=100, line_search_fn="strong_wolfe")
    n = sum(len(y) for y in Y)

    def closure():
        opt.zero_grad()
        tot = 0.0
        for lg, y in zip(L, Y):
            z = torch.einsum("m,mnk->nk", w, lg)
            tot = tot - torch.log_softmax(z, -1).gather(1, y[:, None]).sum()
        loss = tot / n
        loss.backward()
        return loss
    opt.step(closure)
    w = w.detach().cpu().numpy()
    if not np.isfinite(w).all():
        w = np.eye(M)[0]
    return w


def _geo(tabs, w):
    z = sum(wi * np.log(np.maximum(T, 1e-9)) for wi, T in zip(w, tabs))
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def predict(frame, schema):
    """Never raises: if anything below fails (a broken CUDA context included),
    the instrument is retried without TabICL, and only then gets the CPU-only
    smoothed marginal instead of no score."""
    errors = []
    for use_tab in (True, False):
        try:
            return _predict(frame, schema, use_tab)
        except Exception as e:  # noqa: BLE001 -- e.g. CUDA out of memory
            errors.append(type(e).__name__)
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    frame = frame.reset_index(drop=True)
    ins = V.Instrument(frame, schema)
    predict.last_info = {"fallback": "marginal", "errors": errors}
    return V._emit(V.m_marginal(ins, ins.vis, ins.hid), ins)


def _predict(frame, schema, use_tab=True):
    t0 = time.time()
    frame = frame.reset_index(drop=True)
    ins = V.Instrument(frame, schema)
    ins.schema_items = schema["items"]
    if len(ins.vis) < V.MIN_VISIBLE or V.torch is None:
        return V._emit(V.m_marginal(ins, ins.vis, ins.hid), ins)

    _N_EST["n"] = int(os.environ.get("V9_N_EST", 0)) or (
        TAB_N_EST_FINAL if _budget(schema, len(ins.vis)) > 900 else TAB_N_EST)
    oof, tables, folds, info = V.fit_tables(ins)
    info["tab_n_est"] = _N_EST["n"]
    info["t_v7"] = round(time.time() - t0, 1)
    deadline = _tab_deadline(schema, ins)

    tab_o, tab_h, g_o, g_h = {}, {}, {}, {}
    if _TAB_OK and use_tab:
        try:
            X = _features(ins)
            # plain TabICL first; the gate factorisation only with what time is left
            tab_o, tab_h = _tabicl(ins, X, folds, deadline)
            g_o, g_h = _tabgate(ins, X, folds, tab_o, tab_h, deadline)
        except Exception:  # noqa: BLE001
            pass
    info["tab_items"] = f"{len(tab_o)}/{len(ins.targets)}"
    info["tabgate_items"] = len(g_o)

    if tab_o:
        try:
            items = sorted(tab_o)
            # items without a factorised table use plain TabICL in that slot
            g_o = {j: g_o.get(j, tab_o[j]) for j in items}
            g_h = {j: g_h.get(j, tab_h[j]) for j in items}
            comps_o = [oof, tab_o, g_o] if info["tabgate_items"] else [oof, tab_o]
            comps_h = [tables, tab_h, g_h] if info["tabgate_items"] else [tables, tab_h]
            w = _pool_weights(ins, items, comps_o)
            info["pool_w"] = np.round(w, 3).tolist()
            for j in items:
                po = _geo([c[j] for c in comps_o], w)
                ph = _geo([c[j] for c in comps_h], w)
                if np.isfinite(po).all() and np.isfinite(ph).all():
                    oof[j], tables[j] = po, ph
        except Exception:  # noqa: BLE001 -- keep v7 tables
            info["pool_w"] = "failed"

    # v9: country EB (guarded) -> temperature -> monotone attrition (guarded)
    try:
        oof, tables, info["country_eb"] = PP.country_eb(ins, oof, tables, folds)
    except Exception as e:  # noqa: BLE001 -- keep v8's tables
        info["country_eb"] = "failed:" + type(e).__name__
    oof_t, out = PP.temper_both(ins, oof, tables)
    try:
        out, info["monotone"] = PP.monotone(ins, oof_t, out)
    except Exception as e:  # noqa: BLE001
        info["monotone"] = "failed:" + type(e).__name__
    info["seconds"] = round(time.time() - t0, 1)
    info["tab_budget"] = round(deadline - t0 - info["t_v7"], 1)
    predict.last_info = info
    return V._emit(out, ins)
