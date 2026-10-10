"""Candidate base models outside the v7 family, same signature as v7's:
m(ins, tr, rows) -> {config: [table_j of shape (len(rows), K_j)]}.
"""

import os

import numpy as np

THREADS = 8


def _gmat(ins):
    return np.stack([ins.gcodes[g] for g in ins.given], 1).astype(np.int32)


def _expand(p, classes, K):
    out = np.full((len(p), K), 1e-6)
    out[:, classes] = p
    return out / out.sum(1, keepdims=True)


def m_lgbm(ins, tr, rows, lr=0.05, leaves=15, min_leaf=30, l2=5.0, max_rounds=600):
    """Per-item gradient boosting on the GIVEN codes (native categorical splits).
    Rounds by early stopping on an inner 15% of the training rows."""
    import lightgbm as lgb
    G = _gmat(ins)
    rng = np.random.default_rng(0)
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        Xt, yt = G[tr][ok], y[ok]
        K = ins.K[j]
        classes = np.unique(yt)
        if len(classes) < 2:
            out.append(_expand(np.ones((len(rows), 1)), classes, K))
            continue
        remap = np.searchsorted(classes, yt)
        inner = rng.random(len(yt)) < 0.15
        params = dict(objective="multiclass" if len(classes) > 2 else "binary",
                      learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf,
                      lambda_l2=l2, max_cat_to_onehot=4, cat_smooth=10, cat_l2=10,
                      min_data_per_group=30, feature_fraction=0.9, bagging_fraction=0.9,
                      bagging_freq=1, verbose=-1, num_threads=THREADS, seed=0)
        if len(classes) > 2:
            params["num_class"] = len(classes)
        cat = list(range(G.shape[1]))
        dtr = lgb.Dataset(Xt[~inner], remap[~inner], categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(Xt[inner], remap[inner], categorical_feature=cat, reference=dtr)
        bst = lgb.train(params, dtr, max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        n_it = max(bst.best_iteration, 1)
        # refit on all training rows at the chosen size
        dall = lgb.Dataset(Xt, remap, categorical_feature=cat)
        bst = lgb.train(params, dall, int(n_it * 1.1))
        p = bst.predict(G[rows])
        if len(classes) == 2:
            p = np.stack([1 - p, p], 1)
        out.append(_expand(p, classes, K))
    return out


REGISTRY = {
    "lgbm": lambda ins, tr, rows: {"": m_lgbm(ins, tr, rows)},
}


_TABICL = {}


def _patch_tabicl():
    """TabICLClassifier.fit reloads the 110 MB checkpoint every call; keep one copy."""
    from tabicl import TabICLClassifier
    if getattr(TabICLClassifier, "_cached_loader", False):
        return
    orig = TabICLClassifier._load_model

    def cached(self):
        key = (self.checkpoint_version, str(self.model_path))
        if key not in _TABICL:
            orig(self)
            _TABICL[key] = (self.model_, self.model_config_, self.model_path_)
        self.model_, self.model_config_, self.model_path_ = _TABICL[key]
    TabICLClassifier._load_model = cached
    TabICLClassifier._cached_loader = True


def m_tabicl(ins, tr, rows, n_estimators=4):
    """Per-item TabICLv2 (in-context tabular foundation model, public weights).
    GIVEN columns are passed as pandas categoricals."""
    import pandas as pd
    from tabicl import TabICLClassifier
    _patch_tabicl()
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        yt = y[ok]
        K = ins.K[j]
        classes = np.unique(yt)
        if len(classes) < 2:
            out.append(_expand(np.ones((len(rows), 1)), classes, K))
            continue
        m = TabICLClassifier(device="cuda", n_estimators=n_estimators, random_state=0)
        m.fit(df.iloc[tr[ok]], yt)
        p = m.predict_proba(df.iloc[rows])
        out.append(_expand(p, m.classes_, K))
    return out


REGISTRY["tabicl"] = lambda ins, tr, rows: {"": m_tabicl(ins, tr, rows)}


def _tab_fit_predict(Xtr, ytr, Xte, K, n_est=1):
    from tabicl import TabICLClassifier
    _patch_tabicl()
    cl = np.unique(ytr)
    P = np.full((len(Xte), K), 1e-6)
    if len(cl) < 2:
        P[:, cl] = 1.0
    else:
        m = TabICLClassifier(device="cuda", n_estimators=n_est, random_state=0)
        m.fit(Xtr, ytr)
        P[:, m.classes_] = m.predict_proba(Xte)
    return P / P.sum(1, keepdims=True)


def m_tabgate(ins, tr, rows, n_est=1):
    """TabICL per item; an item gated on a PREDICT parent is factorised along
    its gate, P(child|x) = sum_v P(parent=v|x) P(child|parent=v,x), both from
    TabICL. The child model sees the parent's recorded answer as one more
    categorical column (rows where the parent is unknown are dropped)."""
    import pandas as pd
    G = _gmat(ins)
    base = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    tidx = {t: i for i, t in enumerate(ins.targets)}
    plain = {}
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        plain[j] = _tab_fit_predict(base.iloc[tr[ok]], y[ok], base.iloc[rows], ins.K[j], n_est)
    out = []
    for j, t in enumerate(ins.targets):
        g = ins.schema_items[t].get("gate")
        if not g or g["parent"] not in tidx:
            out.append(plain[j])
            continue
        p = tidx[g["parent"]]
        Kp = ins.K[p]
        y, yp = ins.Y[tr, j], ins.Y[tr, p]
        ok = (y >= 0) & (yp >= 0)
        Xtr = base.iloc[tr[ok]].copy()
        Xtr["_parent"] = pd.Categorical(yp[ok], categories=range(Kp))
        reps = []
        for v in range(Kp):
            Xv = base.iloc[rows].copy()
            Xv["_parent"] = pd.Categorical(np.full(len(rows), v), categories=range(Kp))
            reps.append(Xv)
        Xte = pd.concat(reps, ignore_index=True)
        C = _tab_fit_predict(Xtr, y[ok], Xte, ins.K[j], n_est).reshape(Kp, len(rows), ins.K[j])
        P = np.einsum("rv,vrk->rk", plain[p], C)
        out.append(P / P.sum(1, keepdims=True))
    return out


REGISTRY["tabgate"] = lambda ins, tr, rows: {"": m_tabgate(ins, tr, rows)}
REGISTRY["tabicl1"] = lambda ins, tr, rows: {"": m_tabicl(ins, tr, rows, n_estimators=1)}


TABPFN3_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "weights_lab", "tabpfn-v3-classifier-v3_default.ckpt")


def m_tabpfn3(ins, tr, rows, n_estimators=1):
    """Per-item TabPFN-3 (Prior-Labs/tabpfn_3 @ 24a16a8, v3_default), GIVEN codes
    declared categorical. Same interface as m_tabicl, for a paired comparison."""
    from tabpfn import TabPFNClassifier
    G = _gmat(ins)
    cat = list(range(G.shape[1]))
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        yt = y[ok]
        K = ins.K[j]
        classes = np.unique(yt)
        if len(classes) < 2:
            out.append(_expand(np.ones((len(rows), 1)), classes, K))
            continue
        m = TabPFNClassifier(device="cuda", n_estimators=n_estimators, random_state=0,
                             model_path=TABPFN3_CKPT, categorical_features_indices=cat,
                             ignore_pretraining_limits=True)
        m.fit(G[tr[ok]], yt)
        p = m.predict_proba(G[rows])
        out.append(_expand(p, m.classes_, K))
    return out


REGISTRY["tabpfn3"] = lambda ins, tr, rows: {"": m_tabpfn3(ins, tr, rows)}
REGISTRY["tabpfn3x4"] = lambda ins, tr, rows: {"": m_tabpfn3(ins, tr, rows, n_estimators=4)}


def _sep(ins, tr, rows):
    import structev
    return {"": structev.m_sep(ins, tr, rows)}


REGISTRY["sep"] = _sep


def _sep2(ins, tr, rows):
    import structev
    return {"": structev.m_sep2(ins, tr, rows)}


REGISTRY["sep2"] = _sep2


def _band_value(v):
    """'3' -> 3, '6-10' -> 8, '11+' -> 11, '25-34' -> 29.5; None if not numeric."""
    import re
    s = str(v).split(". ", 1)[-1].strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.fullmatch(r"(\d+)\s*\+", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(?:Under|<)\s*(\d+)", s)
    if m:
        return float(m.group(1)) - 1
    m = re.fullmatch(r"\d+", s)
    return float(s) if m else None


def numeric_given(ins):
    """Numeric versions of GIVEN columns whose levels are counts or bands (most
    levels numeric), plus roster totals: all counts, children, elderly."""
    import pandas as pd
    cols = {}
    for g in ins.given:
        opts = ins.opts[g]
        vals = [_band_value(v) for v in opts]
        if sum(v is not None for v in vals) < max(3, 0.7 * len(opts)):
            continue
        lut = np.array([np.nan if v is None else v for v in vals] + [np.nan])
        cols["_n_" + g] = lut[np.minimum(ins.gcodes[g], len(lut) - 1)]
    df = pd.DataFrame(cols)
    words = ("boys", "girls", "men", "women", "children", "males", "females", "elderly")
    roster = [c for c in df if any(w in c[3:].lower() for w in words)]
    if len(roster) >= 2:
        df["_hh_total"] = df[roster].fillna(0).sum(1)
        kid = [c for c in roster if any(w in c.lower() for w in ("0_5", "6_17", "children", "boys", "girls"))]
        old = [c for c in roster if "60" in c]
        if kid:
            df["_hh_kids"] = df[kid].fillna(0).sum(1)
        if old:
            df["_hh_old"] = df[old].fillna(0).sum(1)
    return df


def m_tabicl_num(ins, tr, rows):
    """TabICL on the categorical GIVEN codes plus numeric_given() columns."""
    import pandas as pd
    G = _gmat(ins)
    base = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    df = pd.concat([base, numeric_given(ins)], axis=1)
    m_tabicl_num.last = list(df.columns[len(ins.given):])
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        out.append(_tab_fit_predict(df.iloc[tr[ok]], y[ok], df.iloc[rows], ins.K[j]))
    return out


REGISTRY["tabicl_num"] = lambda ins, tr, rows: {"": m_tabicl_num(ins, tr, rows)}


def _tabicl_frac(ins, tr, rows, frac):
    """TabICL fitted on a seeded fraction of the training rows (learning curve)."""
    sub = np.sort(np.random.default_rng(1).choice(tr, int(len(tr) * frac), replace=False))
    import pandas as pd
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[sub, j]
        ok = y >= 0
        out.append(_tab_fit_predict(df.iloc[sub[ok]], y[ok], df.iloc[rows], ins.K[j]))
    return out


REGISTRY["tab_half"] = lambda ins, tr, rows: {"": _tabicl_frac(ins, tr, rows, 0.5)}
REGISTRY["tab_quarter"] = lambda ins, tr, rows: {"": _tabicl_frac(ins, tr, rows, 0.25)}


def _nsp(ins, tr, rows):
    import structev
    return {"": structev.m_nsp(ins, tr, rows)}


REGISTRY["nsp"] = _nsp


def _country_eb(ins, tr, rows, cols_names):
    """E7: TabICL, then a country-level (or country x col) empirical-Bayes
    correction: q ∝ m(x) * pi_c / mbar_c with pi_c shrunk toward mbar_c by a
    Dirichlet-multinomial alpha per item (research/ebprior.py, mode mult),
    cross-fitted inside tr so the prior for a training row never saw that row."""
    import ebprior
    cols = [c for c in cols_names if c in ins.given]
    if not cols:
        return m_tabicl(ins, tr, rows, n_estimators=1)
    import pandas as pd
    G = _gmat(ins)
    Xdf = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    import structev
    O, H = structev._tab_tables(Xdf, ins, tr, rows, 2, np.random.default_rng(0))
    # ebprior.correct works on ins.vis / ins.hid; present tr as vis and rows as hid
    vis0, hid0 = ins.vis, ins.hid
    ins.vis, ins.hid = tr, rows
    perm = np.random.default_rng(0).permutation(len(tr))
    folds = [perm[i::2] for i in range(2)]
    try:
        _, h, _ = ebprior.correct(ins, O, H, lambda j: cols, folds, "mult")
    finally:
        ins.vis, ins.hid = vis0, hid0
    return h


REGISTRY["ceb_country"] = lambda ins, tr, rows: {"": _country_eb(ins, tr, rows, ["country"])}
REGISTRY["ceb_country_relig"] = lambda ins, tr, rows: {"": _country_eb(ins, tr, rows, ["country", "relig"])}


# ---------------------------------------------------------------------------
# Category embeddings from all items. TabICL ordinal-encodes categoricals
# (sklearn OrdinalEncoder -> integer codes used as numbers; its own code warns
# above 40 levels). Our codes follow schema order (alphabetical for places), so
# a 46-level governorate becomes an arbitrary number line. Here each level of a
# nominal GIVEN column is described by how its respondents answered EVERY
# PREDICT item (EB-smoothed answer shares, prior strength EMB_PRIOR), and the
# level x answer-share matrix is reduced by SVD:
#   tab_reord: categorical codes renumbered along the first component
#   tab_emb:   original columns kept + EMB_DIM numeric components per column

EMB_PRIOR = 20.0
EMB_DIM = 4
EMB_MIN_LEVELS = 6


def level_embeddings(ins, tr):
    out = {}
    Ytr = ins.Y[tr]
    for g in ins.given:
        codes = ins.gcodes[g][tr]
        L = ins.gcard[g]
        if len(np.unique(codes)) < EMB_MIN_LEVELS:
            continue
        blocks = []
        for j in range(len(ins.targets)):
            y = Ytr[:, j]
            ok = y >= 0
            K = ins.K[j]
            tab = np.zeros((L, K))
            np.add.at(tab, (codes[ok], y[ok]), 1.0)
            overall = (tab.sum(0) + 0.5) / (tab.sum() + 0.5 * K)
            sh = (tab + EMB_PRIOR * overall) / (tab.sum(1, keepdims=True) + EMB_PRIOR)
            blocks.append((sh - overall) / np.sqrt(overall))     # chi-square-like scaling
        M = np.concatenate(blocks, 1)
        w = np.bincount(codes, minlength=L).astype(float)
        Mc = M - (w[:, None] * M).sum(0) / w.sum()
        u, s, _ = np.linalg.svd(np.sqrt(w + 1)[:, None] * Mc, full_matrices=False)
        E = u[:, :EMB_DIM] * s[:EMB_DIM] / np.sqrt(w + 1)[:, None]
        out[g] = E                                               # [levels, EMB_DIM]
    return out


def _emb_frame(ins, tr, mode):
    import pandas as pd
    G = _gmat(ins)
    E = level_embeddings(ins, tr)
    cols = {}
    for i, g in enumerate(ins.given):
        c = G[:, i]
        if g in E and mode == "reord":
            rank = np.argsort(np.argsort(E[g][:, 0]))
            c = rank[c]
        cols[g] = pd.Categorical(c)
    df = pd.DataFrame(cols)
    if mode == "emb":
        for g, Eg in E.items():
            for k in range(Eg.shape[1]):
                df[f"_e_{g}_{k}"] = Eg[ins.gcodes[g], k]
    return df


def m_tab_embed(ins, tr, rows, mode):
    df = _emb_frame(ins, tr, mode)
    m_tab_embed.last = [c for c in df.columns if c.startswith("_e_")] or "reordered"
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        out.append(_tab_fit_predict(df.iloc[tr[ok]], y[ok], df.iloc[rows], ins.K[j]))
    return out


REGISTRY["tab_reord"] = lambda ins, tr, rows: {"": m_tab_embed(ins, tr, rows, "reord")}
REGISTRY["tab_emb"] = lambda ins, tr, rows: {"": m_tab_embed(ins, tr, rows, "emb")}


def _level_profiles(ins, g, rows_idx, Ytr_rows):
    codes = ins.gcodes[g][rows_idx]
    L = ins.gcard[g]
    blocks, wts = [], np.bincount(codes, minlength=L).astype(float)
    for j in range(len(ins.targets)):
        y = Ytr_rows[:, j]
        ok = y >= 0
        K = ins.K[j]
        tab = np.zeros((L, K))
        np.add.at(tab, (codes[ok], y[ok]), 1.0)
        overall = (tab.sum(0) + 0.5) / (tab.sum() + 0.5 * K)
        sh = (tab + EMB_PRIOR * overall) / (tab.sum(1, keepdims=True) + EMB_PRIOR)
        blocks.append((sh - overall) / np.sqrt(overall))
    return np.concatenate(blocks, 1), wts


def m_tab_emb_cf(ins, tr, rows, n_folds=5):
    """tab_emb with cross-fitted embeddings: the SVD axes come from all training
    rows, but a training row's level embedding is computed from the other folds
    only (projected on those axes), so no row sees its own answers in a feature."""
    import pandas as pd
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    Ytr = ins.Y[tr]
    perm = np.random.default_rng(0).permutation(len(tr))
    folds = [perm[i::n_folds] for i in range(n_folds)]
    added = []
    for g in ins.given:
        if len(np.unique(ins.gcodes[g][tr])) < EMB_MIN_LEVELS:
            continue
        M, w = _level_profiles(ins, g, tr, Ytr)
        mu = (w[:, None] * M).sum(0) / w.sum()
        _, s, Vt = np.linalg.svd(np.sqrt(w + 1)[:, None] * (M - mu), full_matrices=False)
        V = Vt[:EMB_DIM].T                                   # item-answer loadings
        emb = np.zeros((len(G), EMB_DIM))
        emb[rows] = ((M - mu) @ V)[ins.gcodes[g][rows]]       # hidden rows: all training rows
        for part in folds:
            rest = np.setdiff1d(np.arange(len(tr)), part)
            Mf, _ = _level_profiles(ins, g, tr[rest], Ytr[rest])
            emb[tr[part]] = ((Mf - mu) @ V)[ins.gcodes[g][tr[part]]]
        for k in range(EMB_DIM):
            df[f"_e_{g}_{k}"] = emb[:, k]
            added.append(f"_e_{g}_{k}")
    m_tab_emb_cf.last = added
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        out.append(_tab_fit_predict(df.iloc[tr[ok]], y[ok], df.iloc[rows], ins.K[j]))
    return out


REGISTRY["tab_emb_cf"] = lambda ins, tr, rows: {"": m_tab_emb_cf(ins, tr, rows)}


def _x_frame(ins, mode, max_onehot=200):
    """Label-free encodings of nominal GIVEN columns (>= EMB_MIN_LEVELS levels):
    onehot: 0/1 columns per level; xemb: SVD of the level x (other GIVEN levels)
    co-occurrence table over ALL rows (GIVEN is visible for everyone)."""
    import pandas as pd
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    big = [i for i, g in enumerate(ins.given) if len(np.unique(G[:, i])) >= EMB_MIN_LEVELS]
    add = {}
    for i in big:
        g = ins.given[i]
        lv = np.unique(G[:, i])
        if mode == "onehot":
            for v in lv[:max_onehot]:
                add[f"_o_{g}_{v}"] = (G[:, i] == v).astype(np.float32)
        else:
            others = [k for k in range(G.shape[1]) if k != i]
            blocks = []
            for k in others:
                tab = np.zeros((ins.gcard[g], ins.gcard[ins.given[k]]))
                np.add.at(tab, (G[:, i], G[:, k]), 1.0)
                overall = (tab.sum(0) + 0.5) / (tab.sum() + 0.5 * tab.shape[1])
                sh = (tab + EMB_PRIOR * overall) / (tab.sum(1, keepdims=True) + EMB_PRIOR)
                blocks.append((sh - overall) / np.sqrt(overall))
            M = np.concatenate(blocks, 1)
            w = np.bincount(G[:, i], minlength=ins.gcard[g]).astype(float)
            Mc = M - (w[:, None] * M).sum(0) / w.sum()
            u, s, _ = np.linalg.svd(np.sqrt(w + 1)[:, None] * Mc, full_matrices=False)
            E = u[:, :EMB_DIM] * s[:EMB_DIM] / np.sqrt(w + 1)[:, None]
            for k in range(E.shape[1]):
                add[f"_x_{g}_{k}"] = E[G[:, i], k]
    return pd.concat([df, pd.DataFrame(add)], axis=1) if add else df


def m_tab_x(ins, tr, rows, mode):
    df = _x_frame(ins, mode)
    m_tab_x.last = df.shape[1]
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        out.append(_tab_fit_predict(df.iloc[tr[ok]], y[ok], df.iloc[rows], ins.K[j]))
    return out


REGISTRY["tab_onehot"] = lambda ins, tr, rows: {"": m_tab_x(ins, tr, rows, "onehot")}
REGISTRY["tab_xemb"] = lambda ins, tr, rows: {"": m_tab_x(ins, tr, rows, "xemb")}


# ---------------------------------------------------------------------------
# Per-item GIVEN selection. Adding redundant columns cost TabICL up to 0.003
# skill (tab_onehot / tab_xemb on gss_branch), so the reverse may help: give each
# item only its top-k GIVEN columns by mutual information on the training rows.

def m_tab_topk(ins, tr, rows, k):
    import pandas as pd
    from oof import V7
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        mi = [V7._mi(ins.gcodes[g][tr][ok], y[ok], ins.gcard[g], ins.K[j]) for g in ins.given]
        cols = [ins.given[i] for i in np.argsort(mi)[::-1][:k]]
        out.append(_tab_fit_predict(df.iloc[tr[ok]][cols], y[ok], df.iloc[rows][cols], ins.K[j]))
    return out


REGISTRY["tab_top4"] = lambda ins, tr, rows: {"": m_tab_topk(ins, tr, rows, 4)}
REGISTRY["tab_top8"] = lambda ins, tr, rows: {"": m_tab_topk(ins, tr, rows, 8)}


# ---------------------------------------------------------------------------
# Variance reduction for the final (3600 s): TabICL fold-bagging and seeds.

def m_tab_bag(ins, tr, rows, n_folds=5, n_est=1, seeds=(0,)):
    """Average of TabICL fits on n_folds subsamples of (1 - 1/n_folds) of tr,
    over seeds; n_folds=1 means one fit on all of tr."""
    import pandas as pd
    from tabicl import TabICLClassifier
    _patch_tabicl()
    G = _gmat(ins)
    df = pd.DataFrame({g: pd.Categorical(G[:, i]) for i, g in enumerate(ins.given)})
    out = []
    for j in range(len(ins.targets)):
        y = ins.Y[tr, j]
        ok = y >= 0
        trj, yj = tr[ok], y[ok]
        K = ins.K[j]
        acc = np.zeros((len(rows), K))
        cnt = 0
        for sd in seeds:
            perm = np.random.default_rng(sd).permutation(len(trj))
            parts = [perm[i::n_folds] for i in range(n_folds)] if n_folds > 1 else [np.array([], int)]
            for part in parts:
                keep = np.setdiff1d(np.arange(len(trj)), part)
                cl = np.unique(yj[keep])
                P = np.full((len(rows), K), 1e-6)
                if len(cl) < 2:
                    P[:, cl] = 1.0
                else:
                    m = TabICLClassifier(device="cuda", n_estimators=n_est, random_state=sd)
                    m.fit(df.iloc[trj[keep]], yj[keep])
                    P[:, m.classes_] = m.predict_proba(df.iloc[rows])
                acc += P / P.sum(1, keepdims=True)
                cnt += 1
        out.append(acc / cnt)
    return out


REGISTRY["tab_bag5"] = lambda ins, tr, rows: {"": m_tab_bag(ins, tr, rows, 5)}
REGISTRY["tab_est4"] = lambda ins, tr, rows: {"": m_tab_bag(ins, tr, rows, 1, n_est=4)}
REGISTRY["tab_seed3"] = lambda ins, tr, rows: {"": m_tab_bag(ins, tr, rows, 1, seeds=(0, 1, 2))}
REGISTRY["tab_bag5e4"] = lambda ins, tr, rows: {"": m_tab_bag(ins, tr, rows, 5, n_est=4)}
REGISTRY["tab_bag10"] = lambda ins, tr, rows: {"": m_tab_bag(ins, tr, rows, 10)}
