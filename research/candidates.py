"""Candidate base models outside the v7 family, same signature as v7's:
m(ins, tr, rows) -> {config: [table_j of shape (len(rows), K_j)]}.
"""

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
