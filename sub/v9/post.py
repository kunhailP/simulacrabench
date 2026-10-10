"""v9 post-processing on top of v8's pooled tables, each step guarded by the
visible rows' OOF log score (official floor): a step is applied to an item, or
an instrument, only where it improves that score.

  country_eb  Empirical-Bayes recalibration by the country-like GIVEN column:
              q ∝ m(x) * pi_c / mbar_c, pi_c the cell's answer shares shrunk
              toward the model's cell mean mbar_c by a Dirichlet-multinomial
              alpha (lab: +0.0002..0.0003 on Afrobarometer and Wellcome).
  monotone    Sequential attrition: once a respondent leaves a single session
              every later item reads "Not answered", so P(Not answered) cannot
              fall along the questionnaire order. Segments where the visible
              rows show this, and whether dropout overrides routing (isotonic
              on P(NA)) or keeps it (isotonic on P(NA | asked)), are read off
              the visible rows (lab: +0.0004 on top of bagging, both
              conventions picked right, shuffled-order control unchanged).
"""

import numpy as np

import v7core as V

NA = "Not answered"
MAX_VIOL = 0.05


def _ll(P, y, K):
    p = P / P.sum(1, keepdims=True)
    p = V.FLOOR + (1 - K * V.FLOOR) * p
    return np.log(p[np.arange(len(y)), y])


# --------------------------------------------------------------- country EB
def country_col(ins):
    for g in ins.given:
        if "country" in g.lower():
            return g
    return None


def _cell_post(code_tr, y_tr, prior_tr, code_rows, prior_rows, K, mbar_fallback=True):
    cells, inv = np.unique(code_tr, return_inverse=True)
    tab = np.zeros((len(cells), K))
    np.add.at(tab, (inv, y_tr), 1.0)
    base = np.zeros((len(cells), K))
    np.add.at(base, inv, prior_tr)
    n = tab.sum(1, keepdims=True)
    base /= np.maximum(n, 1e-9)
    alpha = V._eb_alpha(tab, base)
    pos = np.minimum(np.searchsorted(cells, code_rows), len(cells) - 1)
    hit = cells[pos] == code_rows
    cnt = np.where(hit[:, None], tab[pos], 0.0)
    nn = cnt.sum(1, keepdims=True)
    mbar = np.where(hit[:, None], base[pos], prior_rows)
    pi = (cnt + alpha * mbar) / (nn + alpha)
    post = prior_rows * pi / np.maximum(mbar, 1e-9)
    return post / post.sum(1, keepdims=True)


def country_eb(ins, oof, tables, folds):
    g = country_col(ins)
    info = {"col": g, "items": 0}
    if g is None:
        return oof, tables, info
    code = ins.gcodes[g]
    vis, hid = ins.vis, ins.hid
    cv, ch = code[vis], code[hid]
    oof, tables = list(oof), list(tables)
    for j in range(len(ins.targets)):
        y = ins.Y[vis, j]
        K = int(ins.K[j])
        O = np.array(oof[j], dtype=np.float64)
        Onew = O.copy()
        for part in folds:
            trm = np.ones(len(vis), bool)
            trm[part] = False
            trm &= y >= 0
            if trm.sum() < 50:
                continue
            Onew[part] = _cell_post(cv[trm], y[trm], O[trm], cv[part], O[part], K)
        ok = y >= 0
        if ok.sum() == 0 or _ll(Onew[ok], y[ok], K).sum() <= _ll(O[ok], y[ok], K).sum():
            continue
        oof[j] = Onew
        tables[j] = _cell_post(cv[ok], y[ok], O[ok], ch, np.array(tables[j], dtype=np.float64), K)
        info["items"] += 1
    return oof, tables, info


# ---------------------------------------------------------- monotone attrition
def _pava_rows(P):
    out = np.empty_like(P)
    for r in range(P.shape[0]):
        vals, wts, cnt = [], [], []
        for x in P[r]:
            vals.append(x)
            wts.append(1.0)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
                wts.append(w1 + w2)
                cnt.append(c1 + c2)
        out[r] = np.repeat(vals, cnt)
    return out


def _apply(tabs, fam, seg, sent, use_cond):
    """Isotonic P(NA) (or P(NA | asked)) within each segment; other levels rescaled."""
    def ps(T, j):
        return T[:, sent[j]] if (use_cond and sent[j] >= 0) else 0.0
    out = list(tabs)
    P = np.stack([tabs[j][:, k] / np.maximum(1 - ps(tabs[j], j), 1e-9) for j, k in fam], 1)
    Q = P.copy()
    for s in np.unique(seg):
        m = seg == s
        if m.sum() > 1:
            Q[:, m] = _pava_rows(P[:, m])
    for c, (j, k) in enumerate(fam):
        h = np.array(tabs[j], dtype=np.float64)
        old = np.clip(h[:, k], 1e-9, 1 - 1e-9)
        new = np.clip(Q[:, c] * (1 - ps(h, j)), 1e-6, 1 - 1e-6)
        h *= ((1 - new) / (1 - old))[:, None]
        h[:, k] = new
        out[j] = h
    return out


def monotone(ins, oof, tables):
    info = {"items": 0}
    fam = [(j, ins.opts[t].index(NA)) for j, t in enumerate(ins.targets) if NA in ins.opts[t]]
    if len(fam) < 3:
        return tables, info
    sent = {j: (int(ins.K[j]) - 1 if ins.schema_items[t].get("gate") else -1)
            for j, t in enumerate(ins.targets)}
    Yv = ins.Y[ins.vis]
    seg, cur = [0], 0
    n_na = n_se = 0
    for c in range(1, len(fam)):
        (jp, kp), (jc, kc) = fam[c - 1], fam[c]
        yp, yc = Yv[:, jp], Yv[:, jc]
        ok = (yp >= 0) & (yc >= 0) & (yp != sent[jp]) & (yc != sent[jc])
        na_p = ok & (yp == kp)
        viol = ((yc != kc) & na_p).sum() / max(na_p.sum(), 1)
        if viol > MAX_VIOL or na_p.sum() < 20:
            cur += 1
        seg.append(cur)
        if sent[jc] >= 0:
            gone = yp == kp
            n_na += int(((yc == kc) & gone).sum())
            n_se += int(((yc == sent[jc]) & gone).sum())
    seg = np.array(seg)
    if np.bincount(seg).max() < 2:
        return tables, info
    use_cond = n_se > 0.2 * max(n_na + n_se, 1)
    new_oof = _apply(oof, fam, seg, sent, use_cond)
    gain = 0.0
    for j, _ in fam:
        ok = Yv[:, j] >= 0
        K = int(ins.K[j])
        gain += _ll(new_oof[j][ok], Yv[ok, j], K).sum() - _ll(np.asarray(oof[j])[ok], Yv[ok, j], K).sum()
    info.update(segments=int(len(np.unique(seg))), largest=int(np.bincount(seg).max()),
                mode="conditional" if use_cond else "plain", oof_gain=round(float(gain), 2))
    if gain <= 0:
        return tables, info
    info["items"] = len(fam)
    return _apply(tables, fam, seg, sent, use_cond), info


def temper_both(ins, oof, tables):
    """v8's per-item temperature, returned for the OOF tables as well."""
    yv = ins.Y[ins.vis]
    o2, t2 = list(oof), list(tables)
    for j in range(len(ins.targets)):
        ok = yv[:, j] >= 0
        if ok.sum() == 0:
            continue
        T = V.best_temperature(oof[j][ok], yv[ok, j], ins.K[j])
        o2[j] = V._temper(oof[j], T)
        t2[j] = V._temper(tables[j], T)
    return o2, t2
