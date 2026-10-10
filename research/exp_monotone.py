"""WB attrition is sequential: once a respondent leaves the session, every later
item reads "Not answered". So for a hidden respondent P(Not answered on item j)
cannot decrease along the assessment order. Per-item models ignore this. Here,
per respondent, the "Not answered" probabilities of the items that carry that
level are made non-decreasing in schema order by weighted isotonic regression
(pool adjacent violators), the other levels of each item rescaled to keep its
row summing to one. No fitting, so nothing to overfit.

    .venv/bin/python research/exp_monotone.py --lab gss_wb --model tabicl1
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from screen import cache_path  # noqa: E402

FLOOR = 1e-3


def pava_rows(P, w):
    """Row-wise weighted isotonic (non-decreasing) fit of P [n, m]."""
    out = np.empty_like(P)
    for r in range(P.shape[0]):
        vals, wts, cnt = [], [], []
        for x, ww in zip(P[r], w):
            vals.append(x)
            wts.append(ww)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
                wts.append(w1 + w2)
                cnt.append(c1 + c2)
        out[r] = np.repeat(vals, cnt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--model", default="tabicl1")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="negative control: random item order")
    ap.add_argument("--max-viol", type=float, default=0.05)
    ap.add_argument("--equal", action="store_true", help="average P(NA) inside two-way (block) segments")
    ap.add_argument("--conditional", action="store_true", help="isotonic on P(NA | asked) for gated items")
    ap.add_argument("--auto", action="store_true", help="pick plain/conditional from the visible rows")
    a = ap.parse_args()
    t = np.load(cache_path(a.lab, a.rep, "_truth", 0))
    Yh, K, T = t["Yh"], t["K"], list(t["targets"])
    z = np.load(cache_path(a.lab, a.rep, a.model, 0))
    H = [z[f"h{j}"].astype(np.float64) for j in range(len(K))]
    H = [h / h.sum(1, keepdims=True) for h in H]
    sch = json.load(open(os.path.join(ROOT, "data", "proxy", a.lab, "schema.json")))["items"]
    order = [k for k in sch if sch[k]["class"] == "PREDICT"]
    fam = []
    for k in order:
        if "Not answered" in sch[k]["values"]:
            j = T.index(k)
            fam.append((j, sch[k]["values"].index("Not answered")))
    # sentinel slot per item (-1 if ungated): those cells are "not asked", not answers
    sent = {j: (K[j] - 1 if sch[T[j]].get("gate") else -1) for j in range(len(K))}
    if a.shuffle:
        rng = np.random.default_rng(0)
        fam = [fam[i] for i in rng.permutation(len(fam))]
    # runtime check on the VISIBLE rows only: link item c to c+1 when a visible
    # respondent who did not answer c almost never answers c+1
    from oof import build
    _, ins = build(os.path.join(ROOT, "data", "proxy", a.lab), a.rep)
    Yv = ins.Y[ins.vis]
    tidx = {t: i for i, t in enumerate(ins.targets)}
    seg_id, cur = [0], 0
    both = []          # per link: is the reverse direction also (almost) never seen?
    for c in range(1, len(fam)):
        jp, kp = fam[c - 1]
        jc, kc = fam[c]
        yp, yc = Yv[:, tidx[T[jp]]], Yv[:, tidx[T[jc]]]
        ok = (yp >= 0) & (yc >= 0) & (yp != sent[jp]) & (yc != sent[jc])
        na_p = ok & (yp == kp)
        viol = ((yc != kc) & na_p).sum() / max(na_p.sum(), 1)
        ans_p = ok & (yp != kp)
        rev = ((yc == kc) & ans_p).sum() / max(ans_p.sum(), 1)
        if viol > a.max_viol or na_p.sum() < 20:
            cur += 1
        else:
            both.append((cur, rev <= a.max_viol / 5))
        seg_id.append(cur)
    seg_id = np.array(seg_id)
    U = np.mean(np.log(K))

    def skill(Hs):
        lp = []
        for j in range(len(K)):
            p = FLOOR + (1 - K[j] * FLOOR) * Hs[j]
            lp.append(np.log(p[np.arange(len(Yh)), Yh[:, j]]))
        return 1 + np.stack(lp, 1).mean() / U

    base = skill(H)
    # recording convention, read off the visible rows: among respondents who did
    # not answer an earlier item, are later ROUTED-OUT cells written as the
    # sentinel (gate kept -> isotonic on P(NA | asked)) or as "Not answered"
    # (dropout overrides the gate -> isotonic on P(NA))?
    n_na = n_se = 0
    for c in range(1, len(fam)):
        jp, kp = fam[c - 1]
        jc, kc = fam[c]
        if sent[jc] < 0:
            continue
        yp, yc = Yv[:, tidx[T[jp]]], Yv[:, tidx[T[jc]]]
        gone = (yp == kp)
        n_na += ((yc == kc) & gone).sum()
        n_se += ((yc == sent[jc]) & gone).sum()
    use_cond = a.conditional or (a.auto and n_se > 0.2 * max(n_na + n_se, 1))
    mode = "conditional" if use_cond else "plain"

    def ps(j):
        return H[j][:, sent[j]] if (use_cond and sent[j] >= 0) else np.zeros(len(Yh))
    Pna = np.stack([H[j][:, k] / np.maximum(1 - ps(j), 1e-9) for j, k in fam], 1)
    viol = (np.diff(Pna, axis=1) < 0).mean()
    Q = Pna.copy()
    # a segment is a block (all-or-nothing) only if, over the WHOLE segment, people
    # who answered its first item almost never miss its last one; sequential
    # attrition fails this because dropout accumulates along the segment
    equal = set()
    for sg in np.unique(seg_id):
        cs = np.where(seg_id == sg)[0]
        if len(cs) < 2:
            continue
        (j0, k0), (j1, k1) = fam[cs[0]], fam[cs[-1]]
        y0, y1 = Yv[:, tidx[T[j0]]], Yv[:, tidx[T[j1]]]
        ok = (y0 >= 0) & (y1 >= 0) & (y0 != k0) & (y0 != sent[j0]) & (y1 != sent[j1])
        if ok.sum() >= 20 and ((y1 == k1) & ok).sum() / ok.sum() <= a.max_viol / 5:
            equal.add(sg)
    for sg in np.unique(seg_id):
        m = seg_id == sg
        if m.sum() > 1:
            if a.equal and sg in equal:      # block missingness: one shared probability
                Q[:, m] = Pna[:, m].mean(1, keepdims=True)
            else:
                Q[:, m] = pava_rows(Pna[:, m], np.ones(m.sum()))
    H2 = list(H)
    for c, (j, k) in enumerate(fam):
        h = H[j].copy()
        old = np.clip(h[:, k], 1e-9, 1 - 1e-9)
        new = np.clip(Q[:, c] * (1 - ps(j)), 1e-6, 1 - 1e-6)
        h *= ((1 - new) / (1 - old))[:, None]
        h[:, k] = new
        H2[j] = h
    print(f"{a.lab} {a.model} [{mode}]: {len(fam)} attrition items; adjacent decreases in P(NA) {viol:.1%}; "
          f"segments {len(np.unique(seg_id))} (largest {np.bincount(seg_id).max()}, block {len(equal)}); "
          f"skill {base:.4f} -> {skill(H2):.4f} ({skill(H2) - base:+.4f})")


if __name__ == "__main__":
    main()
