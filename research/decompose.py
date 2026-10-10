"""Where is the loss? Decompose a model's hidden log loss by cell type, per lab.

Cell types (from the true answer): 'sentinel' (NA_GATED), 'nonsub' (Not answered /
DK / refused / not recorded ...), 'content' (a substantive answer). Item types:
gated or not, number of options K. For each type: share of cells, share of the
model's total loss, loss per cell, the gain over the marginal (what GIVEN buys)
and the half-data penalty (estimation-limited part).

    .venv/bin/python research/decompose.py --labs gss,afro_r8,reach_irq
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from screen import cache_path  # noqa: E402

FLOOR = 1e-3
NONSUB = re.compile(r"(not answered|don.?t know|refused|no answer|not recorded|not asked|skipped|"
                    r"missing|prefer not|\(dk\)|no response|^9[789]\.)", re.I)


def logp(lab, rep, model, Yh, K):
    z = np.load(cache_path(lab, rep, model, 0))
    out = []
    for j in range(len(K)):
        P = z[f"h{j}"].astype(np.float64)
        P = FLOOR + (1 - K[j] * FLOOR) * P / P.sum(1, keepdims=True)
        out.append(np.log(P[np.arange(len(Yh)), Yh[:, j]]))
    return np.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", required=True)
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    for lab in a.labs.split(","):
        t = np.load(cache_path(lab, a.rep, "_truth", 0))
        Yh, K, T = t["Yh"], t["K"], list(t["targets"])
        sch = json.load(open(os.path.join(ROOT, "data", "proxy", lab, "schema.json")))["items"]
        L = {m: -logp(lab, a.rep, m, Yh, K) for m in ("tabicl1", "marg", "tab_half")}
        U = np.mean(np.log(K))
        n, J = Yh.shape
        ctype = np.empty((n, J), object)
        gated = np.zeros(J, bool)
        for j, tk in enumerate(T):
            opts = sch[tk]["values"] + (["NA_GATED"] if sch[tk].get("gate") else [])
            gated[j] = bool(sch[tk].get("gate"))
            lab_of = np.array(["sentinel" if v == "NA_GATED" else "nonsub" if NONSUB.search(v)
                               else "content" for v in opts])
            ctype[:, j] = lab_of[Yh[:, j]]
        tot = L["tabicl1"].sum()
        print(f"\n== {lab}: {J} items, {n} hidden rows; loss/cell {L['tabicl1'].mean():.3f} nats "
              f"(skill {1 - L['tabicl1'].mean() / U:.4f})")
        print(f"   {'group':22s} {'cells':>6s} {'loss%':>6s} {'nat/cell':>8s} {'GIVEN gain':>10s} {'half pen.':>9s}")

        def row(name, m):
            if m.sum() == 0:
                return
            tl = L["tabicl1"][m]
            print(f"   {name:22s} {m.mean():6.1%} {tl.sum() / tot:6.1%} {tl.mean():8.3f} "
                  f"{(L['marg'][m] - tl).mean():10.4f} {(L['tab_half'][m] - tl).mean():9.4f}")
        for c in ("sentinel", "nonsub", "content"):
            row(f"cell={c}", ctype == c)
        G = np.broadcast_to(gated, (n, J))
        row("item gated", G)
        row("item not gated", ~G)
        for lo, hi in ((2, 3), (4, 6), (7, 12), (13, 99)):
            kk = np.broadcast_to((K >= lo) & (K <= hi), (n, J))
            row(f"K {lo}-{hi}", kk)
        # concentration: share of loss in the worst 10% of items
        per_item = L["tabicl1"].sum(0)
        top = np.sort(per_item)[::-1]
        k10 = max(1, J // 10)
        print(f"   worst 10% of items hold {top[:k10].sum() / tot:.1%} of the loss; "
              f"half-data penalty concentrated: worst 10% items hold "
              f"{np.sort((L['tab_half'] - L['tabicl1']).sum(0))[::-1][:k10].sum() / max((L['tab_half'] - L['tabicl1']).sum(), 1e-9):.1%}")


if __name__ == "__main__":
    main()
