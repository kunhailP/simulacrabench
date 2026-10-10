"""Evaluation suite: one submission across several real-data labs, with fixed roles.

Choosing models on one lab fits the method to that lab. The labs are therefore
split by role and the roles are fixed before results are seen:

  develop   free to look at while building a method
  select    used to accept or reject a change (rule: no loss on ANY select lab,
            i.e. the worst paired difference, not the mean)
  final     evaluated once, at the end, with --final; never used to choose

Paired differences are per lab (same hidden cells); the SE is over respondents
within a lab and says nothing about transfer to another survey, which is what
the spread across labs is for. Each lab is also broken down by item family:
the gate parent for gated items, "attrition" for items with "Not answered",
"plain" otherwise.

    .venv/bin/python tools/suite.py run sub/v8 [--roles develop,select] [--reps 2]
    .venv/bin/python tools/suite.py compare v8 v9 [--roles select]
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lab  # noqa: E402

# Roles are fixed here, before any comparison is run on them.
LABS = {
    "gss": "develop",
    "gss_hcr": "develop",
    "gss_j12": "develop",      # GSS with 12 items (UNICEF size), same respondents
    "gss_j76": "develop",      # GSS with 76 items (WB size)
    "gss_wb": "select",
    "gss_branch": "select",
    "gss_wb2": "select",       # WB layout: session attrition + independent tracer block
    "afro_r8": "select",       # UNICEF-like: country + religion GIVEN, vaccine items
    "reach_irq": "select",     # UNHCR-like: return / stay / undecided branches, multi-select
    "wgm2018": "select",       # UNICEF-like: child-vaccination item asked of parents only
    "srls": "final",           # UNHCR-like: Syrian refugee households (Jordan), CC0
    "afro_r9": "final",        # UNICEF-like, a different round of A
}


def labs_for(roles, final=False):
    out = [k for k, r in LABS.items() if r in roles]
    if "final" in roles and not final:
        raise SystemExit("final labs are evaluated once; pass --final to do it")
    return [k for k in out if os.path.exists(os.path.join(ROOT, "data", "proxy", k))]


def family_of(schema_path):
    s = json.load(open(schema_path))["items"]
    fam = {}
    for k, r in s.items():
        if r["class"] != "PREDICT":
            continue
        g = r.get("gate")
        if g:
            fam[k] = "gate:" + g["parent"]
        elif "Not answered" in r["values"]:
            fam[k] = "attrition"
        else:
            fam[k] = "plain"
    return fam


def compare_lab(a, b, name):
    d = os.path.join(ROOT, "results", "lab", name)
    fam = family_of(os.path.join(ROOT, "data", "proxy", name, "schema.json"))
    rows, per_fam = [], {}
    for rep in range(100):
        fa, fb = os.path.join(d, f"{a}_rep{rep}.npz"), os.path.join(d, f"{b}_rep{rep}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            break
        A, B = np.load(fa), np.load(fb)
        U = float(A["U"])
        df = pd.DataFrame({"rid": A["rid"], "item": A["item"], "d": (B["logp"] - A["logp"]) / U})
        per = df.groupby("rid")["d"].mean()
        rows.append((df["d"].mean(), per.std() / np.sqrt(len(per))))
        df["fam"] = df["item"].map(fam).fillna("plain")
        # contribution of each family to the instrument skill difference
        for f, v in df.groupby("fam")["d"].sum().items():
            per_fam.setdefault(f, []).append(v / len(df))
    if not rows:
        return None
    m = float(np.mean([r[0] for r in rows]))
    se = float(np.sqrt(np.mean([r[1] ** 2 for r in rows]) / len(rows)))
    fams = {f: float(np.mean(v)) for f, v in per_fam.items()}
    return m, se, len(rows), fams


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("sub")
    r.add_argument("--roles", default="develop,select")
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--tag")
    r.add_argument("--final", action="store_true")
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--roles", default="develop,select")
    c.add_argument("--final", action="store_true")
    c.add_argument("--top", type=int, default=4)
    x = ap.parse_args()
    roles = x.roles.split(",")
    names = labs_for(roles, x.final)
    if x.cmd == "run":
        tag = x.tag or os.path.basename(os.path.normpath(x.sub))
        for n in names:
            lab.run(x.sub, os.path.join(ROOT, "data", "proxy", n), x.reps, 0.15, tag)
        return
    worst = None
    for n in names:
        res = compare_lab(x.a, x.b, n)
        if res is None:
            print(f"{n:12s} [{LABS[n]}] missing runs")
            continue
        m, se, k, fams = res
        worst = m if worst is None else min(worst, m)
        top = sorted(fams.items(), key=lambda kv: kv[1])
        show = top[:x.top] + ([("...", 0)] if len(top) > 2 * x.top else []) + top[-x.top:]
        seen, parts = set(), []
        for f, v in show:
            if f in seen:
                continue
            seen.add(f)
            parts.append(f"{f} {v:+.4f}" if f != "..." else "...")
        print(f"{n:12s} [{LABS[n]:7s}] {x.b} - {x.a}: {m:+.4f} (se {se:.4f}, {k} reps)  | "
              + ", ".join(parts))
    if worst is not None:
        print(f"worst lab: {worst:+.4f}  ->  {'ACCEPT' if worst > 0 else 'REJECT'} (rule: no loss on any lab)")


if __name__ == "__main__":
    main()
