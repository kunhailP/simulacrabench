"""Real-data proxy instrument from the public General Social Survey (2016, 2018, 2022).

Synthetic worlds are built by us, so a method can win there because we planted
the structure it finds. GSS is real survey microdata with the same shape as the
competition instruments: a few demographic GIVEN columns, ~150 categorical
attitude/behaviour items, split-ballot and skip-logic routing ("IAP" codes),
and genuine non-response levels. It is a lab, not a target: nothing here says
the competition data looks like GSS, only that it is real.

Gates are inferred the way the UNHCR schema infers one of its own: for each item
that carries IAP, the single candidate parent whose values best separate asked
from not-asked. The delivered rows need not satisfy it.

    tools/fetch_gss.sh                       # -> data/raw/gss (hash-checked)
    .venv/bin/python tools/proxy_gss.py --out data/proxy/gss
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import pyreadstat

SRC = os.environ.get("GSS_SRC", "data/raw/gss")   # tools/fetch_gss.sh
FILES = [(2016, "gss2016/GSS2016.dta"), (2018, "gss2018/GSS2018.dta"),
         (2022, "gss2022/2022/GSS2022.dta")]
GV = "NA_GATED"
MISS = {"d": "Don't know", "n": "No answer", "s": "Skipped"}
GIVEN = ["year", "sex", "race", "region", "agegrp", "degree", "marital",
         "childs", "wrkstat", "relig"]
# admin, interviewer, household roster, and recodes of the GIVEN columns
DROP = re.compile(
    r"^(id|.*wt.*|vpsu|vstrat|sample|form|version|issp|phase|mode|spaneng|"
    r"rvisitor|feeused|respond|consent|subsamprate|srcbelt|phone|dwel.*|"
    r"rgroomed|rlooks|huclean|ratetone|comprend|respnum|rank|intethn|hhrace|"
    r"hhtype.*|hompop.*|adults|earnrs.*|babies.*|preteen.*|teens.*|relate\d+|"
    r"relhh.*|relsp\d+|mar\d+|old\d+|gender\d+|.*_\d{4}|racecen\d|region.*|"
    r"reg16.*|intrace\d|ethnic|race|sex|age|degree|marital|childs|wrkstat|"
    r"relig|year|ballot|huadd.*|cohrs\d|coden|codeg|cowrk.*|cojew|"
    r"spanint|spanself|lngthinv|intage|intsex|intyrs|unrelat.*|famgen.*|"
    r"hispanic|raceacs\d+|res16|family16|mobile16|incom16|reg16|born|"
    r"parborn|granborn|madeg|padeg|mawrk.*|pawrk.*|othlang\d|letin1a)$")


def band_age(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return "Not recorded"
    for hi, lab in [(24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
                    (64, "55-64")]:
        if a <= hi:
            return lab
    return "65+"


def label(meta, var, code):
    if code in MISS:
        return MISS[code]
    if len(code) == 1 and code.isalpha():
        return "Other missing" if code != "i" else "i"
    try:
        f = float(code)
        lab = meta.variable_value_labels.get(var, {}).get(f)
        c = str(int(f)) if f == int(f) else code
    except ValueError:
        lab, c = None, code
    return f"{c}. {lab}" if lab else c


def load(src=SRC):
    frames = []
    for y, f in FILES:
        d, m = pyreadstat.read_dta(os.path.join(src, f), user_missing=True)
        d.columns = [c.lower() for c in d.columns]
        m.variable_value_labels = {k.lower(): v for k, v in m.variable_value_labels.items()}
        out = {}
        for c in d.columns:
            out[c] = [label(m, c, str(v).replace(".0", "") if isinstance(v, float) else str(v))
                      if not (isinstance(v, float) and np.isnan(v)) else "Not recorded"
                      for v in d[c]]
        df = pd.DataFrame(out)
        df["year"] = str(y)
        df["agegrp"] = [band_age(v) for v in d["age"]]
        ch = pd.to_numeric(d["childs"], errors="coerce")
        df["childs"] = ["Not recorded" if np.isnan(v) else ("4+" if v >= 4 else str(int(v)))
                        for v in ch]
        frames.append(df)
    common = set.intersection(*[set(f.columns) for f in frames])
    return pd.concat([f[sorted(common)] for f in frames], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/proxy/gss")
    ap.add_argument("--max-items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = load(a.src)
    for g in GIVEN:  # GIVEN never carries the IAP marker
        df[g] = df[g].replace({"i": "Not recorded"})

    # candidate PREDICT items: 2-12 substantive levels, asked of >= 5% of rows
    cand = []
    for c in df.columns:
        if c in GIVEN or DROP.match(c) or c == "ballot":
            continue
        v = df[c]
        sub = [x for x in v.unique() if x not in ("i", "Not recorded", "Other missing")
               and x not in MISS.values()]
        if not 2 <= len(sub) <= 12 or (v != "i").mean() < 0.05:
            continue
        if (v == "Not recorded").mean() > 0.5:
            continue
        cand.append(c)
    rng.shuffle(cand)
    targets = sorted(cand[:a.max_items])

    # ballot is a hidden routing variable, like s3_11 in UNHCR: a PREDICT parent
    df["ballot"] = df["ballot"].replace({"i": "Not recorded"})
    parents = GIVEN + ["ballot"] + targets
    items = {}
    for g in GIVEN:
        items[g] = {"question": g, "class": "GIVEN", "values": sorted(df[g].unique()),
                    "gate": None}
    items["ballot"] = {"question": "ballot", "class": "PREDICT",
                       "values": sorted(df["ballot"].unique()), "gate": None}
    agree = []
    for t in targets:
        asked = (df[t] != "i").to_numpy()
        gate = None
        if not asked.all():
            best = (-1, None, None)
            for p in parents:
                if p == t:
                    continue
                pv = df[p].to_numpy()
                tab = pd.crosstab(pv, asked)
                if True not in tab.columns:
                    continue
                rate = tab[True] / tab.sum(1)
                obs = [k for k, r in rate.items() if r >= 0.5 and k != "i"]
                if not obs or len(obs) == len(rate):
                    continue
                acc = float((np.isin(pv, obs) == asked).mean())
                if acc > best[0]:
                    best = (acc, p, obs)
            if best[1] is not None:
                gate = {"parent": best[1], "observed_if": sorted(best[2])}
                agree.append(best[0])
        vals = sorted(x for x in df[t].unique() if x != "i")
        if gate is None and not asked.all():
            df[t] = df[t].replace({"i": "Not asked"})
            vals = sorted(df[t].unique())
        else:
            df[t] = df[t].replace({"i": GV})
        items[t] = {"question": t, "class": "PREDICT", "values": vals, "gate": gate}

    # keep the parent order valid: a gate parent must come before its child
    order = GIVEN + ["ballot"] + targets
    n = len(df)
    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    n_dev, n_test = int(0.06 * n), int(0.14 * n)
    n_train = n - n_dev - n_test
    df["respondent_id"] = [f"G{i:06d}" for i in range(n)]
    df["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-gss-1",
                          "description": "GSS 2016/2018/2022 proxy"},
              "items": {k: items[k] for k in order},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    df[["respondent_id"] + order + ["role"]].to_parquet(
        os.path.join(a.out, "respondents.parquet"), index=False)
    ng = sum(1 for k in targets if items[k]["gate"])
    print(f"wrote {a.out}: {n} rows, {len(GIVEN)} GIVEN, {len(targets) + 1} PREDICT, "
          f"{ng} gated (agreement median {np.median(agree):.2f}, "
          f"min {np.min(agree):.2f})")


if __name__ == "__main__":
    main()
