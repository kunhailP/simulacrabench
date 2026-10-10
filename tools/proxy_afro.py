"""Real-data proxy instrument from the public Afrobarometer Round 8 / Round 9 merges.

Same construction as tools/proxy_gss.py, applied to a multi-country survey that
stands in for the UNICEF instrument (country + religion GIVEN, vaccine-attitude
items, some gated on a "vaccinated?" item). Rules fixed in advance:

  GIVEN    country, urban/rural PSU, age band, gender, education (condensed),
           religion (condensed), employment status.
  PREDICT  every eligible substantive question (questionnaire Q-items and the
           R8 COV module; not demographics, interviewer/admin, verbatim, IDs,
           weights, derived indices or multi-response slots) with 2-12
           substantive levels, asked of >= 5% of rows; up to --max-items drawn
           by a seeded shuffle (seed 0). Items whose label mentions a vaccine
           are always included when eligible.
  Routing  codes labelled "Not applicable" / "Not asked in this country" are
           the skip marker -> NA_GATED, with the gate inferred as the single
           EARLIER candidate (GIVEN or an earlier selected item) whose levels
           best separate asked from not-asked. Refused / Don't know / Missing
           stay as ordinary labelled levels.

    tools/fetch_afro.sh                                  # -> data/raw/afro
    .venv/bin/python -I tools/proxy_afro.py --round 8 --out data/proxy/afro_r8
    .venv/bin/python -I tools/proxy_afro.py --round 9 --out data/proxy/afro_r9
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import pyreadstat

SRC = os.environ.get("AFRO_SRC", "data/raw/afro")   # tools/fetch_afro.sh
FILES = {8: "afro_r8.sav", 9: "afro_r9.sav"}
GV = "NA_GATED"
SKIP = "i"   # internal skip marker, as in proxy_gss
SKIP_LAB = re.compile(r"not applicable|not asked", re.I)
MISS_LAB = re.compile(r"missing|refus|don.?t know|do not know|does not know|"
                      r"do not understand|can.?t determine|not recorded", re.I)

# source column for each GIVEN item, per round
GIVEN_SRC = {
    8: {"country": "COUNTRY", "urbrur": "URBRUR", "gender": "Q101",
        "educ": "EDUC_COND", "relig": "RELIG_COND", "employ": "Q95A"},
    9: {"country": "COUNTRY", "urbrur": "URBRUR", "gender": "Q100",
        "educ": "EDUC_COND", "relig": "RELIG_COND", "employ": "Q93A"},
}
GIVEN = ["country", "urbrur", "agegrp", "gender", "educ", "relig", "employ"]
AGE = "Q1"
# questionnaire items that are demographics / GIVEN sources / admin, and the
# first question number of the interviewer/supervisor section
DEMOG = {
    8: {"Q1", "Q2", "Q81", "Q95A", "Q97", "Q98A", "Q101", "Q102", "Q103",
        "Q104", "Q105"},
    9: {"Q1", "Q2", "Q84A", "Q93A", "Q94", "Q95", "Q98A", "Q99", "Q100",
        "Q101", "Q102", "Q103", "Q104"},
}
INTERVIEWER_FROM = {8: 106, 9: 105}
ITEM = re.compile(r"^(?:Q(\d+)[A-Z]*\d*|COV\d+)$")   # no _n slots, no OTHER
VACC = re.compile(r"vaccin", re.I)


def band_age(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Not recorded"
    v = float(v)
    if v == 998:
        return "Refused"
    if v == 999:
        return "Don't know"
    if v < 0:
        return "Not recorded"
    for hi, lab in [(24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
                    (64, "55-64")]:
        if v <= hi:
            return lab
    return "65+"


def labelled(col, vl, skip=True):
    """Readable string levels; skip-coded cells become the SKIP marker."""
    codes = {}
    for v in pd.unique(col):
        if isinstance(v, float) and np.isnan(v):
            codes[v] = "Not recorded"
            continue
        if isinstance(v, str):
            codes[v] = v.strip() or "Not recorded"
            continue
        f = float(v)
        lab = vl.get(f)
        c = str(int(f)) if f == int(f) else str(f)
        if skip and lab and SKIP_LAB.search(lab):
            codes[v] = SKIP
        else:
            codes[v] = f"{c}. {lab}" if lab else c
    out = col.map(codes)
    return out.fillna("Not recorded").astype(str)


def eligible(c, rnd):
    m = ITEM.match(c)
    if not m or c in DEMOG[rnd] or c.endswith("OTHER"):
        return False
    return not (m.group(1) and int(m.group(1)) >= INTERVIEWER_FROM[rnd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[8, 9], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    rnd = a.round
    rng = np.random.default_rng(a.seed)
    raw, meta = pyreadstat.read_sav(os.path.join(a.src, FILES[rnd]), user_missing=True)
    vls, qlab = meta.variable_value_labels, dict(zip(meta.column_names, meta.column_labels))

    df = pd.DataFrame(index=raw.index)
    for g, src in GIVEN_SRC[rnd].items():
        df[g] = labelled(raw[src], vls.get(src, {}), skip=False)
    df["agegrp"] = [band_age(v) for v in raw[AGE]]
    qtext = {g: qlab.get(s) or g for g, s in GIVEN_SRC[rnd].items()}
    qtext["agegrp"] = "Age (banded from Q1)"

    # candidate PREDICT items: 2-12 substantive levels, asked of >= 5% of rows
    order_src = [c for c in raw.columns if eligible(c, rnd)]
    lab = {}
    cand = []
    for c in order_src:
        v = labelled(raw[c], vls.get(c, {}))
        sub = [x for x in v.unique() if x != SKIP and not MISS_LAB.search(x)]
        if not 2 <= len(sub) <= 12 or (v != SKIP).mean() < 0.05:
            continue
        if (v == "Not recorded").mean() > 0.5:
            continue
        lab[c] = v
        cand.append(c)
    vacc = [c for c in cand if VACC.search(qlab.get(c) or "")]
    vacc_all = [c for c in order_src if VACC.search(qlab.get(c) or "")]
    rest = [c for c in cand if c not in vacc]
    rng.shuffle(rest)
    chosen = set(vacc) | set(rest[:max(0, a.max_items - len(vacc))])
    targets = [c for c in order_src if c in chosen]   # questionnaire order
    for c in targets:
        df[c.lower()] = lab[c].to_numpy()
    tk = [c.lower() for c in targets]

    items = {}
    for g in GIVEN:
        items[g] = {"question": qtext[g], "class": "GIVEN",
                    "values": sorted(df[g].unique()), "gate": None}
    agree = []
    for i, (src, t) in enumerate(zip(targets, tk)):
        asked = (df[t] != SKIP).to_numpy()
        gate = None
        if not asked.all():
            best = (-1, None, None)
            for p in GIVEN + tk[:i]:   # only earlier items can be parents
                pv = df[p].to_numpy()
                rate = pd.Series(asked).groupby(pv).mean()
                obs = [k for k, r in rate.items() if r >= 0.5 and k not in (SKIP, GV)]
                if not obs or len(obs) == len(rate):
                    continue
                acc = float((np.isin(pv, obs) == asked).mean())
                if acc > best[0]:
                    best = (acc, p, obs)
            if best[1] is not None:
                gate = {"parent": best[1], "observed_if": sorted(best[2])}
                agree.append(best[0])
        if gate is None and not asked.all():
            df[t] = df[t].replace({SKIP: "Not asked"})
        else:
            df[t] = df[t].replace({SKIP: GV})
        vals = sorted(x for x in df[t].unique() if x != GV)
        items[t] = {"question": qlab.get(src) or src, "class": "PREDICT",
                    "values": vals, "gate": gate}

    order = GIVEN + tk
    n = len(df)
    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    n_dev, n_test = int(0.06 * n), int(0.14 * n)
    n_train = n - n_dev - n_test
    df["respondent_id"] = [f"A{rnd}{i:06d}" for i in range(n)]
    df["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": f"proxy-afro-r{rnd}-1",
                          "description": f"Afrobarometer Round {rnd} merged release proxy"},
              "items": {k: items[k] for k in order},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=1, ensure_ascii=False)
    df[["respondent_id"] + order + ["role"]].to_parquet(
        os.path.join(a.out, "respondents.parquet"), index=False)
    ng = sum(1 for k in tk if items[k]["gate"])
    na = sum(1 for k in tk if "Not asked" in items[k]["values"])
    print(f"wrote {a.out}: {n} rows, {len(GIVEN)} GIVEN, {len(tk)} PREDICT "
          f"(of {len(cand)} eligible), {ng} gated (agreement median "
          f"{np.median(agree):.2f}, min {np.min(agree):.2f}), {na} ungated 'Not asked'")
    print("vaccine items included:", [c.lower() for c in vacc],
          "| vaccine items ineligible:", [c for c in vacc_all if c not in vacc])
    print("gates:", {k: items[k]["gate"]["parent"] for k in tk if items[k]["gate"]})


if __name__ == "__main__":
    main()
