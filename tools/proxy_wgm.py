"""Real-data proxy instrument from the Wellcome Global Monitor 2018 (Gallup).

A second real-data lab next to tools/proxy_gss.py, chosen because its routing
mirrors the UNICEF instrument: the vaccine attitude items are asked only of
people who have heard of vaccines (Q23), and whether a respondent's children
were vaccinated (Q28) only of parents (Q27). ~149k respondents in 144
countries, a handful of demographic GIVEN columns and ~30 categorical items
with explicit (DK)/(Refused) levels. It is a lab, not a target.

Rules fixed in advance (not tuned on predictability, no model is fit here):
  * rows: a seeded random sample of 40,000 respondents (seed 0), drawn before
    any item is looked at;
  * GIVEN: country, world region, age band, gender, education, income
    quintile, urban/rural, employment;
  * PREDICT: every substantive item with 2-12 levels asked of >= 5% of rows,
    seeded draw of up to 100 (all of them fit), plus the vaccine items always;
    derived indices (WGM_Index*, ViewOfScience, AgeCategories, WBI) dropped;
  * routing: a blank cell is "not asked". Gates come from the questionnaire
    notes where they are explicit (QGATES); otherwise inferred like proxy_gss
    (the earlier item whose levels best separate asked from not-asked). The
    delivered rows need not satisfy a gate exactly.

    tools/fetch_wgm.sh                       # -> data/raw/wgm (hash-checked)
    .venv/bin/python -I tools/proxy_wgm.py --out data/proxy/wgm2018
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

SRC = os.environ.get("WGM_SRC", "data/raw/wgm/wgm2018.xlsx")  # tools/fetch_wgm.sh
GV = "NA_GATED"
BLANK = "i"            # internal marker for a blank (not asked) cell
N_SAMPLE = 40000
# GIVEN: output key -> source column
GIVEN = {"country": "WP5", "region": "Regions_Report", "agegrp": "Age",
         "gender": "Gender", "education": "Education",
         "income_quintile": "Household_Income", "urban_rural": "Urban_Rural",
         "employment": "EMP_2010"}
# never items: admin, weights, recodes of other columns, GIVEN sources
DROP = {"wgt", "PROJWT", "FIELD_DATE", "YEAR_CALENDAR", "WGM_Index",
        "WGM_Indexr", "ViewOfScience", "AgeCategories", "WBI"} | set(GIVEN.values())
# always PREDICT (fixed rule): vaccine attitudes, children vaccinated, trust
# in doctors/nurses, and the routing items the vaccine gates hang off
FORCE = ["Q11E", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28"]
# explicit questionnaire routing (data dictionary notes): child -> (parent, observed_if codes)
QGATES = {"Q24": ("Q23", ["1"]), "Q25": ("Q23", ["1"]), "Q26": ("Q23", ["1"]),
          "Q28": ("Q27", ["1"]), "Q29": ("D1", ["1"]), "Q30": ("Q29", ["1"]),
          "Q5B": ("Q5A", "not97"), "Q5C": ("Q5B", "not97")}
NONSUB = ("(DK)", "(Refused)", "(DK)/(Refused)", "Not recorded")


def codes(spec):
    """'1=A lot, 2=Some, 98=(DK)' -> {'1': 'A lot', ...} (tolerates '1-Most')."""
    spec = str(spec)
    out = {}
    for m in re.finditer(r"(\d+)\s*[=\-]\s*(.*?)(?=,?\s*\d+\s*[=\-]|$)", spec):
        out[m.group(1)] = m.group(2).strip().rstrip(",").strip()
    return out


def band_age(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return "Not recorded"
    if a >= 100:  # 100 = (Refused)
        return "Not recorded"
    for hi, lab in [(24, "15-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
                    (64, "55-64")]:
        if a <= hi:
            return lab
    return "65+"


def norm(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return BLANK if np.isnan(v) else (str(int(v)) if v == int(v) else str(v))
    s = str(v).strip()
    return s if s else BLANK


def load(src):
    raw = pd.read_excel(src, sheet_name="Full dataset", dtype=object)
    dic = pd.read_excel(src, sheet_name="Data dictionary", dtype=object)
    spec = {r[0]: codes(r[2]) for r in dic.itertuples(index=False)}
    quest = {r[0]: (str(r[1]).strip() if isinstance(r[1], str) else r[0])
             for r in dic.itertuples(index=False)}
    order = [r[0] for r in dic.itertuples(index=False)]
    return raw, spec, quest, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/proxy/wgm2018")
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    raw, spec, quest, dorder = load(a.src)

    # fixed row sample, drawn before any item is inspected
    keep = np.sort(rng.choice(len(raw), size=min(N_SAMPLE, len(raw)), replace=False))
    raw = raw.iloc[keep].reset_index(drop=True)
    codes_df = raw.apply(lambda s: s.map(norm))

    df = pd.DataFrame(index=raw.index)
    for k, src in GIVEN.items():
        if k == "agegrp":
            df[k] = [band_age(v) for v in raw[src]]
        else:
            m = spec[src]
            df[k] = [m.get(c, c) if c != BLANK else "Not recorded" for c in codes_df[src]]

    # candidate PREDICT items
    cand = []
    for c in dorder:
        if c in DROP or c not in codes_df:
            continue
        v = codes_df[c]
        sub = [x for x in v.unique() if x != BLANK
               and spec[c].get(x, x) not in NONSUB]
        if not 2 <= len(sub) <= 12 or (v != BLANK).mean() < 0.05:
            continue
        cand.append(c)
    pool = [c for c in cand if c not in FORCE]
    rng.shuffle(pool)
    chosen = set(pool[:max(0, a.max_items - len(FORCE))]) | (set(FORCE) & set(cand))
    targets = [c for c in dorder if c in chosen]   # questionnaire order

    def lab(c, x):
        if x == BLANK:
            return BLANK
        t = spec[c].get(x)
        return f"{x}. {t}" if t else x

    for t in targets:
        df[t] = [lab(t, x) for x in codes_df[t]]
    # Subjective_Income: the dictionary says (DK)/(Refused) were blanked, so a
    # blank there is non-response, not routing
    if "Subjective_Income" in targets:
        df["Subjective_Income"] = df["Subjective_Income"].replace(
            {BLANK: "(DK)/(Refused)"})

    items = {}
    for g in GIVEN:
        items[g] = {"question": quest.get(GIVEN[g], g) if g != "agegrp" else "Age (banded)",
                    "class": "GIVEN", "values": sorted(df[g].unique()), "gate": None}
    agree, fixed_agree, accs = [], {}, {}
    for i, t in enumerate(targets):
        asked = (df[t] != BLANK).to_numpy()
        gate, acc_t = None, None
        if not asked.all():
            if t in QGATES and QGATES[t][0] in items:
                p, obs = QGATES[t]
                pv = df[p].to_numpy()
                levels = [x for x in items[p]["values"]]
                if obs == "not97":
                    obs = [x for x in levels if not x.startswith("97.")]
                else:
                    obs = [x for x in levels if x.split(".")[0] in obs]
                acc_t = float((np.isin(pv, obs) == asked).mean())
                gate = {"parent": p, "observed_if": sorted(obs)}
                fixed_agree[t] = acc_t
            else:
                best = (-1, None, None)
                for p in list(GIVEN) + targets[:i]:   # earlier keys only
                    pv = df[p].to_numpy()
                    tab = pd.crosstab(pv, asked)
                    if True not in tab.columns:
                        continue
                    rate = tab[True] / tab.sum(axis=1)
                    obs = [k for k, r in rate.items() if r >= 0.5 and k != GV]
                    if not obs or len(obs) == len(rate):
                        continue
                    acc = float((np.isin(pv, obs) == asked).mean())
                    if acc > best[0]:
                        best = (acc, p, obs)
                if best[1] is not None:
                    gate = {"parent": best[1], "observed_if": sorted(best[2])}
                    acc_t = best[0]
            if acc_t is not None:
                agree.append(acc_t)
                accs[t] = acc_t
        if gate is None and not asked.all():
            df[t] = df[t].replace({BLANK: "Not asked"})
            vals = sorted(df[t].unique())
        else:
            df[t] = df[t].replace({BLANK: GV})
            vals = sorted(x for x in df[t].unique() if x != GV)
        items[t] = {"question": quest.get(t, t), "class": "PREDICT", "values": vals,
                    "gate": gate}

    order = list(GIVEN) + targets
    n = len(df)
    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    n_dev, n_test = int(0.06 * n), int(0.14 * n)
    n_train = n - n_dev - n_test
    df["respondent_id"] = [f"W{i:06d}" for i in range(n)]
    df["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-wgm2018-1",
                          "description": "Wellcome Global Monitor 2018 proxy "
                                         f"(seeded sample of {n} respondents)"},
              "items": {k: items[k] for k in order},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    df[["respondent_id"] + order + ["role"]].to_parquet(
        os.path.join(a.out, "respondents.parquet"), index=False)
    gated = [k for k in targets if items[k]["gate"]]
    print(f"wrote {a.out}: {n} rows, {len(GIVEN)} GIVEN, {len(targets)} PREDICT, "
          f"{len(gated)} gated (agreement median {np.median(agree):.3f}, "
          f"min {np.min(agree):.3f})")
    for k in gated:
        g = items[k]["gate"]
        src = "questionnaire" if k in fixed_agree else "inferred"
        print(f"  {k} <- {g['parent']} {g['observed_if'][:4]}"
              f"{'...' if len(g['observed_if']) > 4 else ''} ({src}, agree {accs[k]:.3f}, "
              f"NA_GATED {(df[k] == GV).mean():.3f})")
    na = [k for k in targets if "Not asked" in items[k]["values"]]
    if na:
        print("  ungated 'Not asked':", na)


if __name__ == "__main__":
    main()
