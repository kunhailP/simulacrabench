"""Real-data proxy instrument from REACH Iraq CCCM IDP Camp Profiling VII.

Source: REACH Initiative on behalf of the CCCM Cluster Iraq, "Iraq - CCCM IDP
Camp Profiling VII" (household data collected 21 Dec 2016 - 26 Jan 2017, 57
camps), published on the Humanitarian Data Exchange (HDX) under the Creative
Commons Attribution for Intergovernmental Organisations licence (CC BY-IGO).
Attribution: REACH Initiative / CCCM Cluster Iraq. This lab is a derived,
recoded copy (banded counts, readable labels, routing sentinel); it is not
endorsed by REACH or the CCCM Cluster.

Same conventions as tools/proxy_gss.py: a handful of household/sampling GIVEN
columns, up to 200 categorical PREDICT items, skip-logic blanks carried as
NA_GATED with a gate parent. Selection is fixed in advance and never tuned by
predictability:

  * always included: the movement-intention item (move in next 3 months),
    its destination and every reason-for-moving select-multiple binary, all
    gated on intention == "Yes" (the questionnaire's relevance);
  * the rest of PREDICT: seeded random draw (seed 0) from eligible items
    (2-12 levels, asked of >= 5% of rows).

Other gates are inferred like proxy_gss: the single earlier item whose levels
best separate asked from not-asked. Only one workbook sheet is household
microdata ("CCCM Camp Profiling VII dataset"); the questionnaire sheet is a
paper layout, so readable levels come from cleaning the ODK choice names.

    tools/fetch_reach.sh                       # -> data/raw/reach (hash-checked)
    .venv/bin/python tools/proxy_reach.py --out data/proxy/reach_irq
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

SRC = os.environ.get("REACH_SRC", "data/raw/reach")   # tools/fetch_reach.sh
FILE = "REACH_IRAQ_CCCM_Camp_Profiling_VII_Dataset_Feb2017.xlsx"
SHEET = "CCCM Camp Profiling VII dataset"
GV = "NA_GATED"
NA = "__NA__"          # internal marker for a blank (not asked) cell
ROSTER = ["0", "1", "2", "3", "4", "5", "6-10", "11+"]

INTENT = "households intending to move in next 3 months"
DEST = "destination of households intending to move"
REASON = "reason for wanting to move/"

# admin, timestamps, ids, free text, derived totals, concatenated
# select-multiple strings (their binaries are kept), and GIVEN sources
DROP = re.compile(
    r"^(today|date of .*|tent_age|meta/.*|_.*|.*other.*text.*|.*_other|"
    r"destination other| other_difficulty_type|.*/other reason|.*other_description|"
    r"other_livelihood|lending_sources_other|priority_needs/needs_other|"
    r".*/other_text|reason for not expressing complaint/other|"
    r"household drinking water source/other|non-drinking water source/other|"
    r"waste_disposal__source/other|tent_type_other|camp_name_other|"
    r"shelter_type_other|complaint_outcome_other|camp_committee_other|"
    r".*calculation.*|.*/calc.*|total male|total female|total population.*|"
    r"contact_group/.*|food_consumption/consume_food|"
    r"food_coping_strategies/foodcoping|"
    r"reason for wanting to move|what family members|"
    r"are any of these camp committees present in this site\?|"
    r"distribution information source|priority_information_needs|"
    r"latrine_type|shower_type|education_type| non_attend_reason|"
    r"civil_documents_missing| health_difficulties| food_source| food_type|"
    r"food_items_poor| food_items_add| food_items_remove|"
    r"nfi_shelter/shelter_needs|nfi_shelter/household_needs|"
    r"livelihood_source|coping_strategies_livelihood|lending_sources|"
    r"priority_needs|governorate|camp_name|governorate of origin|"
    r"district of origin|demographics/.*|head of household .*|"
    r"marital_status)$")

LABELS = {  # code -> readable level where generic cleaning reads badly
    "aoo": "Return to area of origin", "camp": "Another camp",
    "another_governorate_iraq": "Another governorate - not area of origin",
    "another_governorate_KRI": "Another governorate in the KRI",
    "same_governorate": "Same governorate, different district, out of camp",
    "same_district": "Same district, out of camp", "do_not_know": "Don't know",
    "don’t_know": "Don't know", "dont_know": "Don't know",
    "UN_assistance": "UN assistance", "MOmD": "MoMD", "AFAD": "AFAD",
    "UNHCR": "UNHCR",
}
QUESTIONS = {
    INTENT: "Do you intend to move to a different location in the next 3 months?",
    DEST: "Where do you intend to move to?",
    "communal latrines functional lighting.1":
        "Do communal or public showers have functioning lighting?",
}
KEYS = {INTENT: "move_intention", DEST: "move_destination",
        "communal latrines functional lighting.1": "communal_showers_lighting"}


def slug(s):
    s = s.strip().lower().replace("/", "__")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", s)).strip("_")


def pretty(v):
    if v in LABELS:
        return LABELS[v]
    if v.lower() in ("yes", "no"):
        return v.capitalize()
    s = re.sub(r"\s+", " ", v.replace("_", " ")).strip()
    return s[:1].upper() + s[1:]


def roster(x):
    x = int(x)
    return str(x) if x <= 5 else ("6-10" if x <= 10 else "11+")


def band_age(a):
    a = float(a)
    if a < 18:
        return "Under 18"
    for hi, lab in [(24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
                    (64, "55-64")]:
        if a <= hi:
            return lab
    return "65+"


def band_money(x):  # Iraqi dinar
    for hi, lab in [(0, "0"), (100000, "1-100,000"), (250000, "100,001-250,000"),
                    (500000, "250,001-500,000"), (1000000, "500,001-1,000,000")]:
        if x <= hi:
            return lab
    return "1,000,001+"


def band_days(x):
    x = int(x)
    if x == 0:
        return "0"
    for lo in (1, 6, 11, 16, 21, 26):
        if x <= lo + 4:
            return f"{lo}-{lo + 4}"
    return "31+"


def recode(s):
    """One raw column -> string levels, NA for blank (not asked)."""
    name = s.name
    nn = s.dropna()
    if s.dtype == bool or (len(nn) and set(map(str, nn.unique())) <= {"0.0", "1.0", "0", "1"}
                           and pd.api.types.is_numeric_dtype(s)):
        return s.map(lambda v: NA if pd.isna(v) else str(int(v)))
    if pd.api.types.is_numeric_dtype(s):
        if name in ("hh_member_total_income", "credit_amount"):
            f = band_money
        elif name == "hh_member_days_worked":
            f = band_days
        elif nn.max() <= 11:
            f = lambda v: str(int(v))
        else:
            f = roster
        return s.map(lambda v: NA if pd.isna(v) else f(v))
    return s.map(lambda v: NA if pd.isna(v) else
                 ("Not recorded" if str(v).strip() == "0" else pretty(str(v).strip())))


def load(src=SRC):
    d = pd.read_excel(os.path.join(src, FILE), sheet_name=SHEET)
    # 90 rows are exact re-submissions (same instance id, same answers)
    d = d.loc[~d.drop(columns=["_index"]).duplicated()].reset_index(drop=True)
    g = pd.DataFrame(index=d.index)
    g["governorate"] = d["governorate"].str.strip()
    g["camp_name"] = d["camp_name"].str.strip().map(pretty)
    g["governorate_origin"] = [v if v[:1].isupper() else v.title()
                               for v in d["governorate of origin"].str.strip()]
    g["district_origin"] = d["district of origin"].str.strip().map(pretty)
    g["hoh_sex"] = d["head of household gender"].map(pretty)
    g["hoh_age"] = d["head of household age"].map(band_age)
    g["hoh_marital"] = d["marital_status"].map(pretty)
    g["hh_size"] = d["total population"].map(roster)
    dm = lambda c: d["demographics/" + c]
    g["children_0_5"] = (dm("males_0_5") + dm("females_0_5")).map(roster)
    g["children_6_17"] = sum(dm(f"{s}_{a}") for s in ("males", "females")
                             for a in ("6_11", "12_14", "15_17")).map(roster)
    g["males_18_59"] = dm("males_18_59").map(roster)
    g["females_18_59"] = dm("females_18_59").map(roster)
    g["elderly_60plus"] = (dm("males_60_over") + dm("females_60_over")).map(roster)
    return d, g


GQ = {"governorate": "Governorate of the camp", "camp_name": "Camp name",
      "governorate_origin": "What governorate in Iraq were you living in before your displacement?",
      "district_origin": "What district in Iraq were you living in before your displacement?",
      "hoh_sex": "What is the gender of the head of household?",
      "hoh_age": "What is the age of the head of household? (banded)",
      "hoh_marital": "What is the marital status of the head of household?",
      "hh_size": "Total household members (banded)",
      "children_0_5": "Household members aged 0-5, male + female",
      "children_6_17": "Household members aged 6-17, male + female",
      "males_18_59": "Male household members aged 18-59",
      "females_18_59": "Female household members aged 18-59",
      "elderly_60plus": "Household members aged 60 and over, male + female"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/proxy/reach_irq")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    raw, df = load(a.src)
    GIVEN = list(df.columns)

    # recode every non-dropped column, in questionnaire (column) order
    cols, keys, qtext = [], {}, {}
    for c in raw.columns:
        if DROP.match(c) or raw[c].notna().sum() == 0:
            continue
        k = KEYS.get(c)
        if k is None:
            k = ("move_reason__" + slug(c[len(REASON):])) if c.startswith(REASON) else slug(c)
        assert k not in df.columns, k
        df[k] = recode(raw[c]).to_numpy()
        cols.append(k)
        keys[k] = c
        if c in QUESTIONS:
            qtext[k] = QUESTIONS[c]
        elif "/" in c.strip():
            p, o = c.strip().rsplit("/", 1)
            qtext[k] = f"{p.strip()}: {o.strip()}"
        else:
            qtext[k] = c.strip()
    qtext.update({k: "Why do you intend to leave?: " + keys[k][len(REASON):]
                  for k in cols if k.startswith("move_reason__")})

    forced = [k for k in cols if k in ("move_intention", "move_destination")
              or k.startswith("move_reason__")]

    def eligible(k):
        v = df[k]
        sub = [x for x in v.unique() if x not in (NA, "Not recorded")]
        return 2 <= len(sub) <= 12 and (v != NA).mean() >= 0.05 \
            and (v == "Not recorded").mean() <= 0.5

    cand = [k for k in cols if k not in forced and eligible(k)]
    rng.shuffle(cand)
    chosen = set(forced) | set(cand[:max(0, a.max_items - len(forced))])
    targets = [k for k in cols if k in chosen]      # questionnaire order

    items = {g: {"question": GQ[g], "class": "GIVEN",
                 "values": sorted(df[g].unique()), "gate": None} for g in GIVEN}
    agree, gate_src = [], {}
    for i, t in enumerate(targets):
        asked = (df[t] != NA).to_numpy()
        gate = None
        if t in forced and t != "move_intention":
            # questionnaire relevance: asked only if intention == Yes
            obs = ["Yes"]
            gate = {"parent": "move_intention", "observed_if": obs}
            agree.append(float((df["move_intention"].isin(obs).to_numpy() == asked).mean()))
            gate_src[t] = "questionnaire"
        elif not asked.all():
            best = (-1, None, None)
            for p in GIVEN + targets[:i]:   # parent must come earlier
                pv = df[p].to_numpy()
                tab = pd.crosstab(pv, asked)
                if True not in tab.columns:
                    continue
                rate = tab[True] / tab.sum(axis=1)
                obs = [k for k, r in rate.items() if r >= 0.5 and k != NA]
                if not obs or len(obs) == len(rate):
                    continue
                acc = float((np.isin(pv, obs) == asked).mean())
                if acc > best[0]:
                    best = (acc, p, obs)
            if best[1] is not None:
                gate = {"parent": best[1], "observed_if": sorted(best[2])}
                agree.append(best[0])
                gate_src[t] = "inferred"
        if gate is None and not asked.all():
            df[t] = df[t].replace({NA: "Not asked"})
        else:
            df[t] = df[t].replace({NA: GV})
        vals = sorted(x for x in df[t].unique() if x != GV)
        items[t] = {"question": qtext[t], "class": "PREDICT", "values": vals,
                    "gate": gate}
    # a parent's observed_if must name real levels after recoding
    for t in targets:
        g = items[t]["gate"]
        if g:
            g["observed_if"] = [v for v in g["observed_if"] if v in items[g["parent"]]["values"]]

    order = GIVEN + targets
    n = len(df)
    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    n_dev, n_test = int(0.06 * n), int(0.14 * n)
    n_train = n - n_dev - n_test
    df["respondent_id"] = [f"R{i:06d}" for i in range(n)]
    df["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-reach-irq-1",
                          "description": "REACH Iraq CCCM IDP Camp Profiling VII "
                                         "(Dec 2016-Jan 2017) household proxy, CC BY-IGO"},
              "items": {k: items[k] for k in order},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1, ensure_ascii=False)
    df[["respondent_id"] + order + ["role"]].to_parquet(
        os.path.join(a.out, "respondents.parquet"), index=False)
    ng = sum(1 for k in targets if items[k]["gate"])
    print(f"wrote {a.out}: {n} rows, {len(GIVEN)} GIVEN, {len(targets)} PREDICT "
          f"({len(forced)} forced, {len(cand)} eligible drawn from), {ng} gated "
          f"(agreement median {np.median(agree):.3f}, min {np.min(agree):.3f}), "
          f"{sum(1 for k in targets if 'Not asked' in items[k]['values'])} with 'Not asked'")


if __name__ == "__main__":
    main()
