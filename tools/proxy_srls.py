"""Real-data proxy instrument from the Syrian Refugee Life Study (S-RLS), Jordan.

A held-out final-evaluation lab with the shape of the UNHCR instrument: two
survey rounds of Syrian refugee households in Jordan stacked as independent
cross-sections (2021 phone round, 2022 face-to-face round), a GIVEN block of
round / location / camp / respondent / household-roster columns, and PREDICT
items drawn from the household questionnaire with real skip-logic routing.

Round choice (fixed in advance, by common-item count only): the 2021 phone
round shares 49 variable labels (31 names, many of them admin) with the 2024
face-to-face round, below the 60-item threshold, so the 2022 face-to-face
round is used instead.

Items are harmonised across rounds when their codebook labels agree (same
normalised variable label, compatible value labels), plus a short fixed list
of renamed intention-to-move items (MANUAL). Every other item exists in one
round only and is NA_GATED in the other; its gate is inferred like every
other gate. Key prefixes: x_ both rounds, ps_ 2021 phone only, ff_ 2022 F2F
only.

Gates are inferred as in tools/proxy_gss.py: for each item with not-asked
cells (Stata system missing = not asked), the single earlier item whose values
best separate asked from not-asked. The delivered rows need not satisfy it.

    tools/fetch_srls.sh                         # -> data/raw/srls (hash-checked)
    .venv/bin/python tools/proxy_srls.py --out data/proxy/srls
"""

import argparse
import codecs
import json
import os
import re

import numpy as np
import pandas as pd
import pyreadstat

SRC = os.environ.get("SRLS_SRC", "data/raw/srls")   # tools/fetch_srls.sh
FILES = [("2021", "ps2021_label.dta"), ("2022", "f2f2022_label.dta")]
GV = "NA_GATED"
NA = "\x00i"          # internal not-asked marker, never delivered
MAX_ITEMS = 200
BANDS = [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (10, "6-10")]
ROSTER = [("boys_0_5", 1, 0, 5), ("girls_0_5", 2, 0, 5), ("boys_6_17", 1, 6, 17),
          ("girls_6_17", 2, 6, 17), ("men_18_59", 1, 18, 59),
          ("women_18_59", 2, 18, 59), ("men_60plus", 1, 60, 200),
          ("women_60plus", 2, 60, 200)]
GIVEN = (["wave", "governorate", "camp", "rel_head", "agegrp", "education",
          "marital"] + ["n_" + r[0] for r in ROSTER] + ["origin_gov"])
GIVEN_Q = {"wave": "Survey round", "governorate": "Governorate of residence (Jordan)",
           "camp": "Lives in a formal refugee camp",
           "rel_head": "Respondent is head of household",
           "agegrp": "Respondent age band",
           "education": "Respondent education (type of school last attended; not asked in the 2021 phone round)",
           "marital": "Respondent marital status",
           "origin_gov": "Governorate in Syria lived in, January 2011"}
# renamed across rounds; value labels are checked for agreement like any pair
MANUAL = {"q710": "q1621", "q711a": "q1622a", "q711b": "q1622b"}
# intention-to-move / return items and the items conditional on them (fixed)
FORCE = {"2021": re.compile(r"^(q324|q325a|q325b|q325c|q710|q711a|q711b)$"),
         "2022": re.compile(r"^(q1621|q1622a|q1622b|q1622c|q1623|q1624_\d+|q1624__9\d)$")}
# admin, paradata, contact tracing, free text, coded-other, units, computed
# helpers and loop counters (both rounds)
ADMIN = re.compile(
    r"^(id|id_new|sn|today|date|time|total_duration|duration_.*|q104.*|"
    r"correct_info|current_.*|weekday.*|confirm_workingday|working_time|"
    r"consent_email|q10\d|q11[0-2]|enumerator.*|supervisor.*|final_call_status|"
    r"fr_.*|new_fr_selected|calculated_.*|q202.*|sec_q.*|q206.*|q207.*|group|"
    r"country|governorate|submissiondate.*|q105year|q202year|_merge|breakoff.*|"
    r"imp_q.*|q100\d.*|q250\d.*|partic_in_field|filter_.*|sum_.*|still_members|"
    r"above_.*|select_head.*|head_.*|respondent_same.*|.*_count|.*count_\d+|"
    r"hh_size.*|memberage.*|age_diff.*|indiv_birth.*|list_.*|selected_.*|"
    r".*_counter_.*|child_.*|more_than65.*|fr_bio.*|spouse_selected.*|under19.*|"
    r"q511_calc|count_child.*|numof_.*|s\d+_.*|activity_.*|.*label.*|.*_lab|"
    r".*other.*|.*_c\d+(_\d+)?|.*prob.*|.*confirm.*|.*units.*|.*currency.*|"
    r".*year.*|.*month.*|new_members_size.*|new_child_same.*|agreed.*|"
    r"q612b_calc|q1122_modifed.*|q15013|q1615|.*_else.*|.*_why.*|.*_specify.*|"
    r"q1003.*|q1007.*|q15_1_\d+|.*time.*)$")
# measurement-unit / currency / clock fields, by codebook label
ADMIN_LABEL = re.compile(r"currency|\bunits?\b|time period|\(am/pm\)|enter \((mm|hh)\)",
                         re.I)
# sources of the GIVEN block and the respondent / household rosters
DROP = {"2021": re.compile(r"^(q30\d.*|q31\d.*|q32[0-3].*|q401|q401a|"
                           r"q40[3-9]_.*|q41[01]_.*|q410__96_.*|q511|q701a|q701b)$"),
        "2022": re.compile(r"^(q401b|q501|q50[4-9]_.*|q510_.*|q51[01]_.*|pre_r_.*|"
                           r"present_.*|hhr_new.*|location_moved.*|"
                           r"not_member_reason.*|newmember_.*|q5_2_.*|q514|"
                           r"q1601a|q1601b|q611)$")}
MISSING_CODES = {-99, -98, -97, -88}


def _utf8_replace(b, errors="strict"):
    return codecs.utf_8_decode(b, "replace", True)


_U8 = codecs.lookup("utf-8")
codecs.register(lambda n: codecs.CodecInfo(_U8.encode, _utf8_replace, name="utf8r")
                if n == "utf8r" else None)


def read_dta(path):
    """Codes as numbers, plus pyreadstat metadata for labels.

    Some string fields hold UTF-8 Arabic truncated mid-character at 254
    bytes, which both pyreadstat and pandas refuse; decode with replacement.
    """
    with pd.read_stata(path, convert_categoricals=False, iterator=True) as r:
        r._ensure_open()
        r._encoding = "utf8r"
        d = r.read()
    _, m = pyreadstat.read_dta(path, metadataonly=True)
    return d, m


def norm_label(s):
    s = (s or "").lower()
    s = re.sub(r"\$\{[^}]*\}", "", s)
    s = re.sub(r"^\s*q[\d_.a-z]*[.:]\s*", "", s)
    return re.sub(r"[^a-z]", "", s)


def norm_val(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def num(v):
    return pd.to_numeric(v, errors="coerce")


def band_count(n):
    for hi, lab in BANDS:
        if n <= hi:
            return lab
    return "11+"


def band_age(a):
    if not np.isfinite(a):
        return "Not recorded"
    for hi, lab in [(24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
                    (64, "55-64")]:
        if a <= hi:
            return lab
    return "65+"


def vlab(meta, var):
    return meta.variable_value_labels.get(var, {})


def code_str(x):
    f = float(x)
    return str(int(f)) if f == int(f) else repr(f)


def to_levels(col, labels):
    """Value-labelled strings; system missing -> internal not-asked marker."""
    out = np.empty(len(col), dtype=object)
    v = num(col).to_numpy()
    for i, x in enumerate(v):
        if not np.isfinite(x):
            out[i] = NA
        else:
            lab = labels.get(x, labels.get(int(x)) if x == int(x) else None)
            c = code_str(x)
            lab = (lab or "").strip()
            out[i] = f"{c}. {lab}" if lab and lab.lower() != "null" else c
    return out


def given_block(wave, d, m):
    n = len(d)
    g = {"wave": [wave] * n}
    gl = vlab(m, "q301b") or vlab(m, "q1601b")
    g["governorate"] = [gl.get(x, code_str(x)) if np.isfinite(x) else "Not recorded"
                        for x in num(d["governorate"])]
    if wave == "2021":
        camp = num(d["q511"]) == 4                  # own/rent: formal refugee camp
        head = num(d["q401a"]).map({1: "Head of household", 0: "Not head of household"})
        age = num(d["q202a"])
        edu = ["Not asked (phone round)"] * n
        sex = [(num(d["q202c"]), age)]
        sex += [(num(d[f"q404_{i}"]), num(d[f"q405_{i}"])) for i in range(1, 24)]
        oc, og = num(d["q701a"]), num(d["q701b"])
    else:
        camp = num(d["q1615"]).notna() | (num(d["q611"]) == 4)   # camp-only item
        sh = num(d["select_head_of_household"])
        head = pd.Series(np.where(sh == -98, "Head of household", "Not head of household"))
        head[sh.isna().to_numpy()] = "Not recorded"
        age = num(d["calculated_fr_age"])
        el = vlab(m, "q401b")
        edu = [el.get(x, code_str(x)) if np.isfinite(x) else "Not recorded"
               for x in num(d["q401b"])]
        sex = [(num(d["q202c"]), age)]
        k = sorted({int(c.split("_")[1]) for c in d.columns if re.match(r"^q505_\d+$", c)})
        sex += [(num(d[f"q504_{i}"]), num(d[f"q505_{i}"])) for i in k if f"q504_{i}" in d]
        oc, og = num(d["q1601a"]), num(d["q1601b"])
    g["camp"] = np.where(camp, "Camp", "Non-camp").tolist()
    g["rel_head"] = head.fillna("Not recorded").tolist()
    g["agegrp"] = [band_age(a) for a in age]
    g["education"] = edu
    ml = vlab(m, "q202b")
    g["marital"] = [ml.get(x, code_str(x)) if np.isfinite(x) else "Not recorded"
                    for x in num(d["q202b"])]
    for name, s, lo, hi in ROSTER:
        cnt = np.zeros(n, dtype=int)
        for sx, ag in sex:
            cnt += ((sx == s) & (ag >= lo) & (ag <= hi)).to_numpy()
        g["n_" + name] = [band_count(c) for c in cnt]
    ogl = vlab(m, "q701b") or vlab(m, "q1601b")
    org = []
    for c, gv in zip(oc, og):
        if not np.isfinite(c):
            org.append("Not recorded")
        elif c != 2:
            org.append("Not living in Syria")
        elif np.isfinite(gv) and 40 <= gv <= 53:
            org.append(ogl[gv])
        elif np.isfinite(gv) and gv in (-99, -97):
            org.append(ogl[gv].split("[")[0].strip())
        else:
            org.append("Not recorded")
    g["origin_gov"] = org
    return pd.DataFrame(g)


def item_vars(wave, d, m):
    """Source variables that can become PREDICT items, in questionnaire order."""
    cols = list(m.column_names)
    types = m.readstat_variable_types
    lab = {c: norm_label(m.column_names_to_labels.get(c)) for c in cols}
    stem = {c: re.sub(r"_\d+$", "", c) for c in cols}
    sibs = {}
    for c in cols:
        if stem[c] != c:
            sibs.setdefault(stem[c], []).append(c)
    keep = []
    for c in cols:
        if (types[c] == "string" or ADMIN.match(c) or DROP[wave].match(c)
                or ADMIN_LABEL.search(m.column_names_to_labels.get(c) or "")):
            continue
        # repeat-group instance: the same question asked per roster member,
        # job, transfer, move... (siblings carry the same label)
        s = sibs.get(stem[c], [])
        if len(s) >= 2 and len({re.sub(r"\d", "", lab[x]) for x in s}) == 1:
            continue
        # select-multiple parent (its options are kept as binary items)
        if not vlab(m, c) and (f"{c}_1" in m.column_names or f"{c}__96" in m.column_names):
            continue
        keep.append(c)
    return keep


def harmonise(m1, v1, m2, v2):
    """Pairs (2021 var, 2022 var) asking the same question with compatible codes."""
    l2 = {}
    for c in v2:
        l2.setdefault(norm_label(m2.column_names_to_labels.get(c)), []).append(c)
    l1n = {}
    for c in v1:
        k = norm_label(m1.column_names_to_labels.get(c))
        l1n[k] = l1n.get(k, 0) + 1
    pairs = {}
    for c in v1:
        k = norm_label(m1.column_names_to_labels.get(c))
        if c in MANUAL:
            cand = [MANUAL[c]] if MANUAL[c] in v2 else []
        elif not k or l1n[k] > 1:
            continue
        else:
            cand = l2.get(k, [])
        if len(cand) != 1 or cand[0] in pairs.values():
            continue
        a = {x: norm_val(y) for x, y in vlab(m1, c).items()}
        b = {x: norm_val(y) for x, y in vlab(m2, cand[0]).items()}
        ov = set(a) & set(b)
        if (not a and not b) or (ov and all(a[x] == b[x] for x in ov)):
            pairs[c] = cand[0]
    return pairs


def substantive(levels):
    out = []
    for x in levels:
        if x == NA:
            continue
        try:
            if float(x.split(".")[0] if ". " in x else x) in MISSING_CODES:
                continue
        except ValueError:
            pass
        out.append(x)
    return out


def load(src=SRC):
    raw = [(w,) + read_dta(os.path.join(src, f)) for w, f in FILES]
    (w1, d1, m1), (w2, d2, m2) = raw
    v1, v2 = item_vars(w1, d1, m1), item_vars(w2, d2, m2)
    pairs = harmonise(m1, v1, m2, v2)
    inv = {b: a for a, b in pairs.items()}
    n1, n2 = len(d1), len(d2)
    cols, meta = {}, {}
    pos2 = {c: i / len(m2.column_names) for i, c in enumerate(m2.column_names)}
    pos1 = {c: i / len(m1.column_names) for i, c in enumerate(m1.column_names)}
    for c in v2:
        lab2 = to_levels(d2[c], vlab(m2, c))
        if c in inv:
            a = inv[c]
            lab1 = to_levels(d1[a], {**vlab(m2, c), **vlab(m1, a)})
            key = "x_" + c
            q = m2.column_names_to_labels.get(c) or c
            force = bool(FORCE["2021"].match(a) or FORCE["2022"].match(c))
            src_ = f"2021 {a} / 2022 {c}"
        else:
            lab1 = np.full(n1, NA, dtype=object)
            key = "ff_" + c
            q = m2.column_names_to_labels.get(c) or c
            force = bool(FORCE["2022"].match(c))
            src_ = f"2022 {c}"
        cols[key] = np.concatenate([lab1, lab2])
        meta[key] = {"question": f"{q.strip()} [{src_}]", "pos": pos2[c], "force": force}
    for c in v1:
        if c in pairs:
            continue
        key = "ps_" + c
        cols[key] = np.concatenate([to_levels(d1[c], vlab(m1, c)), np.full(n2, NA, dtype=object)])
        q = m1.column_names_to_labels.get(c) or c
        meta[key] = {"question": f"{q.strip()} [2021 {c}]", "pos": pos1[c],
                     "force": bool(FORCE["2021"].match(c))}
    g = pd.concat([given_block(w1, d1, m1), given_block(w2, d2, m2)], ignore_index=True)
    items = pd.DataFrame(cols)
    return g, items, meta, len(pairs)


def best_gate(asked, parents, df):
    best = (-1.0, None, None)
    for p in parents:
        codes, uniq = pd.factorize(df[p])
        tot = np.bincount(codes, minlength=len(uniq))
        yes = np.bincount(codes, weights=asked, minlength=len(uniq))
        rate = yes / tot
        bad = np.isin(uniq, [NA, GV, "Not asked"])
        okl = (rate >= 0.5) & ~bad
        if not okl.any() or okl.sum() == len(uniq):
            continue
        acc = float((okl[codes] == asked).mean())
        if acc > best[0]:
            best = (acc, p, [str(u) for u in uniq[okl]])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/proxy/srls")
    ap.add_argument("--max-items", type=int, default=MAX_ITEMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g, items, meta, n_pairs = load(a.src)
    n = len(g)

    # eligible: 2-12 substantive levels, asked of >= 5% of rows; forced
    # intention items need only 2 levels
    forced, cand = [], []
    for c in sorted(items.columns):
        v = items[c]
        k = len(set(substantive(v.unique())))
        if k < 2:
            continue
        if meta[c]["force"]:
            forced.append(c)
        elif k <= 12 and (v != NA).mean() >= 0.05:
            cand.append(c)
    rng.shuffle(cand)
    targets = forced + cand[:max(0, a.max_items - len(forced))]
    targets.sort(key=lambda c: (meta[c]["pos"], c))   # questionnaire order

    df = pd.concat([g, items[targets]], axis=1)
    out_items = {}
    for c in GIVEN:
        out_items[c] = {"question": GIVEN_Q.get(c, c.replace("n_", "Household members: ")
                                                .replace("_", " ")),
                        "class": "GIVEN", "values": sorted(df[c].unique()), "gate": None}
    agree = []
    for i, t in enumerate(targets):
        asked = (df[t] != NA).to_numpy()
        gate = None
        if not asked.all():
            acc, p, obs = best_gate(asked.astype(float), GIVEN + targets[:i], df)
            if p is not None:
                gate = {"parent": p, "observed_if": sorted(obs)}
                agree.append(acc)
        if gate is None and not asked.all():
            df[t] = df[t].replace({NA: "Not asked"})
        else:
            df[t] = df[t].replace({NA: GV})
        vals = sorted(x for x in df[t].unique() if x != GV)
        out_items[t] = {"question": meta[t]["question"], "class": "PREDICT",
                        "values": vals, "gate": gate}

    order = GIVEN + targets
    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    n_dev, n_test = int(0.06 * n), int(0.14 * n)
    n_train = n - n_dev - n_test
    df["respondent_id"] = [f"S{i:06d}" for i in range(n)]
    df["role"] = ["TRAIN"] * n_train + ["DEV"] * n_dev + ["TEST"] * n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-srls-1",
                          "description": "Syrian Refugee Life Study (Jordan) 2021 phone "
                                         "+ 2022 face-to-face rounds, stacked "
                                         "cross-sections; held-out final lab"},
              "items": {k: out_items[k] for k in order},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    df[["respondent_id"] + order + ["role"]].to_parquet(
        os.path.join(a.out, "respondents.parquet"), index=False)
    ng = sum(1 for k in targets if out_items[k]["gate"])
    nx = sum(1 for k in targets if k.startswith("x_"))
    print(f"wrote {a.out}: {n} rows, {len(GIVEN)} GIVEN, {len(targets)} PREDICT "
          f"({len(forced)} forced intention items, {nx} in both rounds; "
          f"{n_pairs} harmonised pairs, {len(cand)} eligible drawn-from), "
          f"{ng} gated (agreement median {np.median(agree):.2f}, "
          f"min {np.min(agree):.2f})")


if __name__ == "__main__":
    main()
