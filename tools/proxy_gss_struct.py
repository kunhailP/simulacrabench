"""Structural labs on the real GSS proxy: shared events that drive many items at once.

Both competition tracks we target have one event that sets the missing-type
outcome of many items together, as documented in their schemas:

  wb      World Bank: 69 of 76 PREDICT items carry "Not answered". In a single-
          session assessment that is attrition: whoever leaves stops answering
          everything after that point. Here the dropout hazard is driven by the
          *real* interviewer ratings of cooperation and comprehension (coop,
          comprend), so its dependence on the GIVEN block is the one GSS has.
          After the dropout position every item reads "Not answered".
  branch  UNHCR: three branches hang off one hidden item (s3_11, 73 children),
          gate agreement has median 0.83 and range 0.29-0.98, the branches
          overlap, and in round 1 multi-select items were written 0 instead of
          the sentinel. Here a real three-level GSS item is the hidden parent;
          always-asked real binary items become the branch children, observed
          when the respondent is in the branch, NA_GATED otherwise, then the
          routing is broken per item to the documented agreement and wave-1
          rows get "no" instead of the sentinel in the first branch.

Answers and the GIVEN block are real; only the routing/attrition mechanism is
imposed, and it is the one the competition schema documents.

    .venv/bin/python tools/proxy_gss_struct.py --kind wb --out data/proxy/gss_wb
    .venv/bin/python tools/proxy_gss_struct.py --kind branch --out data/proxy/gss_branch
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

GV = "NA_GATED"
SRC_DIR = "data/proxy/gss"
NA = "Not answered"


def binary_always_asked(df, items, min_minor=0.08):
    """PREDICT items with exactly two substantive levels (an 'N. label' code)
    whose rarer level is at least min_minor among those asked. An item that GSS
    already routes keeps its own NA_GATED; the branch gate is laid on top."""
    out = []
    for k, r in items.items():
        if r["class"] != "PREDICT" or k == "ballot":
            continue
        v = df[k]
        sub = [x for x in r["values"] if x[:1].isdigit()]
        if len(sub) != 2:
            continue
        asked = v != GV
        share = v[v.isin(sub)].value_counts(normalize=True)
        if len(share) == 2 and share.min() >= min_minor and v[asked].isin(sub).mean() > 0.9:
            out.append(k)
    return out


def make_wb(df, schema, rng, base=0.10, slope=1.1, keep_gate=False):
    items = schema["items"]
    targets = [k for k, r in items.items() if r["class"] == "PREDICT" and k != "ballot"]
    rng.shuffle(targets)                              # an assessment order
    coop = df["_coop"].str[:1]
    comp = df["_comprend"].str[:1]
    # risk score from the real paradata; rows without a rating get the mean risk
    c = pd.to_numeric(coop, errors="coerce")
    m = pd.to_numeric(comp, errors="coerce")
    risk = (c - 1).fillna((c - 1).mean()) + (m - 1).fillna((m - 1).mean())
    p_drop = 1 - (1 - base) * np.exp(-slope * risk.to_numpy())   # P(leave at some point)
    leaves = rng.random(len(df)) < p_drop
    # where they leave: uniform over the sequence, earlier for the least cooperative
    u = rng.random(len(df)) ** (1 + risk.to_numpy())
    pos = np.where(leaves, (u * len(targets)).astype(int), len(targets))
    for i, t in enumerate(targets):
        gone = pos <= i
        if keep_gate:                     # a routed-out cell stays NA_GATED after dropout
            gone &= (df[t] != GV).to_numpy()
        df.loc[gone, t] = NA
        if NA not in items[t]["values"]:
            items[t]["values"] = items[t]["values"] + [NA]
    order = [k for k in items if items[k]["class"] == "GIVEN"] + ["ballot"] + targets
    schema["items"] = {k: items[k] for k in order}
    rate = (pos < len(targets)).mean()
    return f"attrition: {rate:.1%} leave, mean answered share {np.mean(np.minimum(pos, len(targets))) / len(targets):.2f}"



def make_wb2(df, schema, rng, tracer_share=0.4, tracer_resp=0.4):
    """World Bank layout: a single session (assessment + pre/post survey) with
    sequential attrition, then a separate tracer follow-up. Session items read
    "Not answered" after the dropout point (make_wb); tracer items read "Not
    answered" for everyone who did not answer the tracer, whatever they did in
    the session (response weakly tied to the real cooperation rating)."""
    items = schema["items"]
    msg = make_wb(df, schema, rng)
    items = schema["items"]
    targets = [k for k in items if items[k]["class"] == "PREDICT" and k != "ballot"]
    n_tr = int(tracer_share * len(targets))
    tracer = targets[-n_tr:]
    session = targets[:-n_tr]
    # tracer items: restore the real answers, then blank the non-respondents
    src = pd.read_parquet(os.path.join(SRC_DIR, "respondents.parquet"))
    c = pd.to_numeric(src["_coop"].str[:1], errors="coerce").fillna(1.3).to_numpy()
    p_resp = np.clip(tracer_resp * np.exp(-0.5 * (c - 1)), 0.05, 0.95)
    resp = rng.random(len(df)) < p_resp
    for t in tracer:
        df[t] = src[t].to_numpy()
        df.loc[~resp, t] = NA
    return msg + f"; tracer block {len(tracer)} items, {resp.mean():.0%} answer it; session {len(session)} items"


def pick_parent(df, items, cands):
    """A real 3-level item (2 substantive + a third substantive or DK) whose
    rarest level is 8-25%: the hidden routing item, like s3_11."""
    best = None
    for k, r in items.items():
        if r["class"] != "PREDICT" or r.get("gate") or k in cands or k == "ballot":
            continue
        v = df[k]
        if v.isin(["i", GV]).any():
            continue
        share = v.value_counts(normalize=True)
        top = share.iloc[:3]
        if len(share) < 3 or top.sum() < 0.95:
            continue
        if 0.08 <= top.iloc[2] <= 0.25 and (best is None or top.iloc[2] < best[1]):
            best = (k, top.iloc[2], list(top.index))
    return best


def make_branch(df, schema, rng, n_children=60):
    items = schema["items"]
    cands = binary_always_asked(df, items)
    par = pick_parent(df, items, set(cands))
    if par is None:
        raise SystemExit("no parent")
    p, _, levels = par
    rng.shuffle(cands)
    kids = cands[:n_children]
    # branch sizes in the s3_11 proportions: return 19+... (s7a), no-return (s7b), undecided (s7c)
    nA, nB = int(0.45 * len(kids)), int(0.35 * len(kids))
    branch = {levels[2]: kids[:nA], levels[0]: kids[nA:nA + nB], levels[1]: kids[nA + nB:]}
    pv = df[p].to_numpy()
    # gate agreement per child: documented median 0.83, range 0.29-0.98
    agree = np.clip(rng.beta(5, 1.2, len(kids)), 0.29, 0.98)
    ag = dict(zip(kids, agree))
    w1 = (df["year"] == "2016").to_numpy()
    df["wave"] = np.where(w1, "1", "2")
    first = levels[2]
    for lev, ks in branch.items():
        member = pv == lev
        for k in ks:
            a = ag[k]
            # break routing: flip membership for a share of rows so that the
            # agreement between gate and who holds a value is about `a`
            flip = rng.random(len(df)) < (1 - a)
            held = member ^ flip
            sub = [x for x in items[k]["values"] if x[:1].isdigit()]
            no = next((x for x in sub if "no" in x.lower()), sub[-1])
            v = df[k].to_numpy(dtype=object).copy()
            orig = v == GV                            # GSS's own routing stays
            if lev == first:
                v[~held & w1 & ~orig] = no            # round 1 wrote 0 for the skipped
                v[~held & ~w1] = GV
            else:
                v[~held] = GV
            df[k] = v
            items[k]["gate"] = {"parent": p, "observed_if": [lev]}
            vals = [x for x in items[k]["values"] if x != GV]
            items[k]["values"] = vals
    items["wave"] = {"question": "survey round", "class": "GIVEN", "values": ["1", "2"], "gate": None}
    given = [k for k in items if items[k]["class"] == "GIVEN"]
    rest = [k for k in items if items[k]["class"] != "GIVEN" and k not in (p,)]
    order = given + [p] + [k for k in rest if k != "wave"]
    # the hidden parent has to precede its children; drop original gates onto it
    schema["items"] = {k: items[k] for k in order}
    return (f"parent {p} levels {levels} (rarest {par[1]:.1%}), children "
            f"{len(kids)} = {[len(v) for v in branch.values()]}, agreement median "
            f"{np.median(agree):.2f} range {agree.min():.2f}-{agree.max():.2f}")


def make_subset(df, schema, rng, n_items):
    """Same respondents, n_items of the PREDICT items (seeded): isolates how a
    method's behaviour on an item depends on how many other items there are."""
    items = schema["items"]
    preds = [k for k, r in items.items() if r["class"] == "PREDICT" and k != "ballot"]
    ok = [k for k in preds if not items[k].get("gate")
          or items[items[k]["gate"]["parent"]]["class"] == "GIVEN"
          or items[k]["gate"]["parent"] == "ballot"]
    pick = set(rng.choice(ok, size=min(n_items - 1, len(ok)), replace=False))
    keep = [k for k, r in items.items() if r["class"] == "GIVEN" or k == "ballot" or k in pick]
    schema["items"] = {k: items[k] for k in keep}
    return f"subset: {len(keep) - sum(items[k]['class'] == 'GIVEN' for k in keep)} PREDICT items"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/proxy/gss")
    ap.add_argument("--kind", choices=["wb", "wb2", "wb3", "branch", "subset"], required=True)
    ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    schema = json.load(open(os.path.join(a.src, "schema.json")))
    df = pd.read_parquet(os.path.join(a.src, "respondents.parquet"))
    global SRC_DIR
    SRC_DIR = a.src
    if a.kind == "wb3":
        msg = make_wb(df, schema, rng, keep_gate=True)
    elif a.kind == "wb2":
        msg = make_wb2(df, schema, rng)
    elif a.kind == "subset":
        msg = make_subset(df, schema, rng, a.n_items)
    else:
        msg = (make_wb if a.kind == "wb" else make_branch)(df, schema, rng)
    keep = ["respondent_id"] + list(schema["items"]) + ["role"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    df[keep].to_parquet(os.path.join(a.out, "respondents.parquet"), index=False)
    print(f"wrote {a.out}: {msg}")


if __name__ == "__main__":
    main()
