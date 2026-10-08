"""Harder, more realistic practice data than make_sandbox.py.

The official sandbox makes GIVEN almost useless (one latent trait, no
demographic effects) and obeys every gate exactly. The real leaderboard says
the opposite: GIVEN-conditional models gain +0.2-0.5 skill over the crowd
marginal, and UNHCR gates agree with the data only 0.29-0.98 of the time.

This generator adds, per PREDICT item:
  * main effects of every GIVEN column (one strong "country-like" column),
  * a few pairwise GIVEN interactions,
  * a latent trait correlated with the GIVEN block,
  * dependence on the parent's answer (not only through the gate),
  * noisy gates: per item agreement drawn from [0.55, 0.99],
  * UNHCR-style wave coding: for gated items with a "0" level, rows with
    wave == "1" carry "0" where the routing skipped them.

    python tools/sim2.py --schema raw/repo/data/unhcr.json --out sim2/unhcr --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "raw", "repo"))
from make_sandbox import (assign_roles, generated_items, load_config,  # noqa: E402
                          load_schema, _generation_order)


def simulate(schema, seed):
    rng = np.random.default_rng(seed)
    items = schema["items"]
    n = schema["dataset"]["n_rows"]
    gv = schema["gated_value"]
    names = generated_items(schema)
    given = [k for k in names if items[k]["class"] == "GIVEN"]
    order = _generation_order(schema, names)

    cols, codes = {}, {}
    # GIVEN block: a "country" column first (most levels among first few),
    # others mildly dependent on it.
    anchor = given[0] if given else None
    for k in order:
        rec = items[k]
        if rec["class"] != "GIVEN":
            continue
        vals = list(rec["values"])
        w = len(vals)
        if k == anchor or anchor not in codes:
            p = rng.dirichlet(np.full(w, 1.5))
            idx = rng.choice(w, size=n, p=p)
        else:
            a = codes[anchor]
            na = int(a.max()) + 1
            table = rng.dirichlet(np.full(w, 1.0), size=na)
            u = rng.random(n)
            idx = (table[a].cumsum(1) > u[:, None]).argmax(1)
        v = np.asarray(vals, dtype=object)[idx]
        g = rec.get("gate")
        if g:
            par = np.asarray(cols[g["parent"]], dtype=object)
            skipped = ~np.isin(par, g["observed_if"])
            v = np.where(skipped, gv, v)
            idx = np.where(skipped, w, idx)
        cols[k], codes[k] = v, idx

    onehots = []
    for k in given:
        c = codes[k]
        onehots.append(np.eye(int(c.max()) + 1)[c])
    X = np.concatenate(onehots, 1) if onehots else np.zeros((n, 1))
    # latent trait correlated with GIVEN
    trait = X @ rng.normal(0, 0.4, X.shape[1]) + rng.normal(size=n)
    trait = (trait - trait.mean()) / trait.std()
    trait2 = rng.normal(size=n)

    for k in order:
        rec = items[k]
        if rec["class"] != "PREDICT":
            continue
        vals = list(rec["values"])
        w = len(vals)
        logits = np.log(rng.dirichlet(np.full(w, 0.7)) + 1e-6)[None, :].repeat(n, 0)
        # main effects; anchor column strong
        for gi, k2 in enumerate(given):
            c = codes[k2]
            scale = 1.2 if k2 == anchor else rng.choice([0.0, 0.2, 0.5])
            eff = rng.normal(0, scale, (int(c.max()) + 1, w))
            logits += eff[c]
        # a couple of interactions
        for _ in range(2):
            if len(given) < 2:
                break
            a, b = rng.choice(len(given), 2, replace=False)
            ca, cb = codes[given[a]], codes[given[b]]
            key = ca * (int(cb.max()) + 1) + cb
            eff = rng.normal(0, 0.6, (int(key.max()) + 1, w))
            logits += eff[key]
        logits += np.outer(trait, rng.normal(0, 1.0, w))
        logits += np.outer(trait2, rng.normal(0, 0.7, w))
        g = rec.get("gate")
        if g and items[g["parent"]]["class"] == "PREDICT":
            pc = codes[g["parent"]]
            eff = rng.normal(0, 0.8, (int(pc.max()) + 1, w))
            logits += eff[pc]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        idx = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        v = np.asarray(vals, dtype=object)[idx]
        if g:
            par = np.asarray(cols[g["parent"]], dtype=object)
            should_skip = ~np.isin(par, g["observed_if"])
            agree = rng.uniform(0.55, 0.99)
            flip = rng.random(n) > agree
            # flip both directions, but mostly "asked although routed away"
            skipped = np.where(flip, ~should_skip & (rng.random(n) < 0.3),
                               should_skip)
            if "wave" in cols and "0" in vals:
                wave1 = np.asarray(cols["wave"]) == "1"
                v = np.where(skipped & wave1, "0", np.where(skipped, gv, v))
                idx = np.where(skipped & wave1, vals.index("0"),
                               np.where(skipped, w, idx))
            else:
                v = np.where(skipped, gv, v)
                idx = np.where(skipped, w, idx)
        cols[k], codes[k] = v, idx

    frame = pd.DataFrame({k: cols[k] for k in names})
    frame.insert(0, "respondent_id", ["R%06d" % i for i in range(1, n + 1)])
    return frame.astype(object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "raw", "repo", "config.yml"))
    a = ap.parse_args()
    schema = load_schema(a.schema, load_config(a.config))
    frame = assign_roles(schema, simulate(schema, a.seed))
    os.makedirs(a.out, exist_ok=True)
    frame.to_parquet(os.path.join(a.out, "respondents.parquet"), index=False)
    with open(os.path.join(a.out, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False)
    print("wrote", a.out, frame.shape)


if __name__ == "__main__":
    main()
