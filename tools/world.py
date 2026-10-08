"""Simulated survey worlds with a computable oracle.

A World fixes every parameter first (its own RNG), then samples respondents
with another. Because the parameters are fixed, the Bayes-optimal forecast
P(y_j | GIVEN = x) can be estimated for any respondent by re-drawing the
latent part (traits, answers, routing noise) many times with x held fixed.
That gives the headroom a method has left: oracle skill minus method skill.

Knobs (scenario bundle):
  given_scale   strength of GIVEN main effects (anchor column x2)
  inter_scale   strength of pairwise GIVEN interactions
  trait_scale   loading on a latent trait correlated with GIVEN
  trait2_scale  loading on an independent latent trait
  parent_scale  child answer depends on parent answer
  cell_scale    low-rank high-order cell effects on the top-3 GIVEN cell
  agree_lo/hi   per-item routing agreement range

    python tools/world.py --schema raw/repo/data/unhcr.json --scenario base --out data/worlds/base/unhcr
writes respondents.parquet (+ role) like make_sandbox, and oracle.npz holding
the oracle probabilities of the DEV rows for phase-1 evaluation.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))
from make_sandbox import (ROLE_COUNTS, assign_roles, generated_items,  # noqa: E402
                          load_config, load_schema, _generation_order)

SCENARIOS = {
    "base":     dict(given_scale=1.0, inter_scale=0.6, trait_scale=1.0, trait2_scale=0.7,
                     parent_scale=0.8, cell_scale=0.0, agree_lo=0.55, agree_hi=0.99),
    "weakx":    dict(given_scale=0.3, inter_scale=0.2, trait_scale=1.2, trait2_scale=1.0,
                     parent_scale=0.8, cell_scale=0.0, agree_lo=0.55, agree_hi=0.99),
    "strongx":  dict(given_scale=1.8, inter_scale=1.0, trait_scale=0.5, trait2_scale=0.3,
                     parent_scale=0.8, cell_scale=0.0, agree_lo=0.55, agree_hi=0.99),
    "cells":    dict(given_scale=0.8, inter_scale=0.3, trait_scale=0.8, trait2_scale=0.5,
                     parent_scale=0.8, cell_scale=1.0, agree_lo=0.55, agree_hi=0.99),
    "cleangate": dict(given_scale=1.0, inter_scale=0.6, trait_scale=1.0, trait2_scale=0.7,
                      parent_scale=0.8, cell_scale=0.0, agree_lo=0.97, agree_hi=1.0),
}


class World:
    def __init__(self, schema, seed, **k):
        self.schema, self.k = schema, k
        rng = np.random.default_rng([seed, 1])
        items = schema["items"]
        self.names = generated_items(schema)
        self.given = [n for n in self.names if items[n]["class"] == "GIVEN"]
        self.order = _generation_order(schema, self.names)
        self.gv = schema["gated_value"]
        self.anchor = self.given[0] if self.given else None
        # GIVEN block law
        self.gpar = {}
        for g in self.order:
            if items[g]["class"] != "GIVEN":
                continue
            w = len(items[g]["values"])
            if g == self.anchor:
                self.gpar[g] = rng.dirichlet(np.full(w, 1.5))
            else:
                na = len(items[self.anchor]["values"]) + 1
                self.gpar[g] = rng.dirichlet(np.full(w, 1.0), size=na)
        self.card = {g: len(items[g]["values"]) + 1 for g in self.given}
        dimx = sum(self.card.values())
        self.trait_w = rng.normal(0, 0.4, dimx)
        top = self.given[:3]
        self.top = top
        ncell = int(np.prod([self.card[g] for g in top])) if top else 1
        self.cell_rank = 3
        self.F = rng.normal(0, 1.0, (ncell, self.cell_rank))
        # PREDICT item parameters
        self.par = {}
        for t in self.order:
            rec = items[t]
            if rec["class"] != "PREDICT":
                continue
            w = len(rec["values"])
            P = {"base": np.log(rng.dirichlet(np.full(w, 0.7)) + 1e-6)}
            P["main"] = {}
            for g in self.given:
                sc = k["given_scale"] * (1.2 if g == self.anchor else rng.choice([0.0, 0.2, 0.5]))
                P["main"][g] = rng.normal(0, sc, (self.card[g], w))
            P["inter"] = []
            for _ in range(2):
                if len(self.given) < 2:
                    break
                a, b = rng.choice(len(self.given), 2, replace=False)
                ga, gb = self.given[a], self.given[b]
                P["inter"].append((ga, gb, rng.normal(0, k["inter_scale"],
                                                      (self.card[ga] * self.card[gb], w))))
            P["trait"] = rng.normal(0, k["trait_scale"], w)
            P["trait2"] = rng.normal(0, k["trait2_scale"], w)
            P["cell"] = rng.normal(0, k["cell_scale"], (self.cell_rank, w))
            g = rec.get("gate")
            if g and items[g["parent"]]["class"] == "PREDICT":
                wp = len(items[g["parent"]]["values"]) + (1 if items[g["parent"]].get("gate") else 0)
                P["parent"] = rng.normal(0, k["parent_scale"], (wp, w))
            if g:
                P["agree"] = rng.uniform(k["agree_lo"], k["agree_hi"])
            self.par[t] = P

    # -- GIVEN ------------------------------------------------------------
    def sample_given(self, n, rng):
        items = self.schema["items"]
        codes, cols = {}, {}
        for g in self.order:
            if items[g]["class"] != "GIVEN":
                continue
            vals = items[g]["values"]
            w = len(vals)
            if g == self.anchor:
                idx = rng.choice(w, size=n, p=self.gpar[g])
            else:
                tab = self.gpar[g][codes[self.anchor]]
                idx = (tab.cumsum(1) > rng.random((n, 1))).argmax(1)
            v = np.asarray(vals, dtype=object)[idx]
            gate = items[g].get("gate")
            if gate:
                skipped = ~np.isin(np.asarray(cols[gate["parent"]], dtype=object), gate["observed_if"])
                v = np.where(skipped, self.gv, v)
                idx = np.where(skipped, w, idx)
            codes[g], cols[g] = idx, v
        return codes, cols

    # -- PREDICT given GIVEN ---------------------------------------------
    def sample_predict(self, codes, cols, rng):
        items = self.schema["items"]
        n = len(next(iter(codes.values())))
        X = np.concatenate([np.eye(self.card[g])[codes[g]] for g in self.given], 1)
        trait = X @ self.trait_w + rng.normal(size=n)
        trait = trait / np.sqrt(1 + (self.trait_w ** 2).sum() * 0.2)
        trait2 = rng.normal(size=n)
        ck = np.zeros(n, np.int64)
        for g in self.top:
            ck = ck * self.card[g] + codes[g]
        pcodes = {}
        out = {}
        for t in self.order:
            rec = items[t]
            if rec["class"] != "PREDICT":
                continue
            P = self.par[t]
            vals = list(rec["values"])
            w = len(vals)
            L = np.tile(P["base"], (n, 1))
            for g in self.given:
                L += P["main"][g][codes[g]]
            for ga, gb, eff in P["inter"]:
                L += eff[codes[ga] * self.card[gb] + codes[gb]]
            L += np.outer(trait, P["trait"]) + np.outer(trait2, P["trait2"])
            L += (self.F[ck] @ P["cell"])
            g = rec.get("gate")
            if "parent" in P:
                L += P["parent"][pcodes[g["parent"]]]
            pr = np.exp(L - L.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            idx = (pr.cumsum(1) > rng.random((n, 1))).argmax(1)
            v = np.asarray(vals, dtype=object)[idx]
            if g:
                par = out[g["parent"]] if g["parent"] in out else cols[g["parent"]]
                should = ~np.isin(np.asarray(par, dtype=object), g["observed_if"])
                flip = rng.random(n) > P["agree"]
                skipped = np.where(flip, ~should & (rng.random(n) < 0.3), should)
                if "wave" in cols and "0" in vals:
                    w1 = np.asarray(cols["wave"]) == "1"
                    v = np.where(skipped & w1, "0", np.where(skipped, self.gv, v))
                    idx = np.where(skipped & w1, vals.index("0"), np.where(skipped, w, idx))
                else:
                    v = np.where(skipped, self.gv, v)
                    idx = np.where(skipped, w, idx)
            out[t], pcodes[t] = v, idx
        return out, pcodes

    def oracle(self, codes, cols, reps, rng):
        """MC estimate of P(y_t | GIVEN) per row: {item: [n, K]}."""
        items = self.schema["items"]
        n = len(next(iter(codes.values())))
        Kof = {t: len(items[t]["values"]) + (1 if items[t].get("gate") else 0)
               for t in self.par}
        acc = {t: np.zeros((n, Kof[t])) for t in self.par}
        tile_codes = {g: np.tile(c, reps) for g, c in codes.items()}
        tile_cols = {g: np.tile(c, reps) for g, c in cols.items()}
        _, pc = self.sample_predict(tile_codes, tile_cols, rng)
        for t in self.par:
            idx = pc[t].reshape(reps, n)
            for r in range(reps):
                acc[t][np.arange(n), idx[r]] += 1
        return {t: (acc[t] + 0.5 / Kof[t]) / (reps + 0.5) for t in acc}


def build(schema_path, scenario, out, seed, reps):
    config = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    schema = load_schema(schema_path, config)
    W = World(schema, seed, **SCENARIOS[scenario])
    n = schema["dataset"]["n_rows"]
    rng = np.random.default_rng([seed, 2])
    codes, cols = W.sample_given(n, rng)
    ans, _ = W.sample_predict(codes, cols, rng)
    frame = pd.DataFrame({k: (cols[k] if k in cols else ans[k]) for k in W.names})
    frame.insert(0, "respondent_id", ["R%06d" % i for i in range(1, n + 1)])
    frame = assign_roles(schema, frame.astype(object))
    os.makedirs(out, exist_ok=True)
    frame.to_parquet(os.path.join(out, "respondents.parquet"), index=False)
    # oracle for DEV rows (phase 1) and TEST rows (phase 2)
    s = schema["split"]
    sl = {"DEV": slice(s["n_train"], s["n_train"] + s["n_dev"]),
          "TEST": slice(s["n_train"] + s["n_dev"], n)}
    orng = np.random.default_rng([seed, 3])
    save = {}
    for role, sli in sl.items():
        c = {g: v[sli] for g, v in codes.items()}
        cl = {g: v[sli] for g, v in cols.items()}
        orc = W.oracle(c, cl, reps, orng)
        for t, P in orc.items():
            save[role + "::" + t] = P
    np.savez_compressed(os.path.join(out, "oracle.npz"), **save)
    print("wrote", out, frame.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--scenario", default="base", choices=sorted(SCENARIOS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args()
    build(a.schema, a.scenario, a.out, a.seed, a.reps)


if __name__ == "__main__":
    main()
