"""Where does a submission lose to the oracle? Decompose the excess loss.

On a world built by tools/world.py the oracle P*(y | GIVEN) is known, so a
method's expected excess log loss per cell is KL(P* || Q), with no answer
noise. This splits that excess over the parts of the problem:

  item        which questions carry the gap
  gate        no gate / gate on a GIVEN parent / gate on a PREDICT parent
  cellsize    how many visible rows share the row's full GIVEN pattern
  sharpness   oracle entropy (near-deterministic cells vs diffuse ones)

    .venv/bin/python tools/headroom.py sub/v2 --data data/worlds/base --instruments world_bank
Predictions are cached next to the data (pred_<sub>.npz) so a re-split is free.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluate import load_sub  # noqa: E402
from make_sandbox import load_config, load_schema  # noqa: E402
from score import floored, load_frames, sample_rows  # noqa: E402
from world import split_options  # noqa: E402


def table(df, key, total):
    g = df.groupby(key, observed=True)["kl"].agg(["count", "mean", "sum"])
    g["share"] = g["sum"] / total
    return g.sort_values("sum", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--data", required=True)
    ap.add_argument("--instruments", default="unicef,world_bank")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    config = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    floor = config["scoring"]["floor"]
    tag = os.path.basename(os.path.normpath(a.submission))
    for inst in a.instruments.split(","):
        schema = load_schema(os.path.join(ROOT, "raw", "repo", "data", inst + ".json"), config)
        d = os.path.join(ROOT, a.data, inst)
        resp = load_frames(d, schema)
        masked, cells, _ = sample_rows(schema, resp, 1)
        cache = os.path.join(d, "pred_%s.npz" % tag)
        if os.path.exists(cache):
            Z = np.load(cache, allow_pickle=True)
            vec = list(Z["vec"])
        else:
            mod = load_sub(os.path.abspath(a.submission))
            vec = mod.predict(masked.copy(), json.loads(json.dumps(schema)))
            np.savez_compressed(cache, vec=np.array(vec, dtype=object))
        vec = floored(vec, schema, cells, floor)
        O = np.load(os.path.join(d, "oracle.npz"))
        items = schema["items"]
        targets = [k for k, r in items.items() if r["class"] == "PREDICT"]
        given = [k for k, r in items.items() if r["class"] == "GIVEN"]
        nh = len(vec) // len(targets)
        orc = floored([O["DEV::" + t][r] for r in range(nh) for t in targets], schema, cells, floor)

        # GIVEN pattern sizes among visible rows
        vis = masked[~masked[targets].isna().any(axis=1)]
        hid = masked[masked[targets].isna().any(axis=1)].reset_index(drop=True)
        key = lambda f: f[given].astype(str).agg("|".join, axis=1)  # noqa: E731
        cnt = key(vis).value_counts()
        size = key(hid).map(cnt).fillna(0).to_numpy()

        rows = []
        for i, (p, q) in enumerate(zip(orc, vec)):
            r, t = divmod(i, len(targets))
            t = targets[t]
            kl = float((p * (np.log(p) - np.log(q))).sum())
            ent = float(-(p * np.log(p)).sum() / np.log(len(p)))
            g = items[t].get("gate")
            gk = "none" if not g else ("given-parent" if items[g["parent"]]["class"] == "GIVEN"
                                       else "predict-parent")
            _, nr, ordered = split_options(items[t]["values"])
            kind = ("ordinal" if ordered else "nominal") + ("+nonresp" if nr else "")
            rows.append((t, gk, kind, size[r], ent, kl))
        df = pd.DataFrame(rows, columns=["item", "gate", "kind", "n", "ent", "kl"])
        df["cellsize"] = pd.cut(df["n"], [-1, 0, 2, 5, 10, 20, 50, 1e9],
                                labels=["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"])
        df["sharpness"] = pd.cut(df["ent"], [-1, 0.1, 0.3, 0.6, 2],
                                 labels=["<.1", ".1-.3", ".3-.6", ">.6"])
        total = df["kl"].sum()
        pd.set_option("display.width", 160)
        print("=== %s  %s  excess %.4f nats/cell over %d cells" % (
            inst, tag, total / len(df), len(df)))
        for k in ("kind", "gate", "cellsize", "sharpness"):
            print(table(df, k, total).round(4).to_string(), "\n")
        print(table(df, "item", total).head(a.top).round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
