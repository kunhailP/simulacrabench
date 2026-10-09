"""Paired, repeated-split evaluation on any instrument directory (schema.json + respondents.parquet).

Each repeat hides a fresh random set of respondents (all PREDICT cells blanked),
shows the rest complete, calls predict() exactly as the grader does, and scores
with the official floor and skill. Per-cell log scores are saved so that two
submissions can be compared on identical cells (paired SE over respondents).

    .venv/bin/python tools/lab.py sub/v7 --data data/proxy/gss --reps 3
    .venv/bin/python tools/lab.py --compare sub/v7 sub/v8 --data data/proxy/gss
"""

import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "raw", "repo"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from make_sandbox import load_config, load_schema  # noqa: E402
from score import check, floored, score  # noqa: E402

CONFIG = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))


def load_sub(path):
    name = "sub_" + os.path.basename(os.path.normpath(path))
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, path)
    spec.loader.exec_module(mod)
    return mod


def split(data, rep, frac):
    schema = load_schema(os.path.join(data, "schema.json"), CONFIG)
    resp = pd.read_parquet(os.path.join(data, "respondents.parquet"))
    items = [k for k, r in schema["items"].items() if r["class"] in ("GIVEN", "PREDICT")]
    scored = [k for k in items if schema["items"][k]["class"] == "PREDICT"]
    rng = np.random.default_rng(1000 + rep)
    hid = rng.random(len(resp)) < frac
    vis = resp[~hid][["respondent_id"] + items].reset_index(drop=True)
    hidden = resp[hid][["respondent_id"] + items].reset_index(drop=True)
    blank = hidden.copy()
    blank[scored] = np.nan
    frame = pd.concat([vis, blank], ignore_index=True)
    off = len(vis)
    cells, truth = [], []
    H = hidden[scored].to_numpy(dtype=object)
    ids = hidden["respondent_id"].to_numpy(dtype=object)
    for r in range(len(hidden)):
        for j, t in enumerate(scored):
            cells.append((off + r, ids[r], t))
            truth.append(H[r, j])
    return schema, frame, cells, truth


def cell_logp(schema, vec, truth, cells):
    out = np.empty(len(cells))
    for i, (v, y, c) in enumerate(zip(vec, truth, cells)):
        opts = list(schema["items"][c[2]]["values"])
        if schema["items"][c[2]].get("gate"):
            opts.append(schema["gated_value"])
        out[i] = np.log(v[opts.index(y)])
    return out


def run(sub, data, reps, frac, tag):
    mod = load_sub(os.path.abspath(sub))
    outdir = os.path.join(ROOT, "results", "lab", os.path.basename(os.path.normpath(data)))
    os.makedirs(outdir, exist_ok=True)
    skills = []
    for rep in range(reps):
        schema, frame, cells, truth = split(data, rep, frac)
        t = time.time()
        vec = mod.predict(frame.copy(), json.loads(json.dumps(schema)))
        dt = time.time() - t
        check(schema, vec, cells)
        vec = floored(vec, schema, cells, CONFIG["scoring"]["floor"])
        r = score(schema, CONFIG, vec, truth, cells)
        lp = cell_logp(schema, vec, truth, cells)
        np.savez_compressed(os.path.join(outdir, f"{tag}_rep{rep}.npz"), logp=lp,
                            rid=np.array([c[1] for c in cells]),
                            item=np.array([c[2] for c in cells]),
                            U=r["uniform_reference"])
        skills.append(r["skill"])
        info = getattr(mod.predict, "last_info", None)
        print(f"{tag} rep{rep} skill {r['skill']:.4f} ({dt:.0f}s) "
              f"{json.dumps(info, default=str)[:300] if info else ''}", flush=True)
    print(f"{tag} MEAN {np.mean(skills):.4f}")


def compare(a, b, data):
    d = os.path.join(ROOT, "results", "lab", os.path.basename(os.path.normpath(data)))
    diffs, items = [], {}
    for rep in range(100):
        fa, fb = os.path.join(d, f"{a}_rep{rep}.npz"), os.path.join(d, f"{b}_rep{rep}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            break
        A, B = np.load(fa), np.load(fb)
        U = float(A["U"])
        df = pd.DataFrame({"rid": A["rid"], "item": A["item"], "d": (B["logp"] - A["logp"]) / U})
        per = df.groupby("rid")["d"].sum() / df.groupby("rid").size().mean()
        n = len(per)
        diffs.append((df["d"].mean(), per.std() / np.sqrt(n)))
        for it, v in df.groupby("item")["d"].sum().items():
            items[it] = items.get(it, 0.0) + v / len(df)
    m = np.mean([x[0] for x in diffs])
    se = np.sqrt(np.mean([x[1] ** 2 for x in diffs]) / len(diffs))
    print(f"{b} - {a}: skill {m:+.4f} (paired se {se:.4f}, {len(diffs)} reps)")
    top = sorted(items.items(), key=lambda kv: kv[1])
    print("  worst items:", [(k, round(v / len(diffs), 5)) for k, v in top[:6]])
    print("  best items: ", [(k, round(v / len(diffs), 5)) for k, v in top[-6:]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", nargs="?")
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.data)
    else:
        run(a.sub, a.data, a.reps, a.frac, a.tag or os.path.basename(os.path.normpath(a.sub)))
