"""Score a submission in-process (fast) on sandbox/sim2 data, all instruments.

    .venv/bin/python tools/evaluate.py sub/v1 --data sim2 --phase 1
Prints exact (un-noised) skill per instrument, the mean, and seconds.
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
from make_sandbox import load_config, load_schema  # noqa: E402
from score import check, floored, load_frames, sample_rows, score  # noqa: E402


def load_sub(path):
    spec = importlib.util.spec_from_file_location("sub_main", os.path.join(path, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, path)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--data", default="data/sim2")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--instruments", default="unicef,world_bank,unhcr")
    a = ap.parse_args()
    config = load_config(os.path.join(ROOT, "raw", "repo", "config.yml"))
    t_all = time.time()
    mod = load_sub(os.path.abspath(a.submission))
    res = {}
    for inst in a.instruments.split(","):
        schema = load_schema(os.path.join(ROOT, "raw", "repo", "data", inst + ".json"), config)
        resp = load_frames(os.path.join(ROOT, a.data, inst), schema)
        masked, cells, truth = sample_rows(schema, resp, a.phase)
        t = time.time()
        vec = mod.predict(masked.copy(), json.loads(json.dumps(schema)))
        dt = time.time() - t
        check(schema, vec, cells)
        vec = floored(vec, schema, cells, config["scoring"]["floor"])
        r = score(schema, config, vec, truth, cells)
        res[inst] = r["skill"]
        info = getattr(mod.predict, "last_info", None)
        print("%-11s skill %.4f  (se %.4f)  %.1fs  %s" % (
            inst, r["skill"], r["std_error"] / r["uniform_reference"], dt,
            json.dumps(info, default=str) if info else ""), flush=True)
    print("MEAN %.4f   total %.1fs" % (np.mean(list(res.values())), time.time() - t_all))


if __name__ == "__main__":
    main()
