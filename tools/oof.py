"""Cache cross-fitted base-model tables per (dataset, split repeat, model).

The split is tools/lab.py's (same seed, same hidden rows); the K folds over the
visible rows are v7's. For each model the cache holds OOF tables for visible
rows and the fold-bagged tables for hidden rows, plus the truth needed to score
the hidden rows. Blending experiments then run in seconds (tools/blend.py).

    .venv/bin/python tools/oof.py --data data/proxy/gss --rep 0 --models eb,lin,mlp,lgbm
"""

import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

from lab import split  # noqa: E402


def v7():
    spec = importlib.util.spec_from_file_location("v7main", os.path.join(ROOT, "sub", "v7", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V7 = v7()
import candidates as C  # noqa: E402

N_FOLDS = V7.N_FOLDS


def registry():
    r = {
        "marg": lambda ins, tr, rows: {"": V7.m_marginal(ins, tr, rows)},
        "eb": lambda ins, tr, rows: {"": V7.m_eb(ins, tr, rows)},
        "lin": lambda ins, tr, rows: {str(l2): V7.m_linear(ins, tr, rows, l2) for l2 in V7.L2_GRID},
        "mlp": lambda ins, tr, rows: {str(e): t for e, t in V7.m_mlp(ins, tr, rows, V7.MLP_EPOCHS).items()},
        "sint": lambda ins, tr, rows: {"_".join(map(str, c)): t for c, t in V7.m_sint(ins, tr, rows).items()},
    }
    r.update(C.REGISTRY)
    return r


def cache_dir(data, rep):
    d = os.path.join(ROOT, "results", "oof", os.path.basename(os.path.normpath(data)), f"rep{rep}")
    os.makedirs(d, exist_ok=True)
    return d


def build(data, rep, frac=0.15):
    schema, frame, cells, truth = split(data, rep, frac)
    frame = frame.reset_index(drop=True)
    ins = V7.Instrument(frame, schema)
    ins.schema_items = schema["items"]
    return schema, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--models", required=True)
    ap.add_argument("--frac", type=float, default=0.15)
    a = ap.parse_args()
    schema, frame, cells, truth = split(a.data, a.rep, a.frac)
    frame = frame.reset_index(drop=True)
    ins = V7.Instrument(frame, schema)
    ins.schema_items = schema["items"]
    d = cache_dir(a.data, a.rep)
    # truth for hidden rows in Instrument coding
    J = len(ins.targets)
    T = np.full((len(ins.hid), J), -1, np.int64)
    tl = np.array(truth, dtype=object).reshape(len(ins.hid), J)
    for j, t in enumerate(ins.targets):
        lut = {v: i for i, v in enumerate(ins.opts[t])}
        T[:, j] = [lut[v] for v in tl[:, j]]
    np.savez_compressed(os.path.join(d, "_truth.npz"), Yh=T, Yv=ins.Y[ins.vis], K=ins.K,
                        targets=np.array(ins.targets))
    rng = np.random.default_rng(V7.SEED)
    perm = rng.permutation(len(ins.vis))
    folds = [perm[i::N_FOLDS] for i in range(N_FOLDS)]
    reg = registry()
    for name in a.models.split(","):
        t0 = time.time()
        oof, bag = {}, {}
        for part in folds:
            tr = ins.vis[np.setdiff1d(np.arange(len(ins.vis)), part)]
            rows = np.concatenate([ins.vis[part], ins.hid])
            for cfg, tabs in reg[name](ins, tr, rows).items():
                if cfg not in oof:
                    oof[cfg] = [np.zeros((len(ins.vis), k)) for k in ins.K]
                    bag[cfg] = [np.zeros((len(ins.hid), k)) for k in ins.K]
                for j in range(J):
                    oof[cfg][j][part] = tabs[j][:len(part)]
                    bag[cfg][j] += tabs[j][len(part):] / N_FOLDS
        for cfg in oof:
            key = name + (":" + cfg if cfg else "")
            np.savez_compressed(os.path.join(d, key + ".npz"),
                                **{f"o{j}": oof[cfg][j].astype(np.float32) for j in range(J)},
                                **{f"h{j}": bag[cfg][j].astype(np.float32) for j in range(J)})
        print(f"{name}: {len(oof)} configs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
