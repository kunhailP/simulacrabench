# Main-effects vs all-pairs logistic against the oracle (n=5195): uniform L2 cannot use pairs.
# Run from the repo root: .venv/bin/python experiments/x1_structure_main_vs_pairs.py <args>  (see docs/EXPERIMENTS.md)
import sys, os, json, itertools, time, numpy as np
sys.path.insert(0, "tools")
import proto_latent as P
from proto_latent import v4
from make_sandbox import load_config, load_schema
from score import load_frames, sample_rows
inst, sc = sys.argv[1], sys.argv[2]
cfg = load_config("raw/repo/config.yml"); floor = cfg["scoring"]["floor"]
schema = load_schema("raw/repo/data/%s.json" % inst, cfg)
d = "data/worlds/%s/%s" % (sc, inst)
masked, cells, _ = sample_rows(schema, load_frames(d, schema), 1)
ins = v4.Instrument(masked.reset_index(drop=True), json.loads(json.dumps(schema)))
O = np.load(d + "/oracle.npz"); orc = [O["DEV::" + t] for t in ins.targets]
def kl(tabs):
    tot = 0.0
    for j, Pj in enumerate(tabs):
        p = floor + (1 - ins.K[j] * floor) * orc[j]
        q = floor + (1 - ins.K[j] * floor) * (Pj / Pj.sum(1, keepdims=True))
        tot += (p * (np.log(p) - np.log(q))).sum()
    return tot / (len(ins.hid) * len(ins.targets))
X1 = ins.X.copy()
cols = []
for a, b in itertools.combinations(ins.given, 2):
    k = ins.gcodes[a] * ins.gcard[b] + ins.gcodes[b]
    oh = np.eye(ins.gcard[a] * ins.gcard[b], dtype=np.float32)[k]
    cols.append(oh[:, oh[ins.vis].sum(0) >= 5])
X2 = np.concatenate([X1] + cols, 1)
print("features main %d, +pairs %d" % (X1.shape[1], X2.shape[1]))
for name, X, grid in [("main", X1, (1e-5, 3e-5, 1e-4, 3e-4)), ("pairs", X2, (3e-5, 1e-4, 3e-4, 1e-3, 3e-3))]:
    ins.X = X
    for l2 in grid:
        t = time.time()
        print("%-6s l2=%g  KL %.4f  %.0fs" % (name, l2, kl(v4.m_linear(ins, ins.vis, ins.hid, l2)), time.time() - t), flush=True)
