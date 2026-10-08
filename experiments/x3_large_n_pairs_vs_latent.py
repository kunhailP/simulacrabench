# At n=80k: pairwise logistic 0.034 vs MLP/latent 0.060 -> the gap is pairwise GIVEN interactions.
# Run from the repo root: .venv/bin/python experiments/x3_large_n_pairs_vs_latent.py <args>  (see docs/EXPERIMENTS.md)
import sys, json, itertools, numpy as np, pandas as pd, torch
sys.path.insert(0, "tools")
import proto_latent as PL
from proto_latent import v4
from world import World, SCENARIOS
from make_sandbox import load_config, load_schema
from score import load_frames, sample_rows
cfg = load_config("raw/repo/config.yml"); floor = cfg["scoring"]["floor"]
schema = load_schema("raw/repo/data/world_bank.json", cfg)
d = "data/worlds/base/world_bank"
masked, cells, _ = sample_rows(schema, load_frames(d, schema), 1)
targets = [k for k, r in schema["items"].items() if r["class"] == "PREDICT"]
hidden = masked[masked[targets].isna().any(axis=1)].reset_index(drop=True)
O = np.load(d + "/oracle.npz")
W = World(schema, 7, **SCENARIOS["base"])
n = int(sys.argv[1])
rng = np.random.default_rng([7, 100 + n])
c, cols = W.sample_given(n, rng); ans, _ = W.sample_predict(c, cols, rng)
vis = pd.DataFrame({k: (cols[k] if k in cols else ans[k]) for k in W.names}).astype(object)
vis.insert(0, "respondent_id", ["V%07d" % i for i in range(n)])
ins = v4.Instrument(pd.concat([vis, hidden[vis.columns]], ignore_index=True), json.loads(json.dumps(schema)))
orc = [O["DEV::" + t] for t in ins.targets]
def kl(tabs):
    tot = 0.0
    for j, Pj in enumerate(tabs):
        p = floor + (1 - ins.K[j] * floor) * orc[j]
        q = floor + (1 - ins.K[j] * floor) * (Pj / Pj.sum(1, keepdims=True))
        tot += (p * (np.log(p) - np.log(q))).sum()
    return tot / (len(ins.hid) * len(ins.targets))
X1 = ins.X.copy(); cl = []
for a, b in itertools.combinations(ins.given, 2):
    k = ins.gcodes[a] * ins.gcard[b] + ins.gcodes[b]
    oh = np.eye(ins.gcard[a] * ins.gcard[b], dtype=np.float32)[k]
    cl.append(oh[:, oh[ins.vis].sum(0) >= 5])
ins.X = np.concatenate([X1] + cl, 1)
print(n, "pairs", {l2: round(kl(v4.m_linear(ins, ins.vis, ins.hid, l2)), 4) for l2 in (1e-7, 1e-6)}, flush=True)
ins.X = X1
for D in (0, 2):
    ck = PL.fit_predict(ins, ins.vis, ins.hid, D, 30)
    print(n, "latent D=%d" % D, {e: round(kl(t), 4) for e, t in ck.items()}, flush=True)
