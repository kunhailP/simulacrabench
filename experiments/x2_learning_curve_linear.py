# Additive logistic learning curve to 200k rows: floors at ~0.072 KL (approximation error).
# Run from the repo root: .venv/bin/python experiments/x2_learning_curve_linear.py <args>  (see docs/EXPERIMENTS.md)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "tools")
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
for n in (5195, 20000, 80000, 200000):
    rng = np.random.default_rng([7, 100 + n])
    c, cols = W.sample_given(n, rng); ans, _ = W.sample_predict(c, cols, rng)
    vis = pd.DataFrame({k: (cols[k] if k in cols else ans[k]) for k in W.names}).astype(object)
    vis.insert(0, "respondent_id", ["V%07d" % i for i in range(n)])
    frame = pd.concat([vis, hidden[vis.columns]], ignore_index=True)
    ins = v4.Instrument(frame, json.loads(json.dumps(schema)))
    orc = [O["DEV::" + t] for t in ins.targets]
    def kl(tabs):
        tot = 0.0
        for j, Pj in enumerate(tabs):
            p = floor + (1 - ins.K[j] * floor) * orc[j]
            q = floor + (1 - ins.K[j] * floor) * (Pj / Pj.sum(1, keepdims=True))
            tot += (p * (np.log(p) - np.log(q))).sum()
        return tot / (len(ins.hid) * len(ins.targets))
    res = {l2: kl(v4.m_linear(ins, ins.vis, ins.hid, l2)) for l2 in (1e-7, 1e-6, 1e-5)}
    print(n, {k: round(v, 4) for k, v in res.items()}, flush=True)
