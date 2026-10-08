# Per-component timing of a submission on UNHCR: .venv/bin/python experiments/x9_profile.py sub/v7/main.py
# Run from the repo root (see docs/EXPERIMENTS.md, v7).
import sys, json, time, importlib.util, collections, numpy as np, torch
sys.path.insert(0, "tools"); sys.path.insert(0, "raw/repo")
from make_sandbox import load_config, load_schema
from score import load_frames, sample_rows
spec = importlib.util.spec_from_file_location("vX", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
T = collections.defaultdict(float)
def wrap(name):
    f = getattr(m, name)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.time(); r = f(*a, **k); torch.cuda.synchronize(); T[name] += time.time() - t; return r
    setattr(m, name, g)
for n in ("m_marginal", "m_eb", "m_linear", "m_mlp", "m_sint", "_sint_design", "gate_factor", "_eg", "_lfdr"):
    wrap(n)
cfg = load_config("raw/repo/config.yml"); schema = load_schema("raw/repo/data/unhcr.json", cfg)
masked, cells, _ = sample_rows(schema, load_frames("data/worlds/base/unhcr", schema), 1)
t = time.time(); m.predict(masked.copy(), json.loads(json.dumps(schema))); tot = time.time() - t
print("total %.1fs" % tot)
for k, v in sorted(T.items(), key=lambda x: -x[1]): print("  %-14s %.1fs" % (k, v))
ins = m.Instrument(masked.reset_index(drop=True), json.loads(json.dumps(schema)))
X, blk, nmain = m._sint_design(ins)
print("sint features", X.shape, "blocks", blk.max() + 1, "main blocks", nmain, "J", len(ins.targets), "Kmax", ins.Kmax, "W params %.1fM" % (X.shape[1] * len(ins.targets) * ins.Kmax / 1e6))
