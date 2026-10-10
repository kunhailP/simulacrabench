"""Fast screening: a candidate model vs the reference, raw hidden skill, many labs at once.

Smoke and screen tiers of the experiment ladder (docs/PLAN_v9.md):
  * one fit on all visible rows of lab.py's split (no K-fold cross-fitting, no
    blend, no temperature): the hidden rows are scored with the official floor
  * tables are cached per (lab, rep, model, item subset) under results/screen,
    so the reference is computed once and every new idea reuses it
  * labs run in parallel processes (--jobs); scores do not depend on sharing the
    GPU, only timings do, so never read timings off a parallel run
  * --items N keeps a seeded random N items (smoke tier)

    .venv/bin/python tools/screen.py --models tabicl1,sep2 --labs reach_irq --items 30   # smoke
    .venv/bin/python tools/screen.py --models tabicl1,sep2 --roles select --jobs 5       # screen
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

FLOOR = 1e-3


def cache_path(lab, rep, model, items):
    d = os.path.join(ROOT, "results", "screen", lab, f"rep{rep}" + (f"_i{items}" if items else ""))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, model + ".npz")


def run_one(lab, rep, models, items, force):
    """In-process: build the split once, fit each model once, cache hidden tables."""
    from oof import V7, build, registry
    schema, ins = build(os.path.join(ROOT, "data", "proxy", lab), rep)
    J = len(ins.targets)
    keep = np.arange(J)
    if items and items < J:
        keep = np.sort(np.random.default_rng(0).choice(J, items, replace=False))
        # restrict the instrument to the kept items (targets, codes, option counts)
        ins.targets = [ins.targets[j] for j in keep]
        ins.Y = ins.Y[:, keep]
        ins.K = ins.K[keep]
        ins.Kmax = int(ins.K.max())
    reg = registry()
    # truth for hidden rows, in Instrument coding
    tpath = cache_path(lab, rep, "_truth", items)
    if not os.path.exists(tpath):
        from lab import split
        _, _, cells, truth = split(os.path.join(ROOT, "data", "proxy", lab), rep, 0.15)
        full_targets = [k for k, r in schema["items"].items() if r["class"] == "PREDICT"]
        tl = np.array(truth, dtype=object).reshape(len(ins.hid), len(full_targets))
        T = np.full((len(ins.hid), len(ins.targets)), -1, np.int64)
        for c, t in enumerate(ins.targets):
            lut = {v: i for i, v in enumerate(ins.opts[t])}
            T[:, c] = [lut[v] for v in tl[:, full_targets.index(t)]]
        np.savez_compressed(tpath, Yh=T, K=ins.K, targets=np.array(ins.targets))
    for m in models:
        p = cache_path(lab, rep, m, items)
        if os.path.exists(p) and not force:
            continue
        t0 = time.time()
        tabs = reg[m](ins, ins.vis, ins.hid)
        cfg = next(iter(tabs))          # first config if a model returns several
        info = {}
        for mod in ("structev", "candidates"):
            for fn in ("m_sep2", "m_tabicl_num", "m_nsp"):
                f = getattr(sys.modules.get(mod), fn, None)
                if f is not None and hasattr(f, "last"):
                    info[fn] = str(f.last)[:200]
        np.savez_compressed(p, **{f"h{j}": tabs[cfg][j].astype(np.float32) for j in range(len(ins.targets))},
                            seconds=time.time() - t0, info=json.dumps(info))
        print(f"  {lab} {m}: {time.time() - t0:.0f}s", flush=True)


TS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def _tempered(P, T):
    Q = np.power(np.maximum(P, 1e-12), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def lab_skill_oracleT(lab, rep, model, items):
    """Skill after a per-item temperature chosen ON THE HIDDEN ROWS: optimistic,
    but applied alike to every model, it compares what is left once the
    overconfidence a temperature can fix is gone."""
    t = np.load(cache_path(lab, rep, "_truth", items))
    Yh, K = t["Yh"], t["K"]
    z = np.load(cache_path(lab, rep, model, items))
    U = np.mean(np.log(K))
    lp = []
    for j in range(len(K)):
        P = z[f"h{j}"].astype(np.float64)
        P = P / P.sum(1, keepdims=True)
        best = None
        for T in TS:
            Q = FLOOR + (1 - K[j] * FLOOR) * _tempered(P, T)
            l = np.log(Q[np.arange(len(Yh)), Yh[:, j]])
            if best is None or l.sum() > best.sum():
                best = l
        lp.append(best)
    return 1 + np.stack(lp, 1).mean() / U


def lab_skill(lab, rep, model, items):
    t = np.load(cache_path(lab, rep, "_truth", items))
    Yh, K = t["Yh"], t["K"]
    z = np.load(cache_path(lab, rep, model, items))
    U = np.mean(np.log(K))
    lp = []
    for j in range(len(K)):
        P = z[f"h{j}"].astype(np.float64)
        P = P / P.sum(1, keepdims=True)
        P = FLOOR + (1 - K[j] * FLOOR) * P
        lp.append(np.log(P[np.arange(len(Yh)), Yh[:, j]]))
    lp = np.stack(lp, 1)
    return 1 + lp.mean() / U, lp / U, float(z["seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="first one is the reference")
    ap.add_argument("--labs", default="")
    ap.add_argument("--roles", default="select")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--items", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--oracle-t", action="store_true", help="also report skill after a per-item oracle temperature")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    models = a.models.split(",")
    if a.labs:
        labs = a.labs.split(",")
    else:
        from suite import labs_for
        labs = labs_for(a.roles.split(","))
    if a.worker:
        run_one(labs[0], a.rep, models, a.items, a.force)
        return
    # one worker process per lab, at most --jobs at a time
    t0 = time.time()
    procs, queue = [], list(labs)
    while queue or procs:
        while queue and len(procs) < a.jobs:
            lab = queue.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--labs", lab,
                   "--models", a.models, "--rep", str(a.rep), "--items", str(a.items)]
            if a.force:
                cmd.append("--force")
            log = open(os.path.join(ROOT, "results", "screen", f"{lab}.log"), "w")
            procs.append((lab, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(2)
        for lab, p in list(procs):
            if p.poll() is not None:
                procs.remove((lab, p))
                if p.returncode:
                    print(f"{lab}: worker failed (exit {p.returncode}), see results/screen/{lab}.log")
    print(f"wall {time.time() - t0:.0f}s")
    ref = models[0]
    worst = None
    for lab in labs:
        try:
            s0, l0, t_0 = lab_skill(lab, a.rep, ref, a.items)
        except FileNotFoundError:
            continue
        parts = [f"{ref} {s0:.4f} ({t_0:.0f}s)"]
        for m in models[1:]:
            try:
                s, l, tm = lab_skill(lab, a.rep, m, a.items)
            except FileNotFoundError:
                parts.append(f"{m} missing")
                continue
            d = (l - l0).mean(1)                       # per respondent
            se = d.std() / np.sqrt(len(d))
            worst = d.mean() if worst is None else min(worst, d.mean())
            parts.append(f"{m} {s:.4f} ({s - s0:+.4f} ± {se:.4f}, {tm:.0f}s)")
        print(f"{lab:11s} " + " | ".join(parts))
        if a.oracle_t:
            print(f"{'':11s} oracle-T: " + " | ".join(
                f"{m} {lab_skill_oracleT(lab, a.rep, m, a.items):.4f}" for m in models
                if os.path.exists(cache_path(lab, a.rep, m, a.items))))
    if worst is not None:
        print(f"worst lab difference {worst:+.4f}")


if __name__ == "__main__":
    main()
