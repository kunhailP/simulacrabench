# set-shift-probe (working title)

**Status: Block 1 confirmatory complete — see [`RESULTS-BLOCK1.md`](RESULTS-BLOCK1.md).
Blocks 2–4 (intervention power law, per-candidate audit, paper) in progress.**

A pre-registered measurement study of retrieval-set construction under
distribution shift, on 13 BEIR datasets × 4 retrieval systems.

**Questions.** When candidates are added to a retrieval set scored by set-F1
(exact-match / set metrics, as in RAG-style pipelines), (1) *when* does adding
help vs hurt — is there a decision boundary, and where do real deployments sit
on it? (2) Can label-free signals decide it? (3) How far does a tiny labeled
probe (k=10 queries) go?

**Block 1 answers** (confirmed on 4 datasets never inspected during design):

1. **Threshold law.** ΔF1(add) > 0 ⟺ p_add > tp₀/(n₀+n_G), an algebraic
   identity per query. Query-level sign agreement 100.0% (n=32,089), r=0.955;
   dataset-level Spearman 0.995.
2. **Probe-corrected admission works.** The label-free expected-F1 gate's
   R̂=Σp is structurally biased under shift; a 10-label probe correction
   ĉ = median(n_G/Σp) recovers oracle-level calibration on 13/13 datasets
   (|probe − oracle| ≤ 0.005) and is the strongest policy on the untouched
   datasets (0.367 vs 0.358 uncorrected gate, 0.277 learned truncation).
3. **Probe-based policy selection fails.** The same 10 labels cannot safely
   arbitrate between policy families: 3-way selection is significantly *worse*
   than the best label-free policy on untouched data (−0.024,
   CI [−0.037, −0.013]). Parameter correction needs far fewer labels than
   model selection — a registered, confirmed negative result.

## Evidence trail

This repo doubles as its own audit log. Read in this order:

| file | role |
|---|---|
| [`DESIGN.md`](DESIGN.md) | pre-registered design v1.0 → v1.2.1; every amendment is timestamped and states which datasets it was allowed to see |
| [`DEVIATIONS.md`](DEVIATIONS.md) | kill-check firings and hypothesis restatements, recorded when they happened |
| [`RESULTS-BLOCK1.md`](RESULTS-BLOCK1.md) | confirmatory verdicts against the registered criteria (two confirmed, one refuted — reported as registered) |
| `runs/` | every table regenerates from these persisted per-query artifacts |

Ground rules: headline numbers come only from `src/block1_confirm.py`
(DESIGN §11); the four confirmatory datasets (nq, hotpotqa, fever,
climate-fever) were never opened during exploration; criteria were never
edited after seeing results; git history is append-only.

## Layout

```
DESIGN.md  DEVIATIONS.md  RESULTS-BLOCK1.md   # the study
src/
  common.py                  # paths, dataset roster, metrics, run persistence
  download_data.py           # fetch the 13 BEIR datasets
  build_candidates.py        # candidate dumps: 4 systems × top-30, features + qrels
  build_climate_fever_delta.py  # disk-safe climate-fever build (see RESULTS-BLOCK1.md)
  h1_mechanism.py            # corner map (threshold-law test)
  block1_confirm.py          # THE confirmatory runner
  explore/                   # exploratory diagnostics — never headline evidence
runs/                        # persisted per-run config + per-query + aggregate
runs/pilot/                  # frozen artifacts from auditing the original pilot
```

Origin / motivating case study: https://github.com/kunhailP/legal-ir-pilot —
a legal-IR deployment sitting exactly in the predicted hurt corner, kept as a
curated frozen excerpt of the pilot working repo (the exploratory record).

## Reproduce

```
pip install -r requirements.txt   # pinned; needs a CUDA GPU for `candidates`
make data                         # ~130 GB BEIR raw data
make candidates                   # GPU: embeds 13 corpora, builds candidate dumps
make confirm                      # CPU: regenerates all Block 1 headline numbers
```

`make confirm` is deterministic (fixed seeds, LODO folds keyed by dataset-name
CRC): it reproduces `runs/block1_confirm/agg.json` exactly from the candidate
dumps.
