# When the Treatment Talks Back

Replication material for *When the Treatment Talks Back: Information Limits of
Observational Logs from Adaptive Generative Political Communication*.

The paper asks what conversation logs from an adaptive generative system can
identify about the system's turn-level behavior, and answers that competence
and informativeness are in tension: a policy that has learned which action is
better at a given history stops producing the contrast an analyst needs. The
manuscript is `paper/tex/main.tex`; the compiled PDF is `paper/tex/main.pdf`.

## Claims and where they are checked

| Claim | Statement | Verified by |
|---|---|---|
| Efficient influence function for the turn-level excursion estimand | Prop. 6, App. A | `theory/excursion_bound.md`; Monte Carlo in that file (25.98 vs 26.04) |
| Efficiency bound grows exponentially in policy decisiveness `D = -log min_a π(a\|h)` | Prop. 2, App. A | `theory/excursion_bound.md` |
| Minimax risk bounded away from zero when `M(π_n)/n → ∞` | Prop. 4, App. A | `theory/minimax_collapse.md`; `simulation/src/minimax_twopoint.py` |
| The "biased but stable" regime is two-point indistinguishability | §3 | `simulation/src/deceptive_regime.py` |
| Overlap mass localizes on the decision boundary (1-D) | Prop. 5, App. C | `theory/temperature_collapse.md`, `theory/proofs_w3_draft.md` |
| Localization on a decision manifold in R^d | Cor. 1, App. D | `theory/localization_manifold.md` |
| Fixed-τ counterexample: the estimand does not silently change | §4 | `simulation/src/continuous_h.py` (`overlap` estimator column) |
| Three regimes in `(n, τ)` | §4, Fig. 3 | `simulation/src/continuous_h.py` |
| Study 0-a, 0-b, 0-c | App. E | `analysis/src/` (see below) |

## Reproducing

```bash
pip install -r requirements.txt
pytest tests/                          # 7 numerical identities, ~1s
bash scripts/validate_repo.sh          # 6 steps incl. manuscript-vs-results check, ~30s
```

Simulations (`simulation/results/` is regenerated in place):

```bash
python simulation/src/simulation.py             # sign reversal, ~4s
python simulation/src/temperature_collapse.py   # two-state collapse, P1-P3, ~11s
python simulation/src/continuous_h.py --dgp uniform   # battery + phase diagram, ~11s
python simulation/src/continuous_h.py --dgp normal
python simulation/src/label_error.py            # label-noise sensitivity
python simulation/src/minimax_twopoint.py       # Le Cam construction checks
python simulation/src/minimax_crosscheck.py     # independent re-derivation (TV, Bernoulli, Bayes risk)
python simulation/src/deceptive_regime.py       # indistinguishability in the main DGP
```

Analysis. `01` and `07` need network and a GPU respectively; everything
downstream runs from committed data:

```bash
python analysis/src/02_reproduce.py --strict    # Study 0-a direction check
python analysis/src/13_inference.py             # Study 0-a with cluster-robust + ordinal
python analysis/src/08_policy_entropy.py --labels analysis/data/strategy_labels_judge32b.jsonl
python analysis/src/11_model_comparison.py
python analysis/src/12_effective_temperature.py
python analysis/src/09_fidelity_audit.py        # needs pyreadr
python analysis/src/14_human_agreement.py       # reports that coding is not yet done
```

## Layout

```
paper/          manuscript (LaTeX + compiled PDF) and bibliography
theory/         proofs and derivations; formal_framework.md is the source of truth
simulation/     DGPs, estimator battery, phase diagram; results/ is generated
analysis/       Study 0 pipeline, committed inputs and outputs
tests/          numerical identities used by the proofs
scripts/        validate_repo.sh
companion/      design spec and draft preregistration for the experiment,
                which has not been run
docs/           publication_map.md (scope split with the companion paper)
docs/internal/  dated historical record: audits, superseded drafts, design
                history, presentation materials, reading notes. Not part of
                the replication package and not to be cited. Every file
                carries a banner naming what superseded it, and
                docs/internal/README.md lists the claims in there that are
                known false.
```

## Data

`analysis/data/raw/debategpt.csv` is the public release accompanying Salvi et
al. From the Hackenburg et al. release we vendor only the three files the
pipeline reads, `data_and_analysis_code/study_{1,2,3}/output/data_prepared.rds`,
together with the upstream README for provenance. The rest of that release —
R analysis code, figures, fact-checker assessments, supplementary materials —
is not redistributed here; `analysis/data/fetch_upstream.sh` retrieves it.

Neither vendored file contains conversation text. The `.rds` files hold
derived variables and response identifiers only. The upstream authors state
that raw conversation logs contain personally identifiable information and
are not publicly available, and this repository does not contain them. The
56,831 figure in the paper counts released persuasion rows, not that study's
full sample.

Re-query outputs (`analysis/data/requery_*.jsonl`, 12,000 generations) and
judge labels are committed so that `08`, `11` and `12` reproduce without a
GPU. Their manifests record model, temperature and `k`, but not decoding
parameters beyond temperature or engine versions, so `07` itself is not
byte-reproducible.

## Companion paper

The G-MRT experiment is a separate paper; `docs/publication_map.md` records
which results belong where. No experimental data has been collected.

## License

Code (`simulation/`, `analysis/src/`, `tests/`, `scripts/`) is MIT. Text,
figures and derived results are CC BY 4.0, matching the license the article is
intended to appear under. Data redistributed from the Salvi et al. and
Hackenburg et al. public releases remains under those authors' terms and is
not covered by either; see `LICENSE` for the specifics and for what is
deliberately absent.
