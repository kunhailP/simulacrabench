# Latent coverage from noisy calibration (Statistica Sinica revision, work in progress)

Successor of `kunhailP/latent-coverage-noisy-thresholds` (tag `biometrika-submission`, commit
`3a9358d`). The certified modules `interval.py` and `noise_classes.py` are copied unchanged
from there (package `uai` renamed `latentcov`); `legacy/` keeps the certificate outputs they produced.

## Layout

```
src/latentcov/  interval, noise_classes (from the Biometrika code); levels, rank, gaussian (new);
                kv/ procedures that use known noise variances (competitors in R04, R05)
experiments/    r01-r06 and fetch_apipop.py (see below)
results/        outputs of the experiments
tests/          fast checks
notes/          roadmap and decisions, proofs (gaussian_exact.md, heterogeneous_latent.md)
legacy/         certificate files of the Biometrika version
data/           downloaded data, not redistributed (created by `make data`)
```

## Usage

```python
from latentcov import rank
r = rank(K=10_000, q=0.9, delta=0.05, noise='gaussian', latent='bi_log_concave')
r.k, r.k_necessary, r.level.support      # 9058, 9058, 'proved; certified in ball arithmetic'
```

`noise` is `gaussian`, `symmetric_unimodal` or `mean_zero_log_concave`; `latent` is
`bi_log_concave`, `symmetric_unimodal` (Anderson) or `none` (Proposition 2); `sided` is `two`
(interval `[-T, T]`) or `one` (`(-inf, T]`). `k` is the valid rank; the smallest valid rank lies
in `[k_necessary, k]`, and the two coincide for Gaussian and symmetric unimodal noise. `k` is
`None` when no rank is valid.

## Reproduce

Python 3.12; exact versions in `requirements-lock.txt`.

```
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt && .venv/bin/pip install -e .
make test
make all            # or one target at a time, as below
```

| target | script | what | time (32 cores) | output in `results/` |
|---|---|---|---|---|
| `reliability` | r01 | exact latent reliability at the Gaussian-extremal law, q = 0.8, 0.9, 0.95 | seconds | `reliability_*`, `fig_reliability_q.*` |
| `same-information` | r02 | rules with the same information; bimodal latent law, mixed noise | minutes | `same_information_v2*` |
| `mixed-noise` | r03 | exact stress test under mixed noise laws | minutes | `mixed_noise_stress*`, `fig_mixed_noise.*` |
| `competitors` | r04 | comparison with existing methods (LatentCP, Fay-Herriot, deconvolution, HetLDC) | hours | `competitors_v2_hetldc*` |
| `data` | fetch_apipop | pinned CRAN `survey` 4.5, checked by SHA-256 | seconds | `data/apipop.pkl` |
| `apipop` | r05 | school-district application | minutes | `apipop*` |
| `heterogeneous` | r06 | exact check of the heterogeneous-latent corollary | seconds | `heterogeneous_latent.csv` |

The Makefile sets one BLAS thread per process (`OMP_NUM_THREADS=1` etc.); set the same when
running a script directly, or the parallel experiments oversubscribe the cores. `PROCS` sets the
number of worker processes (default 32). R04 writes one line per finished job to
`results/competitors_v2_hetldc.ckpt.jsonl` and, rerun with the same arguments, skips the jobs
already there; delete that file to start afresh.
