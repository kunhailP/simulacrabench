# unsampled-area-intervals

Research code and working notes on **prediction intervals for the realised mean of an area
that the survey did not sample**, when the only calibration data are noisy survey estimates
(with estimated design variances) for the areas that were sampled.

**Status: a draft short paper with proved results and a finite-sample procedure
([`paper/main.tex`](paper/main.tex)); not yet externally refereed.** The current state of every
claim — proved, certified, numerical only, already in the literature, or refuted — is in
[`docs/FINDINGS.md`](docs/FINDINGS.md). Read that first. The notes are in Korean.

Scope of the guarantees: known, fixed Gaussian noise variances independent of the latent
residual; calibration and new residuals i.i.d. from one log-concave law (bi-log-concave for the
small-noise and slack results); coverage of a new random area, conditional on the calibration
sample. Estimated variances are covered under a pooled variance-scale model (rule S, C50–C51);
area-wise variance estimates, dependence between residuals and variances, and design-based
coverage of a fixed finite population are outside the theory (experiments only).

## What is here

| Piece | Where | Status (see FINDINGS) |
|---|---|---|
| Exact two-stage SRSWOR observation kernel on small binary finite populations; rank of the observation operator; a two-world counterexample in which areas with <= 3 sampled PSUs cannot distinguish latent distributions that need very different interval widths | `src/uai/kernel.py`, `experiments/e01` | confirmed |
| Mixing two-stage designs across areas identifies more moments, but did not shorten the identified interval | `e02` | numerical, small model |
| Log-concave quantile/sd constants (truncated-Laplace extremal) and the worst-case coverage of a normal interval (87.75% at 90%) | `src/uai/logconcave.py`, `e03` | **known** for symmetric log-concave laws (Barthe–Koldobsky; He–Tkocz–Wyczesany 2023, Lemma 3) |
| Log-concave deconvolution shrink factor r_alpha(D/t^2): from a noisy conformal threshold to a latent-target threshold | `logconcave.py`, `e04`, `results/r_table.json` | numerical; novelty undetermined |
| Marginal latent-coverage bound of split conformal rules via the Beta law of the conformal order statistic (i.i.d. scores) | `e05`, `results/exact_coverage.csv` | Beta law exact; worst-case H numerical |
| Synthetic Monte Carlo; apipop finite-population comparisons with noisy conformal and Fay–Herriot (REML and parametric bootstrap) | `e06`–`e08` | FH met nominal coverage on apipop and was 12–17% narrower than the certified rule |
| Negative checks (multi-quantile constraints, depth allocation); ACS PUMS structure check | `e09`–`e11` | numerical |
| Conditional reliability Pr_D[coverage ≥ .90] without evaluation noise (synthetic) and on apipop; PAC versions of FH, CP and LDC; widths at equal reliability; grid cross-check of the shrink table | `e12`–`e15` | FH_PAC shortest but misses the target in some shapes; LDC_PAC always meets it, 11–20% wider |
| Exact 2-D reduction of the shrink function; small-noise law and boundary map; heterogeneous-noise kernel; HetLDC (synthetic, apipop) | `e16`, `e18`–`e21` | see `docs/THEORY_NOTE.md`, FINDINGS C29–C35 |
| Small-noise law R ≤ 1 + c_q√x for all x, sharp as x → 0; slack p > q removes widening; extension to bi-log-concave laws | `docs/THEORY_NOTE.md`, `paper/main.tex` | proved (C30, C38, C47) |
| Estimated variances plugged in; budget allocation | `e22`, `e23` | numerical, outside the theory |
| Certified upper bounds of R_{p,.9}, p = .9, .9036, .9068 (monotone branch and bound, double precision with margin); certified LDC and HetLDC | `src/uai/certify.py`, `e24`–`e26`, `e20` | certified up to floating point (C39–C41) |
| Ball-arithmetic enclosures of c_q and C_{p,q} (Arb, outward rounding) | `src/uai/interval.py`, `e27` | certified (C42, C46) |
| Level vs exact-kernel decomposition of the HetLDC gain | `e28` | at matched level the exact kernel is 0.3% shorter (C45) |
| Estimated variances: kernel-domination and scale lemmas; rule S (pooled variance scale) and area-wise envelope H | `src/uai/estimated.py`, `e29`, `e30` | S keeps 62–84% of the known-variance gain (baseline-dependent); certification adds 0.14% (C50–C54); this H construction needs ~50+ d.f. per area |
| Shape-free baseline (LatentCP-type Markov rule) | `procedures.shape_free_markov`, `e31` | the certified log-concave rule is 25–28% shorter than this baseline (C53) |

## Reproduce

Python >= 3.10.

```bash
pip install -e ".[test]"
make test          # fast checks of the core numbers
make data          # downloads apipop from the CRAN `survey` package into data/ (not redistributed)
make quick         # E01, E03, E05 (from the stored tables), E10: minutes
make tables PROCS=64   # E04, E05 compute, E09: global searches, long
make apipop PROCS=64   # E07, E08: 1,000 replications each
make theory PROCS=14   # E16-E28: exact reduction, certification, HetLDC; hours
```

Run scripts from the repository root. Stored outputs are in `results/`. Per-replication
pickles and downloaded data go to `data/` (git-ignored).

## Layout

```
src/uai/        kernel.py  logconcave.py  extremal.py  certify.py  interval.py  procedures.py
experiments/    e01 ... e28, fig_*.py, fetch_apipop.py, _common.py
paper/          main.tex (short paper draft), figures; `make paper` needs a TeX engine
results/        small JSON/CSV outputs of the experiments
docs/           FINDINGS.md (current state)  HISTORY.md  RESEARCH_DESIGN.md (dated log)  START_GATE_MEMO.md
tests/          test_core.py
```

## Data

- `apipop` (California Academic Performance Index population, via the R `survey` package), fetched by
  `experiments/fetch_apipop.py`.
- ACS 2024 1-year PUMS person file for one state, fetched by `e11`. Treated as a fixed record set; its
  means are not population truths for US residents.
