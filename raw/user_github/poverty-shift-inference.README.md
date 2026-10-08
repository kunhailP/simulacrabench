# Honest Uncertainty for Transported Social Indicators under Population Shift

Valid simultaneous confidence bands for poverty headcount curves when transporting
an estimate to a population that was never surveyed.

> **Problem.** Machine-learning models predict household poverty accurately, but
> social measurement needs *valid uncertainty* for population-level quantities when
> generalising to an unseen population (a new survey, country, or year). We study the
> poverty headcount curve `θ_t(P) = P_P(Y < t)` — a non-smooth distributional
> functional — transported to a population for which no outcome survey is available.

## Summary of results

- **Standard within-population uncertainty collapses across populations.** On real
  household survey data, a standard 90% within-sample interval for the headcount
  covers the truth about **9%** of the time under real cross-survey shift; the
  between-population standard deviation is ≈ 6.8× the within-sample standard error.
- **Per-threshold conformal bands are dishonest about the *curve*.** A population-level
  conformal band that is valid at each threshold marginally covers the *entire* curve
  jointly only **37.5%** of the time (real pseudo-populations) and **54–60%** of the
  time across 123 real countries — far below the nominal 90%.
- **The Population Conformal Band (PCB) restores honest simultaneous coverage.** A
  finite-sample simultaneous band, with the *population* as the exchangeable unit and
  the CDF shape exploited, recovers **≈ 89%** joint coverage on both real household data
  and 123 countries, and approaches nominal coverage as the number of source
  populations grows.
- **Policy relevance.** When poverty is nowcast for countries without a recent survey,
  point-estimate targeting misallocates **12–18%** of the truly poorest countries even
  under a realistic nowcasting baseline; using the honest band to allocate audits
  reduces targeting regret.

![Marginal vs. simultaneous coverage of a transported poverty curve](figures/fig7_band_vs_marginal.png)

*The per-threshold marginal band (narrow, can fall below 0) fails to cover the whole
curve jointly; the PCB band is valid simultaneously and respects the CDF shape.*

## Contributions

1. A measurement of the **collapse of within-population uncertainty** across
   populations, with a decomposition (`Proposition 1`) into bias and between-population
   variance.
2. The **Population Conformal Band (PCB)**: a finite-sample-valid *simultaneous*
   confidence band for a transported non-smooth distributional functional, with the
   population as the exchangeable unit and a coverage-preserving isotonic (CDF-shape)
   tightening (`Proposition PCB-1`, `Lemma PCB-2`).
3. A **characterisation** of when Gaussian / random-effects variance inflation is valid
   for such functionals (it under-covers as the transport error becomes biased).
4. A **policy evaluation** of how honest uncertainty changes survey-less poverty
   targeting, on World Bank multi-country data.

The method machinery (functional conformal prediction, group/cluster conformal,
isotonic rearrangement) is adapted from existing tools; what is new is the target (a
transported distribution function), the population-level exchangeability, the
simultaneous-band construction with CDF tightening, and the empirical quantification.
See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Method

The interval procedures (all sharing one fixed base predictor, so coverage differences
are attributable to the interval method) are in [`src/inference/`](src/inference/):

| ID | Procedure | Module |
|---|---|---|
| M0 | Standard within-population bootstrap | `bootstrap_intervals.py` |
| M1 | Gaussian variance inflation (random-effects) | `variance_inflation.py` |
| M2 | Population split-conformal (per-threshold) | `population_conformal.py` |
| M3 | Shift-weighted conformal | `weighted_conformal.py` |
| **PCB** | **Population Conformal Band (simultaneous)** | `conformal_band.py` |
| M3′ | Localized / bias-corrected band | `localized_band.py` |

Formal statements and guarantees are in [`docs/METHOD_NOTES.md`](docs/METHOD_NOTES.md)
and [`docs/METHOD_PCB.md`](docs/METHOD_PCB.md); the full study protocol, hypotheses, and
experiment matrix are in [`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) and
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md). Per-experiment results are logged
in [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Repository layout

```
.
├── src/
│   ├── data/          # loaders: DrivenData household data (E1), World Bank PIP (E3)
│   ├── models/        # fixed base predictor (LightGBM) + out-of-sample protocols
│   ├── inference/     # interval procedures M0–M3, PCB, localized band
│   ├── simulation/    # synthetic population generator + grid driver (E2)
│   ├── evaluation/    # coverage, interval width, decision loss
│   ├── experiments/   # E1–E4 drivers and the two robustness tests
│   └── figures/       # figure scripts
├── scripts/pipeline/  # original numbered pipeline (preserved for provenance)
├── configs/           # declarative experiment configs
├── docs/              # design, method notes, pre-registration, experiment log, data
├── results/           # summary CSVs
├── figures/           # generated figures
└── tests/             # contract tests for the interval procedures
```

## Reproduce

```bash
pip install -r requirements.txt   # Python >= 3.10; CPU only, no GPU required

make test                         # interval-procedure contract tests (no data needed)
python -m src.simulation.run_sim_grid       # E2 simulation (no external data)
python -m src.data.fetch_pip                 # download World Bank PIP poverty curves
python -m src.experiments.e3_pip             # E3 multi-country validation
python -m src.experiments.e1_real_reanalysis # E1 (requires the DrivenData files in data/)
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the compute profile and
determinism notes.

## Data availability

- **E1 household data** (DrivenData competition): household features, ground-truth
  consumption, and per-threshold headcounts. **Not redistributed** under the competition
  licence; obtain it from the competition and place the files in `data/`.
- **E3 multi-country data** (World Bank Poverty and Inequality Platform): poverty curves
  for ~2,475 country-year surveys, fetched reproducibly with `src/data/fetch_pip.py`
  (World Bank, CC-BY 4.0). The processed curve table `data/external/pip_curves.csv` is
  included.

Details and candidate extension sources in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Citation

See [`CITATION.cff`](CITATION.cff). This is research code accompanying a working paper.

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). World Bank PIP data is
distributed under CC-BY 4.0; the DrivenData competition data is not included and is
subject to its own licence.
