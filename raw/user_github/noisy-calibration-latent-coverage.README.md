# Noisy calibration, latent coverage and the location of the mean

Code, results and manuscript for the paper *Noisy calibration, latent coverage and the location of the mean* by Kun Woo Park (in preparation for *Biometrika*). Every number in the paper and its Supplementary Material can be regenerated from this repository.

## The question

A prediction interval for a latent quantity, such as the true mean of an area that a survey did not sample, is often calibrated on noisy proxies. The latent residual `W` is observed only through `V = W + e`, with `e ~ N(0, D)`. The threshold `t` at which `|V|` has coverage `p` is observable. The question is which radius gives `W` coverage `q`, uniformly over a class of laws of `W`. Writing `x = D/t²`, the answer for log-concave laws is `t R_{p,q}(x)`, the sharp radius studied in the paper; `R_{p,q}(x) - 1` is the radius correction, a correction to a length, not to a coverage probability. It is sharp for this class and this single constraint; it is not claimed to be optimal among all procedures that use the whole calibration sample. For the larger class of bi-log-concave laws the upper bounds below (items 2, 3, 5, 6) hold and are sharp as `x -> 0`, but the exact radius at a fixed `x` and its computation (items 1, 7) use log-concavity; whether the two classes have the same sharp radius at every `x` is open.

**In one sentence:** the location of the latent mean governs the worst-case widening (order `D` away from the boundary, `c_q D^{1/2}` without restriction, `c_0.9 = 0.0191`); the widening is at most 1.16% of the radius at `q = 0.9` for every bi-log-concave law (numerically at most 0.22% for log-concave laws), and a calibration slack of `10^{-3}` removes it, so the noisy threshold at a suitable rank is valid even with unknown heterogeneous variances.

## Main results

1. **Reduction.** `R_{p,q}` is a supremum over point masses and log-affine laws on a segment, with closed-form Gaussian convolutions (Proposition 1).
2. **No restriction on the mean.** `R^B_{q,q}(x) <= 1 + c_q x^{1/2}` for every `x` and every bi-log-concave `W`, sharp as `x -> 0`. Ball arithmetic gives `c_0.9` in `[0.0190618, 0.0190619]` (Theorem 1, Lemma 1).
3. **Mean away from the boundary.** If `|E(W)| <= B < t`, then `Q_q(|W|) <= B + {(t - B)² + D}^{1/2} <= t + D/{2(t - B)}` for every `D` (Theorem 2). Asymmetric laws with mean zero attain the order `D` (Proposition 2).
4. **Transition.** A correction of order `D^{1/2}` requires the mean within `O(D^{1/2})` of the boundary. At distance `κ D^{1/2}` the limiting coefficient is the one-dimensional extremum `L_q(κ)`, which equals `c_q` up to `κ*`; `κ*` is about 20 at `q = 0.9` (Lemma 2, Theorem 3, Figure 1). These are worst-case statements: a particular law with its mean near the boundary need not require any correction.
5. **Slack.** A calibration slack of `10^{-3}` at `q = 0.9` removes the correction at every noise level (Corollary 1).
6. **Unknown heterogeneous variances.** With the order-statistic level of Proposition 3, the noisy threshold at the smallest rank with `p_k >= 0.901` and `k >= K p_k + 1` (`certified_rank`; rank 105 of 110) is itself valid for the latent target when the Gaussian noise variances are heterogeneous and unknown (Proposition 4). At `q = 0.9`, `delta = 0.05` such a rank exists if and only if `K >= 29`; for `K <= 3000` it is at most three ranks above the usual PAC rank (e.g. 46 instead of 45 at `K = 46`); the gap grows with `K` (9060 instead of 9050 at `K = 10000`). A lower bound `D_min` on the variances gives the analytic shrinkage `T - 0.114 D_min^{1/2}` (Corollary 2). Both rest only on ball-arithmetic constants.
7. **Computed radius.** With known variances and a log-concave latent law, the average-kernel radius `T R^mix` shortens the noisy threshold by about 10% in the simulations of Table 1. The procedure with the exact `R^mix` is valid; the implementation returns a numerically computed upper bound (see Precision).

`e33_edge.py` studies a regime that is not in the manuscript: near the feasibility edge `x_p`, `R_{p,q}(x) ~ M_q (x_p - x)^{1/2}`, with `M_q = sup Q_q(|W|)/E(W²)^{1/2}` over log-concave laws (`M_0.9 = 1.8532`, computed, not certified).

## Layout

```
paper/          main.tex and supplement.tex (Biometrika 2025 class), figures, references, PDFs
src/uai/        library: closed forms and constants (extremal), certificates (certify, interval),
                procedures, estimated-variance rules, closed-form latent laws
experiments/    one script per experiment, numbered as in the Supplementary Material
results/        outputs of the experiments (CSV/JSON) used in the paper
tests/          fast checks of constants, counterexamples and bounds
```

## Installation

Python 3.10 or later.

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,figures]"
make test
```

`python-flint` provides the Arb ball arithmetic. Scripts take the number of worker processes as an argument (`PROCS`, default 8). Set `OMP_NUM_THREADS=1` when running many workers.

## Reproducing the paper

| Item | Command | Output |
|---|---|---|
| Constants `c_q`, `C_{p,q}`, slack intervals (Lemma 1, Corollary 1) | `make constants` | `results/interval_constants.json` |
| Centring and transition values, Figure 1 | `make quick figures` | `results/centering.json`, `paper/fig_transition.pdf` |
| Edge constants `M_q` (not in the manuscript) | `make edge` | `results/edge.json` |
| Radius correction by location of the mean (Supplement S2) | `make quick figures` | `results/mean_location.csv`, `paper/fig_mean_location.pdf` |
| Certified `R_{p,q}`, boundary map (Supplement S2–S3) | `make certified` | `results/certified_R.csv`, `results/boundary_map.csv` |
| Table 1: computed radius, shape-free rule | `make simulations` | `results/hetldc_synth_summary.csv`, `results/shape_free_summary.csv` |
| Table 1: noisy threshold, Corollary 2; comparison with LatentCP and deconvolution | `make comparison` | `results/competitors_summary.csv` |
| Reliability of the analytic rules with 2000 data sets per law (Supplement S5) | `make comparison` | `results/analytic_reliability_summary.csv` |
| Table 1, last three laws only (truncated exponential, bimodal bi-log-concave, `t_3`) | `make table1-extra-laws` | merged into the files above |
| Estimated variances (Supplement S4) | `make estimated` | `results/estimated_scale_summary.csv`, `results/areawise_variance_summary.csv` |
| School-district application | `make apipop` | `results/hetldc_apipop_summary.csv` |
| Plug-in counterexamples, oracle-matched widths | `make supplement-extras` | `results/hetero_kernel.csv`, `results/conditional_synth_exact_*.csv` |
| Manuscript and supplement | `make paper` | `paper/main.pdf`, `paper/supplement.pdf` |

`make certified`, `make simulations` and `make table1-extra-laws` take hours on a few cores (about 2.5 minutes of one core per data set of Table 1 for the branch and bound of the computed radius; the estimated-variance rerun of Supplement S4 needs about 3.6 minutes per data set), `make constants` about 15 minutes on 4 cores and `make edge` more than 25 minutes; the other targets take minutes.

## Precision

Three levels of support are distinguished throughout.

| What | Status |
|---|---|
| Proposition 4 and Corollary 2 (`noisy_threshold_halfwidth`, `simple_shrink_halfwidth`, `certified_rank`) | Proofs plus constants `c_q`, `C_{p,q}` and slack intervals certified in Arb ball arithmetic with outward rounding (`uai.interval`, `make constants`) |
| The computed radius `T R^mix` with the exact `R^mix` | A valid procedure for log-concave latent laws (Proposition 1) |
| Values returned by `hetldc_certified` and `CertifiedShrinkTable` | Numerically computed upper bounds of `R^mix` and `R_{p,q}`: a branch and bound whose closed forms are evaluated in double precision, boxes cleared at a margin of `10^{-9}`; not interval arithmetic |

`hetldc_halfwidth` and `ShrinkTable` return grid values, which are lower bounds of the sharp radius and not valid radii. `results/r_table.json` (E04, differential evolution with quadrature) is a legacy table: at `x = 0.001` its values exceed the bound of Theorem 1 because the quadrature is inaccurate there, and the paper does not use it; likewise the column `R_DE` of `results/r_exact_p0.9_q0.9.csv` at `x = 0.001`. The transition coefficients, the centred-law coefficients and the edge constants `M_q` are computed in double precision and are not certified.

## Citation

```
@unpublished{park2026noisy,
  author = {Park, Kun Woo},
  title  = {Noisy calibration, latent coverage and the location of the mean},
  year   = {2026},
  note   = {Manuscript}
}
```
