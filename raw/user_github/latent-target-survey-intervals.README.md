# What a survey prediction interval is sensitive to, and what that costs

Research repository for a manuscript prepared for the *Journal of Survey
Statistics and Methodology*.

A survey covering many regions produces one estimate for each. Those estimates
are then used to describe a place the survey did not cover. That means separating
real variation between places from sampling error — and the sampling error is
itself estimated, from designs supplying few degrees of freedom.

**The result this repository exists for is not about estimation accuracy.**
Competing constructions can be given exactly the same information about the
unknown scale and still report very different widths, because they differ in how
hard the reported width reacts to being wrong about it. That reaction is an
elasticity: `rho^2/2` for a band that deconvolves, exactly `1/2` for any band
proportional to the square root of a variance bound, and `0` for an uncorrected
anchor. Proposition S1 in `docs/THEORY_scale_sensitivity.md` is the exact
identity relating the first two.

Two targets are kept apart everywhere, in the code as well as the prose:

| target | object | reached by |
|---|---|---|
| **T1** | a further *survey estimate* | exchangeability of observed scores, finite sample |
| **T2** | its *latent population value* | a sampling-error bound, or a distributional shape assumption |

Nothing converts T1 coverage into T2 coverage without an explicit assumption.

## Layout

```
manuscript/  the current paper: blinded main.tex, separate titlepage.tex,
             figures.py, SUBMISSION.md
docs/        theory memos, claims ledger, protocols, data access, research log
experiments/ numbered, each with a protocol written before execution
results/     generated outputs; results/archived/ holds inherited tables
src/dac/     bands, generators, diagnostics
tests/       contract tests binding each claim to executable behaviour
paper/       the earlier full-CDF manuscript, kept as written and not superseded
```

`manuscript/` is scoped to a **scalar** population summary. It is a different
paper from `paper/`, and the two are deliberately not placed under one guarantee.

## Reproduce

```
pip install numpy scipy pandas matplotlib pytest
make                 # simulation experiments and the contract tests
make survey          # the experiments needing ESS rounds 9-11
cd manuscript && python3 figures.py && latexmk -pdf main.tex
```

Experiments needing no microdata run from this repository alone. The rest need
European Social Survey rounds 9–11; `experiments/fetch_ess.py` retrieves them
through the portal API for any End User Licence holder. **No user identifier is
stored here and the cache is written outside the working tree.** See
`docs/DATA.md`.

## The main results, and where each is settled

| result | where |
|---|---|
| The band depends on the two scales only through their ratio; the population requirement is `rho^-4` smaller than a variance-component criterion implies | `exp10`, `exp13` |
| That boundary is a property of the problem, not of our estimator — a matching necessary condition, and the reciprocal duality making the two axes harmonic | `exp30`, `docs/THEORY_optimality.md` |
| Scale elasticity, the exchange identity, and three consequences | `exp29`, `docs/THEORY_scale_sensitivity.md` |
| An exact `F` pivot for the correction ratio, a closed-form structure confidence set, and a *certified* bound making the supremum a computation | `exp20`–`exp23` |
| Against a comparator sharing target, information and budget: 13.9–22.2% narrower | `exp26` |
| Where it stops: containment fails at 4–8 design degrees of freedom | `exp24`, `exp25` |
| ESS rounds 9–11, 45 configurations: narrower in 44 of 45, median 0.821 | `exp27`, `exp28` |

## What is not claimed

The guarantee is proved in a restricted normal model with a known centre and
independent populations. On ESS the centre is estimated, the scores carry a
measured dependence of about `-1/(n_r-1)`, two of three pre-specified items are
badly non-normal, and no `chi^2` law for a complex-design variance estimator is
established. **The ESS analysis is an application under stated approximations and
the guarantee is not attached to it.** The necessary condition of `exp30` bounds
the *certification* problem, not interval width, and claims no optimality; in the
equal-variance submodel it is the classical intraclass-correlation design problem
and is cited as such.

## Working rules

1. **A protocol before an experiment.** Written, dated and committed before
   execution, recording what a disappointing outcome would look like.
2. **A claim needs a row.** Every number intended for the manuscript appears in
   `docs/CLAIMS.md` with the artefact that produces it.
3. **Targets are declared, never inferred.**
4. **Retractions are recorded next to what replaced them.** `docs/CLAIMS.md` and
   `docs/RESEARCH_LOG.md` keep the claims that were withdrawn — the endpoint
   claim for the supremum, refuted by counterexample; the one-axis reduction; a
   linearised two-point separation that inflated a bound; and a reading of prior
   work that a full-text check did not support.

Cell seeds derive from cell labels by SHA-256, so any cell reproduces on its own
and no cell can be selected after its outcome is known.

## Provenance

Archived tables under `results/archived/` come from
`github.com/kunhailP/design-aware-conformal` at `ea592e9`. They were produced on
licensed microdata and have **not** been re-executed here; `docs/CLAIMS.md` marks
them `inherited`.
