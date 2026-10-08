# Certifying retrieval-policy deployment with non-neutral AI judges — code and results

Anonymous research artifact accompanying a TMLR submission. Every number in the paper is regenerated from the row-level
result files under `05_results/` by the scripts under `04_code/`.

## Layout
- `04_code/lib/certificates.py` — split / PPI++ certificates (cross-fitted lambda, simultaneous corrections), recalibration bounds
- `04_code/lib/neutrality.py` — rho lower bounds (Fisher-z, percentile, BCa) and the judge-adoption score
- `04_code/6x_*.py` — candidate-pool builders (BEIR legacy/modern stacks, fully judged pools for TREC DL v1/v2, CAsT 2019, ANTIQUE), LLM judges
- `04_code/63_planner_v2.py` — sequential certificates (humans only / always-PPI / rule-based adoption) with per-method random streams
- `04_code/67,70,72,77` — PPI gain, neutrality diagnostic, self-preference test, mechanism map (F6)
- `04_code/80–83` — document-level auditing: decision-weighted sampling, judge control variate (coefficient 1 and pilot-lambda), sampling baselines, faithful active inference, menu-level allocation
- `04_code/84–85` — unified metric J50 and Figure F7; `86` — set-F1 linearised audit; `88` — pre-registered report
- `03_data/PROSPECTIVE_LOCK_v0.4/v0.5/v0.6` — pre-registration documents (committed before the target data were analysed)
- `06_paper/tmlr_submission` — LaTeX source and compiled PDF; `06_paper/THEORY_v0.1.md` — propositions and proofs
- `01_design/PAPER_2027_DESIGN_v0.1.md` — complete development log including negative results and retractions

## Reproduction
Pools: `python3 04_code/61_build_pool.py --beir <BEIR dir> --out <pools>` (BEIR), `64/75/87/89` for the fully judged pools.
Judges: `python3 04_code/66_llm_judge.py --cand <pool dir> --texts ... --queries ... --model Qwen/Qwen3-8B --tag llm <name>`.
Experiments: run the scripts with `--pools <pools> --stack <stack> --names <collection> --judge <llm|rr|mistral|inv> --train_dir <BEIR legacy pools>`.
Compile the paper: `cd 06_paper/tmlr_submission && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Requirements: Python 3.11, numpy, pandas, scipy, scikit-learn, torch, transformers>=4.51, sentence-transformers, bm25s; one 24 GB GPU for the judges.
