# LLM welfare discretion: research hub

How do large language models decide welfare cases, and do social "deservingness" cues (job-search effort,
controllability of job loss) enter decisions where the law does not allow them? This repo is the working hub:
design notes, rules-as-code, item generators, runners, scorers, results, and the full decision log.

Target: ACL 2027 via the January 2027 ARR cycle, at EMNLP-main quality. Domain: U.S. SNAP (FY2026 federal rules,
P.L. 119-21 ABAWD work requirement), chosen because the law says which facts a decision may use.

## Status (2026-10-05): restarting the design
Design C (discretionary exemptions) was stopped before its primary test because the prompt left the decision
undefined (see docs/RESTART_PLAN.md). The next design is decided on paper before any GPU time.
**Start with docs/RESTART_PLAN.md.**

## What is established
| setting | finding | where |
| --- | --- | --- |
| Rule-computable eligibility, direct answer (4 open models) | income tests at chance; cue effects sit on near-chance verdicts | docs/results/design_B_rules_pilot.md |
| Same, thinking on (Qwen3-14B, 32B-AWQ) | income tests ~100% correct; deservingness cue effects 0 | docs/results/design_B_rules_pilot.md |
| Discretion, direct answer (4 models) | enumerated standards followed (14B/32B 100%); open standards flat in need; apparent cue effects are a yes-bias exposed by flipping the question (grant? / deny?) | archive/stage2_design_c/docs/design_C_discretion.md |
| Explicit need arithmetic (stage 1) | solved at 14B; 8B failures were format effects | archive/stage1_need_v1/ |

## Layout
    docs/       RESTART_PLAN, NORTH_STAR, decisions (log), design/, results/   (index: docs/README.md)
    src/rules/  SNAP rules-as-code (snap.py) + PolicyEngine-US cross-check
    src/gen/    item generator for the rule-computable set (make_rules_pilot.py)
    src/run/    first-token runner (log-probs, both answer orders), thinking runner (Qwen3 on/off, gpt-oss), model download
    src/eval/   scorers (cluster bootstrap over bases)
    tests/      rule tests (decision table + per-item gold)
    configs/    models.yaml (pinned revisions), rule_packet_fy2026.md, requirements.lock.txt
    data/       rules_pilot items; external/ third-party data (not redistributed)
    results/    rules (PolicyEngine cross-check), rules_pilot (direct), rules_think (thinking)
    lit/        literature notes, novelty checks, references.bib
    archive/    stage1_need_v1, stage2_design_c, old_queues (see archive/README.md)

## Environment (rebuild on a new pod)
    git clone https://github.com/kunhailP/llm-welfare-discretion.git && cd llm-welfare-discretion
    python3.12 -m venv /workspace/venv && /workspace/venv/bin/pip install -r configs/requirements.lock.txt
    #   (vllm 0.30.0, transformers 5.14.1; transformers >= 5.17 breaks Ministral)
    python3.12 -m venv /workspace/venv_pe && /workspace/venv_pe/bin/pip install -r configs/requirements_policyengine.lock.txt
    #   (only for the PolicyEngine-US gold cross-check, src/rules/crosscheck_pe.py)
    export HF_HOME=/workspace/hf && bash src/run/download_models.sh   # pinned revisions: configs/models.yaml
    unzip -o data/external/candidates/snap_qc_fy2024/qcfy2024_csv.zip -d data/external/candidates/snap_qc_fy2024/
    tar xzf archive/run_logs_2026-10-05.tar.gz                         # optional: restore logs/
    export WN_ROOT=<repo path>          # optional; defaults to the repo containing the script

Models so far: Qwen3-8B / 14B / 32B-AWQ, Ministral-3-8B, gpt-oss-20b. Open models only until the design is final.

## Reproduce the rule-computable reference (Design B)
    unzip -o data/external/candidates/snap_qc_fy2024/qcfy2024_csv.zip -d data/external/candidates/snap_qc_fy2024/
    python src/gen/make_rules_pilot.py --bases 40 --seed 11 --out data/rules_pilot/pilot.jsonl
    pytest tests                      # 9 rule tests
    python src/run/run_rules_pilot.py --model qwen3-14b --data data/rules_pilot/pilot.jsonl --out results/rules_pilot/qwen3-14b.jsonl
    python src/run/run_rules_think.py --model qwen3-14b --data data/rules_pilot/pilot.jsonl --out results/rules_think/qwen3-14b.jsonl
    python src/eval/score_rules_pilot.py --results results/rules_pilot/qwen3-14b.jsonl
    python src/eval/score_rules_think.py results/rules_think/qwen3-14b.jsonl

## Working rules
- Every design decision is logged in docs/decisions.md with its reason.
- Pre-register hypotheses and fallback rules before reading results; no task changes to chase an effect.
- Look at each result before running the next; read model reasoning before scoring it.
- Third-party data (SNAP QC public-use file, FairFund-Bench) is not redistributed; see
  archive/stage1_need_v1/docs/02_data_spec.md.
