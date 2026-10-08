# Decision-Specific Audit Research Hub

상태: **Research Hub v0.2 — method freeze 전**  
기준일: 2026-08-20

## One-sentence paper

> 분포가 달라진 목표 환경에서, 같은 검증자료라도 보정과 정책선택에는 서로 다른 증거가 필요하므로, 의사결정별로 필요한 감사 예산을 계획하고 충분하지 않으면 결정을 유보하는 절차가 필요하다.

## What this project must contribute

이 프로젝트는 사후적으로 `k`별 성능곡선을 그리는 데서 끝나지 않는다. 최종 논문은 다음 인터페이스를 가진 **Decision-Specific Audit Planner**를 제안하고 검증해야 한다.

```text
Input:  decision, candidate policies, tolerance, risk level, pilot audit
Output: act now / collect more audits / abstain, with a decision certificate
```

핵심 과학적 비교는 다음 두 결정이다.

1. **Recalibration:** 고정된 retrieval rule의 target-domain 보정값을 정하고, 그 결정이 허용오차 안에서 안정적인지 판단한다.
2. **Policy selection:** 여러 policy 가운데 target-domain utility가 최선에 가까운 policy를 고르고, ε-regret 기준을 만족하는지 판단한다.

`abstain`은 별도의 거대한 연구주제가 아니라, 증거가 부족할 때 planner가 내놓아야 하는 정직한 출력이다.

## Ambition and venue logic

- **Scientific bar:** ICML/NeurIPS/ICLR main-track evaluation 또는 SIGIR full-paper 심사를 견디는 수준
- **Likely strong publication route:** 결과의 범용성과 방법적 깊이에 따라 상위 학회/저널 또는 TMLR 중 하나
- **Prohibited framing:** “상위 학회 게재 후 같은 결과를 TMLR 확장판으로 제출”

TMLR은 기존 archival paper와 겹치는 확장 투고를 받지 않는다. 따라서 이 프로젝트의 투고 경로는 순차 출판이 아니라, 결과를 본 뒤 하나를 선택하는 분기다. 자세한 기준은 `00_admin/SUBMISSION_STRATEGY.md`에 있다.

## Current truth

- 기존 `shift-study`의 공개 수치는 내부적으로 대체로 일관된다.
- 기존 `k=10`은 10개 개별 label이 아니라 10개의 audited/fully judged query다.
- `runs/block1_confirm/per_query.csv`는 이름과 달리 13개의 dataset-level 행이다.
- 새 분석에 필요한 `runs/candidates/`는 현재 공개·로컬 작업본에 없다.
- 기존 13개 BEIR dataset은 이미 노출되었으므로 prospective confirmation이 아니다.
- 따라서 현재 결과는 파일럿이며, candidate artifact 복구와 새로운 locked collection 없이는 flagship claim을 확증할 수 없다.

## Research hub map

```text
00_admin/            결정, 상태, 로드맵, 품질 게이트, 투고 전략
01_design/           RQ, estimand, method, 주장–증거 계약
02_literature/       최근접 문헌과 novelty 판정
03_data/             dataset provenance, split, license, judgment coverage
04_code/             실행 코드, 설정, 테스트, 재현 계약
05_results/          row-level 결과, 요약, 그림, 실패 기록
06_paper/            논문 개요와 원고
07_presentation/     DXW 발표와 연구 발표자료
08_review/           적대적 심사와 수정 이력
09_artifact/         외부 재현 가능한 공개 패키지
10_research_program/ 단일 논문을 넘어서는 연구 프로그램 연결
```

## Work gates

1. **G0 — Claim freeze:** top-level claim, estimands, decision losses, 금지 주장 고정
2. **G1 — Evidence recovery:** candidate artifact 복구 또는 재생성, provenance·checksum 기록
3. **G2 — Method validity:** planner 구현, simulation과 개발 데이터에서 coverage·stopping 검증
4. **G3 — Retrospective study:** 기존 13개 collection 재분석, 이질성·실패조건 확인
5. **G4 — Prospective validation:** 완전히 잠근 새 collection에서 단 한 번의 최종 평가
6. **G5 — Artifact audit:** 빈 환경에서 headline 결과와 그림 재생성
7. **G6 — Submission decision:** 결과에 따라 venue branch 선택

현재는 두 트랙이 병렬로 진행된다.

- **Local track:** method freeze, literature/novelty audit, test와 논문 구조
- **Pod track:** candidate 복구·재생성, 대규모 반복실험, artifact bundle 반환

Pod 실행계약은 `00_admin/POD_HANDOFF.md`가 기준이다. candidate-level evidence가 돌아오기 전에는 headline 실증결과를 갱신하지 않지만, 로컬의 설계·문헌·fixture 작업은 계속 진행한다.

## Non-negotiable rules

- query 수와 query–document judgment 수를 동일한 “label budget”으로 부르지 않는다.
- `ad_c`는 `full-target scalar reference`이며 oracle retrieval policy가 아니다.
- 개발에 노출된 collection을 confirmatory holdout이라고 부르지 않는다.
- 결과를 보고 tolerance, risk, metric, split을 바꾸면 `deviations.md`에 기록한다.
- 모든 표와 그림은 저장된 row-level artifact에서 재생성되어야 한다.
- 평균 성능만으로 성공을 선언하지 않고 coverage, regret, failure rate, worst-case collection을 보고한다.
- 정치학·로비·법학 스토리를 이 논문의 novelty로 사용하지 않는다.

## Start here

1. `00_admin/STATUS.md`
2. `01_design/METHOD_SPEC_v0.2.md`
3. `01_design/CLAIM_EVIDENCE_MATRIX.md`
4. `00_admin/QUALITY_GATES.md`
5. `04_code/CONTRACTS.md`
6. `00_admin/POD_HANDOFF.md`

원자료는 직접 수정하지 않는다.

- 기존 shift-study commit: `ef825e1d35d58d178c50054122ccc9e9ab54012c`
- GitHub: `https://github.com/kunhailP/shift-study`
- 현재 로컬 snapshot: `/Users/baggeon-u/Documents/Codex/2026-08-20/new-chat/work/shift-study`
