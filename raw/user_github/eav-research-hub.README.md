# EAV Research Hub — LLM으로 잰 정당들을 비교해도 되는가

> **고정 질문: *Can We Compare Parties Measured by Language Models?***
> 비교정치의 핵심 결론(급진우파 전염, 이슈 소유, 동서 차이, 양극화)은 **집단 간 비교**다. LLM이 정당가족·언어마다 다르게 틀리면 그 결론은 측정 도구의 산물일 수 있다. 이 저장소는 이 질문에 **예 / 아니오 / 이런 조건에서만**으로 확실한 답을 내기 위한 사전등록 연구다. **질문은 결과에 따라 바꾸지 않고, 방법을 보강한다.**
> 목표: AJPS (결과에 따라 PA·PSRM). 단독 연구.

## 증거 원장 (2026-09-14)

| # | 주장 | 상태 | 근거 | 기존 연구와의 차이 |
|---|---|---|---|---|
| E1 | 소형 open-weight 모델(7–8B)은 정당마다 다르게 틀려, UK 보수 − 노동 사회 영역 비교를 인간 코더 범위 밖으로 왜곡한다 (Mistral −6.97pp, Phi −5.55pp) | **확립** (모델 계열 2개). 단 이 모델들의 전문가 다수결 일치율은 30–58% | `prereg-v3-uk` · [docs/15](docs/15_uk_results_interpretation.md) | 같은 문장의 인간 다평정자 코딩을 기준으로 한 집단 조건부 동등성 검정, 인간 코더 간 범위를 기준선으로 사용 |
| E2 | 위 모델들 사이에서는 정확도가 높을수록 왜곡이 작다 (Spearman −0.95, 12셀) | **탐색** | [docs/15](docs/15_uk_results_interpretation.md) | — |
| E3 | Benoit et al. (2026) 프런티어 앙상블 점수와 전문가 조사는 사전 지정 11개 비교에서 결론이 갈리지 않는다. 사회 동서는 일치, 환경 생태 − 사민 +0.48 SD는 경계선 | **확립** (그들에게 우호적). 정당가족 대비 검정력 0.15라 대부분 판정 불가 | `prereg-benoit-e` · [docs/17](docs/17_benoit_e_results_interpretation.md) | 그들의 합산 검증(상관·분포)을 처음으로 집단 조건부로 검정. 전문가는 정당을, LLM은 텍스트를 재므로 "오류"가 아니라 "도구 간 비교 결론 불일치"로 판정 |
| E4 | 집단당 300문장 인간 감사가 관행적 정확도 기준보다 위험한 비교를 더 잘 가려내거나 더 나은 모델을 고른다는 증거는 없다 (왜곡 순위는 UK에서 잘 맞힘: 0.88 vs 1 − 정확도 0.15) | **확증에서 지지 안 됨** (PImPo 12셀: F1·F2·F3·F4 모두, 판단 보류 87%; UK 탐색 48셀도 같음). "못한다"가 입증된 것은 아니다 | `prereg-audit-f` · `prereg-audit-f2` · [docs/18](docs/18_prereg_audit_forecast_draft.md) · [docs/20](docs/20_prereg_audit_screening_draft.md) | 사후 교정(DSL/PPI)이 아니라 전체 코딩 **전** 예측 |
| E5 | 검증을 통과한 오픈 모델 4개(BA 0.82–0.90)는 급진우파 − 주류 이민 주목도 비교에서 왜곡이 확립되지 않았고, Qwen2.5-7B는 SESOI 1pp 안의 동등성이 확립됐다. Granite·Phi는 6/6 셀 +방향(판정 불가), OLMo는 급진우파 민감도가 11–14%p 낮다(2차) | **확립** (H1 지지 안 됨). 선언문 간 오류 이질성으로 CI가 보정 예상(약 1pp)보다 4–10배 넓음. Qwen 동등은 "어쩌면" 매핑에 견고하지 않음 | `prereg-v3-pimpo` · [docs/22](docs/22_pimpo_results_interpretation.md) | 같은 텍스트 크라우드 기준, 14개국 |
| E6 | 같은 텍스트 기준에서 현재 프런티어 모델도 동등한가 | **미정: 결정적 검정** | API 실험 (별도 진행) | — |
| E7 | 오류가 텍스트가 아니라 정당 평판에서 오는가 | **미정** (프롬프트 정당 단서 없음·실제·바꿈 무작위 실험, 사전등록 G 예정) | [docs/21 설계 메모](docs/21_mechanism_design_memo.md) | Vallejo Vera & Driggers (2025) 확장: 다국어·정당가족, 비교 추정량으로 연결 |
| E8 | 같은 계열에서 모델이 커질수록 집단별 오류 차이가 줄어드는가, 오히려 커지는가 | **미정** (Qwen2.5 0.5–14B · OLMo-2 1–13B · Granite-3.3 2–8B × 정당 단서, 사전등록 G 예정) | [docs/21](docs/21_mechanism_design_memo.md) | — |

**격하·철회한 주장:**
- 상관 기반 검증의 한계식(P6)은 알려진 결과(Bland–Altman) → 헤드라인이 아니다.
- UK 전문가 vs 크라우드 차이는 감쇠(기울기 0.62) → 비동등 증거가 아니다.
- v1 킬테스트 규칙은 귀무에서 100% 거짓 판정 → 폐기.
- 경위는 모두 [docs/12](docs/12_red_team.md)에 있다.

## 사전등록 잠금

| 잠금 | 시각 (UTC) | 사전등록 | 상태 |
|---|---|---|---|
| `prereg-v3-uk` | 2026-09-14 05:50 | [docs/13](docs/13_prereg_v3_draft.md) | 결과 [docs/15](docs/15_uk_results_interpretation.md) |
| `prereg-v3-pimpo` | 2026-09-14 09:09 | [docs/14](docs/14_prereg_pimpo_draft.md) | 결과 [docs/22](docs/22_pimpo_results_interpretation.md) |
| `prereg-benoit-e` | 2026-09-14 10:38 | [docs/16](docs/16_prereg_benoit_e_draft.md) | 결과 [docs/17](docs/17_benoit_e_results_interpretation.md) |
| `prereg-audit-f` | 2026-09-14 11:19 | [docs/18](docs/18_prereg_audit_forecast_draft.md) | PImPo 확증: 지지 안 됨 ([docs/22 §5](docs/22_pimpo_results_interpretation.md)) |
| `decision-map-19` | 2026-09-14 12:04 | [docs/19](docs/19_decision_map.md) | 결과 → 결론 → 투고처 지도 (PImPo 결과 전) |
| `prereg-audit-f2` | 2026-09-14 12:10 | [docs/20](docs/20_prereg_audit_screening_draft.md) | PImPo 확증: 지지 안 됨 ([docs/22 §5](docs/22_pimpo_results_interpretation.md)) |

- 잠금은 사전등록 문서·코드·자료·보정 결과의 sha256이다([LOCKS](docs/decisions/LOCKS.md)).
- `make locks`로 전부 확인한다. 저장소에 없는 자료 파일의 해시도 포함되므로, 자료를 같은 스크립트로 다시 만들면 검증된다.
- 잠금 뒤의 일탈과 버그는 모두 [docs/12 §D](docs/12_red_team.md)에 기록한다.

## 재현

```bash
pip install -e ".[dev,data]"        # 분석·테스트. LLM 라벨링은 ".[label]"(Anthropic) 또는 별도 vLLM 환경
make test                           # 테스트 전체
make locks                          # 사전등록·결정 지도 잠금 전부 검증
make uk                             # UK 확증 (라벨링 포함, GPU)
make pimpo                          # PImPo 확증 (라벨링 포함, GPU)
make benoit-e                       # 사전등록 E: 자료 준비 + 분석 (잠금 필요)
make audit-f-uk / make audit-f-pimpo  # 사전등록 F (F′는 python v3/run_audit_screening.py --study uk|pimpo)
make sims                           # 방법론 시뮬레이션 S1–S3
```

## 데이터 접근 (저장소에 포함하지 않음)

재배포가 금지되거나 제한된 자료(원문 텍스트, 코더 자료, LLM 라벨, 실행 로그)는 이 저장소에 없다. `data/`와 로그는 `.gitignore`로 제외한다. `results/`에는 집계 결과표만 있다.

| 자료 | 받는 곳 | 준비 스크립트 |
|---|---|---|
| Benoit et al. (2016) UK 코더 자료 | github.com/kbenoit/CSTA-APSR | `data_prep/benoit2016.py`, `data_prep/benoit2016_multirater.py` |
| PImPo (Lehmann & Zobel 2018) | manifesto-project.wzb.eu/datasets/pimpo | `data_prep/pimpo.py`, `data_prep/pimpo_multirater.py` |
| Manifesto Corpus 텍스트 · MPDS2024a | Manifesto Project API (개인 API 키 필요) | `data_prep/manifesto_texts.py`, `data_prep/pimpo_sentences.py` |
| Benoit et al. (2026) 복제 자료 | Harvard Dataverse doi:10.7910/DVN/XY1FFE (재배포하지 않음, [데이터 카드](docs/data_cards/benoit2026.md)) | `data_prep/benoit2026.py` |

API 키는 저장소 밖(`~/.config/manifesto/api_key`)에 두고, 스크립트에는 환경변수로만 넘긴다.

## 지도

| 문서 | 내용 |
|---|---|
| [00_assessment](docs/00_assessment.md) · [01_positioning](docs/01_positioning.md) | 평가, 문헌 지도와 차별점 (§3b) |
| [02_theory](docs/02_theory.md) · [03_eav_method](docs/03_eav_method.md) | error regression(P0)과 명제, 감사 절차 |
| [09](docs/09_ajps_substantive_design.md) · [11](docs/11_ajps_case_and_engagement.md) | AJPS 설계, Benoit et al. (2026)과의 관계·어조 규칙 |
| [12_red_team](docs/12_red_team.md) | 자체 점검, 검토자 반론, **개발·일탈 기록 D1–D46** |
| [13](docs/13_prereg_v3_draft.md) · [14](docs/14_prereg_pimpo_draft.md) · [16](docs/16_prereg_benoit_e_draft.md) · [18](docs/18_prereg_audit_forecast_draft.md) · [20](docs/20_prereg_audit_screening_draft.md) | 사전등록 (잠김) |
| [19_decision_map](docs/19_decision_map.md) | 결과 → 결론 → 투고처 지도, "검증을 통과한 모델" 정의, AJPS 최소 조건 (PImPo 결과 전 잠김) |
| [21_mechanism_design_memo](docs/21_mechanism_design_memo.md) | 사전등록 G(모델 크기 × 정당 단서) 설계 메모 |
| [15](docs/15_uk_results_interpretation.md) · [17](docs/17_benoit_e_results_interpretation.md) · [22](docs/22_pimpo_results_interpretation.md) | 결과 해석 (UK · Benoit 재분석 · PImPo) |
| [data_cards/](docs/data_cards/) · [decisions/](docs/decisions/) | 데이터 카드, ADR·잠금 |
| [paper/ajps/outline.md](paper/ajps/outline.md) | 원고 뼈대 v0.2 (원장에 맞춤) |
| 보관 (대체됨) | [04 파일럿 사전등록](docs/04_pilot_prereg.md), [10 킬테스트 ①](docs/10_killtest1_prereg.md), `killtest/`, `pilot/`, `v3/orchestrate_background.sh` — 각 파일 머리에 대체 사유 표시 |
| 기타 | [05 자료 계획](docs/05_data_plan.md) · [06 투고 전략](docs/06_publication_strategy.md) · [07 로드맵](docs/07_roadmap.md) · [08 위험](docs/08_risk_register.md) · [templates/](templates/) · [lit/references.bib](lit/references.bib) |

## 코드

```
src/eav/latent.py          다평정자 잠재계층 (인간 코더만으로 참값 정의)
src/eav/v3.py              집단 조건부 DIF 판정: 군집 bootstrap, 모수적 귀무 max-T, TOST 3분류      (UK)
src/eav/v3_weighted.py     설계 가중 표본과 귀무 중심 보정판                                    (PImPo)
src/eav/contrast.py        연속 점수 도구 간 비교 결론 동등성, 감쇠 교정                         (사전등록 E)
src/eav/audit_forecast.py  인간 감사로 비교 왜곡 예측 vs 합산 정확도                            (사전등록 F)
src/eav/audit_screening.py 판단 보류 선별, 사후 기대 손실 선택                                  (사전등록 F′)
src/eav/audit.py · misclass.py · simulate.py   error regression, 오분류 대수, 시뮬레이션 DGP
label/label.py             LLM 라벨러 (mock · vLLM · Anthropic sync/batch, 캐시, 선택지 섞기)
v3/                        연구별 실행기 · 보정 · 잠금 도구(lock.py)
data_prep/                 자료 준비 (조건 검사 포함)
tests/                     명제 수치 예시 · 판정 규칙 · 실행기 종단 테스트
```

## 방법론 배경 (시뮬레이션, 8개 후보 모델)

- **불변 오류이면** balanced accuracy가 정당 격차 편향 최적 모델을 100% 고른다.
- **차등 오류이면** 어떤 라벨 지표든 25–30%로 떨어진다.
- **DSL/PPI로 교정하면** 소수정당 격차 연구에서 accuracy 기준 선택은 유효표본을 최대 28% 잃는다 (P4).
- **감사 기반 기준**은 n ≈ 200–400 이상에서, 차등 오류가 있을 때만 라벨 지표를 이긴다. 편향 불편추정(B̂²−V̂)은 선택기로 실패한다 (ADR-0004). UK 탐색(E4)에서도 작은 감사의 winner's curse가 확인됐다.

## 운영 원칙

1. **질문은 고정한다.** 불리한 결과는 질문을 바꿀 이유가 아니라 방법을 보강할 이유다.
2. **사전등록은 결과보다 먼저 잠근다.** 잠금 뒤 변경은 일탈로 기록하고, 일탈 전 결과도 함께 보고한다.
3. **불리한 결과도 같은 비중으로 보고한다.** 우호적 결과(E3, E5)와 비지지 결과(E4)도 원장에 그대로 싣는다.
4. **주장은 원장의 "확립" 범위 안에서만 한다.** 선행연구(01_positioning §3b)가 이미 말한 것을 신규로 주장하지 않는다.
5. **재현성:** seed 고정, 모델 revision 전체 해시 고정, 원응답 캐시, `make`로 재생성, 모든 숫자는 결과 CSV에서 옮긴다.
