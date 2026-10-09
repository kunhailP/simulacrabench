# SimulacraBench 내부 레포

NeurIPS 2026 SimulacraBench 대회 준비용이다. Codabench 대회 17822, 공식 레포는 `SituatedEvals/public`이다.

- 대회 원문 조사와 규정·일정·채점: [`RESEARCH.md`](RESEARCH.md)
- **다음 세션은 먼저 읽을 것: [`docs/HANDOFF.md`](docs/HANDOFF.md)**
- 실험 기록: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- 전략: [`docs/STRATEGY.md`](docs/STRATEGY.md)
- 연구 설계서 원본: [`docs/research_design_2026-10-08.pdf`](docs/research_design_2026-10-08.pdf), 검토: [`docs/DESIGN_REVIEW.md`](docs/DESIGN_REVIEW.md)
- 방법 제안 v0.1 (다변량 소지역 추정, H2는 내부 기각): [`docs/METHOD_PROPOSAL.md`](docs/METHOD_PROPOSAL.md)
- 제출 프로토콜과 제출 기록: [`docs/SUBMISSION_PROTOCOL.md`](docs/SUBMISSION_PROTOCOL.md), [`docs/submissions/`](docs/submissions/)

## 현재 상태 (2026-10-09)
- **현재 제출본: `sub/v8`** (zip은 git에 없음. `make zip SUB=sub/v8`로 다시 만들고, 체크포인트는 `tools/fetch_tabicl.sh`로 받는다). 구성은 v7 + TabICLv2 + TabICL 분기 분해 + 기하평균 결합이고, 시간 감시와 v7·marginal fallback을 갖췄다.
  - 근거는 공개 실데이터 실험실(`docs/LAB.md`): GSS +0.0022 ± 0.0001, UNHCR식 GSS +0.0024 ± 0.0001 (v7 대비, 3회 반복, 쌍대 SE)
  - 공식 score.py: sandbox 세 기관 × phase 1·2 전부 PASS (3090 단독)
  - 제출 기록과 예상: `docs/submissions/v8.md`
- **다음 목표: UNICEF 0.53 / WB 0.51 / UNHCR 0.95**, WB와 UNHCR은 별도 트랙(`docs/HANDOFF.md`).
- **할 일:** TabICL 체크포인트(`jingang/TabICL` @ `4dcd344`)를 Test 시작(11-14) 전에 주최 측에 신고한다(Terms 9).
- **v8 Dev 결과 (submission 971475): Grand 0.65 (0.6467) / UNICEF 0.52 / WB 0.49 / UNHCR 0.93.** v7보다 UNHCR이 한 칸 올랐고, 떨어진 기관은 없다. 이득의 크기는 반올림 때문에 알 수 없다(`docs/submissions/v8.md`).
- 이전 v7: Dev Grand 0.64 / UNICEF 0.52 / WB 0.49 / UNHCR 0.92

## 구조

```
RESEARCH.md          원문 조사 요약 (규정, 일정, 채점, 리더보드, 출처 간 불일치)
raw/                 수집한 원문 (사이트, Codabench terms/리더보드, GitHub PR/issue, 참고 글)
  repo/              공식 레포 clone (gitignore; make setup이 고정 commit으로 받음)
sub/                 제출본. 각 폴더의 main.py 하나가 제출 단위다.
  v1                 marginal + backoff 격자 + linear + MLP, OOF 혼합
  v2                 경험적 베이즈 backoff, fold bagging, 문항별 temperature
  v3                 다변량 FH (MSAE) 프로토타입 — H2 내부 기각
  v4                 v2 + PREDICT 부모 gate 분해
  v5                 v4 + 쌍 상호작용 top-k 선별 수축
  v6                 v4 + 쌍 상호작용 local fdr 선별 수축
  v7                 v6 + GPU 블렌딩 + Lindsey local fdr (점수 같고 더 빠름)
  v8                 v7 + TabICLv2(동봉) + 분기 분해 + 기하평균 결합  ← 현재 후보
tools/
  evaluate.py        in-process 채점: 세 기관 skill(정확값), 시간, 오라클 headroom
  world.py           오라클을 계산할 수 있는 합성 world (시나리오 묶음, 순서형·무응답 구조)
  headroom.py        오라클 대비 초과 KL을 문항·gate·셀 크기·유형별로 분해
  learning_curve.py  학습 행을 늘려 근사 오차와 추정 오차를 분리
  proto_latent.py    조건부 IRT 잠재요인 모형 프로토타입 (H3, 이득 없음)
  bundle.sh          한 제출본을 모든 world에서 평가
  sim2.py            현실적인 합성 데이터 생성기 (GIVEN 효과, 노이즈 gate, wave 코딩)
  schema_stats.py    스키마 통계 (U, 문항 수, gate 구조)
experiments/         EXPERIMENTS.md의 결과를 낸 일회성 스크립트 (x1–x9, 각 파일 첫 줄에 요지)
research/            실데이터 실험실의 후보 모델과 실험 스크립트 (docs/LAB.md)
tools/proxy_gss*.py  공개 GSS를 대회 형식으로 바꾼 실험실 데이터 (UNHCR식 기록 변형 포함)
tools/lab.py         반복 분할 + 쌍대 비교 평가, tools/oof.py·blend.py는 모델별 캐시와 조합 실험
tools/fetch_tabicl.sh  v8 체크포인트를 고정 commit에서 받고 SHA-256 검증 (git에는 넣지 않음)
data/                sandbox / sim2 / sim3 / worlds (gitignore, make data / make worlds로 재생성)
docs/
```

## 사용법

```
make setup                      # venv, 공식 레포, 연습 데이터
make worlds                     # 오라클 world 8종 × 3기관 (CPU 병렬)
make eval SUB=sub/v7            # sim2 채점
make eval SUB=sub/v7 DATA=data/worlds/base   # 오라클 headroom도 출력
tools/bundle.sh sub/v7          # 모든 world에서 평가
make zip SUB=sub/v7             # sub/v7.zip + 공식 zip 검사
make official SUB=sub/v7        # 공식 score.py (subprocess, 네트워크 차단)
make lab-data                   # 실데이터 실험실: GSS 다운로드(해시 검증) → data/proxy/gss, gss_hcr
make tabicl                     # v8 체크포인트 (고정 commit, SHA-256 검증)
make lab SUB=sub/v8 LAB=gss     # 반복 분할 평가, make compare A=v7 B=v8 로 쌍대 비교
```

## 제출 전 체크리스트
- [ ] `make zip` OK, `make official` 3/3 PASS, phase 2 PASS
- [ ] 총 시간 < 900s (H100 기준; 3090 기준이면 여유를 둘 것)
- [ ] 외부 HF 모델을 쓰면 commit hash를 고정하고 Test 시작 전에 신고 (Terms 9)
- [ ] Test 제출 시 4쪽 방법 설명서 (Terms 7)

## 공개 전 할 일 (대회 종료 후)
- 레포는 대회 기간 동안 비공개로 둔다. 수상하면 비상업 연구 재현 라이선스로 공개한다(Terms 7).
- 공개 전에 `raw/codabench/`의 리더보드 스냅샷(다른 팀 이름 포함)을 뺀다.
