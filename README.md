# SimulacraBench 내부 레포

NeurIPS 2026 SimulacraBench 대회 준비용이다. Codabench 대회 17822, 공식 레포는 `SituatedEvals/public`이다.

- 대회 원문 조사와 규정·일정·채점: [`RESEARCH.md`](RESEARCH.md)
- 실험 기록: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- 전략: [`docs/STRATEGY.md`](docs/STRATEGY.md)
- 제출 프로토콜 (리뷰 승인 필수): [`docs/SUBMISSION_PROTOCOL.md`](docs/SUBMISSION_PROTOCOL.md)

## 구조

```
RESEARCH.md        원문 조사 요약 (규정, 일정, 채점, 리더보드, 출처 간 불일치)
raw/               수집한 원문 (사이트, Codabench terms/리더보드, GitHub PR/issue, 참고 글)
  repo/            공식 레포 clone (gitignore; make setup이 고정 commit으로 받음)
tools/
  sim2.py          현실적인 합성 데이터 생성기 (GIVEN 효과, 노이즈 gate, wave 코딩)
  evaluate.py      in-process 채점: 세 기관 skill(정확값)과 시간
  schema_stats.py  스키마 통계 (U, 문항 수, gate 구조)
sub/
  v1/              최초 제출본
  v2/              경험적 베이즈 backoff + bagging + temperature
data/              sandbox / sim2 (gitignore, make data로 재생성)
docs/EXPERIMENTS.md
```

## 사용법

```
make setup                      # venv, 공식 레포, 연습 데이터
make eval SUB=sub/v2            # sim2 채점
make eval SUB=sub/v2 DATA=data/sandbox
make zip SUB=sub/v2             # sub/v2.zip + 공식 zip 검사
make official SUB=sub/v2        # 공식 score.py (subprocess, 네트워크 차단)
```

## 제출 전 체크리스트
- [ ] `make zip` OK, `make official` 3/3 PASS, phase 2 PASS
- [ ] 총 시간 < 900s (H100 기준; 3090 기준이면 여유를 둘 것)
- [ ] 외부 HF 모델을 쓰면 commit hash를 고정하고 Test 시작 전에 신고 (Terms 9)
- [ ] Test 제출 시 4쪽 방법 설명서 (Terms 7)
