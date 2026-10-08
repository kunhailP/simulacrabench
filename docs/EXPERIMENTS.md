# 실험 기록

Skill은 un-noised 정확값이다. 이 표의 숫자는 모두 **가짜 데이터** 기준이다. 실데이터 점수는 Dev 리더보드에서만 확인할 수 있다.

- `sandbox`: 공식 `make_sandbox.py`. GIVEN 정보량이 거의 없고 gate를 100% 지킨다.
- `sim2`: `tools/sim2.py` (seed 1). GIVEN 주효과·상호작용, GIVEN과 상관된 잠재특성, gate 일치도 0.55–0.99, UNHCR wave1 "0" 코딩을 넣었다.

| 날짜 | 버전 | 데이터 | UNICEF | WB | UNHCR | 평균 | 시간(3090) | 메모 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 10-08 | baseline/marginal_counts | sim2 | 0.2053 | 0.1881 | 0.1942 | 0.1959 | 1s | |
| 10-08 | v1 | sim2 | 0.4125 | 0.3797 | 0.4232 | 0.4051 | 162s | marg+backoff격자+linear+MLP, OOF 혼합 |
| 10-08 | v2 | sim2 | 0.4139 | 0.3807 | 0.4258 | 0.4068 | 198s | backoff→경험적 베이즈(DM 주변우도), fold bagging, 문항별 temperature |
| 10-08 | baseline/marginal_counts | sandbox | 0.16 | 0.24 | 0.43 | | | score.py, 노이즈 포함 |
| 10-08 | v1 | sandbox | 0.50 | 0.45 | 0.56 | | 280s | score.py PASS 3/3, phase2 PASS |

## 실데이터 Dev 리더보드 제출

| 날짜 | 버전 | Grand | UNICEF | WB | UNHCR | 비고 |
|---|---|---:|---:|---:|---:|---|
| (아직 없음) | | | | | | |

## 관찰
- sim2에서 MLP 가중치가 0.7–0.9로 지배적이다. sim2의 생성 구조 때문일 수 있어서 실데이터에서도 그런지는 모른다.
- sim2에서는 L2 격자와 MLP epoch가 격자 끝값에서 선택된다. 실데이터에서는 런타임 CV가 알아서 고르므로 격자를 넓게 두는 게 안전하다.
