# SimulacraBench 원문 조사 정리 (2026-10-08 KST 기준 수집)

원문은 모두 `raw/` 아래에 그대로 저장했다. 이 문서는 요약과 교차검증 결과다.

## 0. 원문 위치

| 출처 | 로컬 파일 | 비고 |
|---|---|---|
| 공식 사이트 simulacrabench.org | `raw/site.html`, `raw/site.txt` | 일부 수치가 최신 스키마와 다름(§5) |
| GitHub `SituatedEvals/public` 전체 | `raw/repo/` (git clone, HEAD `f980ef5`, 2026-10-07) | README, FAQ, config, score.py, make_sandbox.py, 스키마 3+1개, 베이스라인 3개, 튜토리얼 6개 언어, zip 검사기 |
| GitHub PR/Issue/댓글/브랜치 | `raw/github/*.json` | PR #3, #4 open / Issue #1, #2 closed / 브랜치 `marginal-reference`(=PR #4) |
| Codabench 대회 17822 메타데이터·**Terms 전문** | `raw/codabench/comp.json`, `terms.md`, `page_Overview.md` | Terms는 API로 전문 확보 |
| Codabench Dev 리더보드 (92개 엔트리) | `raw/codabench/leaderboard*.json`, `leaderboard_table.txt` | 2026-10-08 03:2x UTC 스냅샷 |
| 튜토리얼(en) 본문+출력 | `raw/tutorial_en_extracted.txt` | |
| 스키마 통계 | `raw/schema_stats.txt` (`tools/schema_stats.py`로 재생성) | |

확보하지 못한 것: Codabench 포럼(로그인 필요), UNHCR 블로그 글(봇 차단 403).

## 1. 일정 (출처마다 다름. Codabench가 실제 시스템)

| | config.yml / README / 사이트 | Codabench 실제 설정 (UTC) | 한국시간 |
|---|---|---|---|
| Dev 단계 | ~2026-11-13 | 2026-08-17 00:00 ~ **2026-11-14 07:55** | ~11/14(토) 16:55 |
| Test(Final) | 2026-11-14 ~ 11-18 | **2026-11-15 00:00 ~ 2026-11-19 07:55** | 11/15 09:00 ~ 11/19 16:55 |

- Dev는 하루 1회 제출이고 총 횟수 제한은 없다. 오늘(10/8) 기준 남은 제출은 약 37회다.
- Test는 팀당 **1회**, `execution_time_limit` 3600s.
- 두 일정 사이에 하루 공백이 있다. 마감 직전 일정은 이메일로 확인하는 게 안전하다.

## 2. 규정 핵심 (Codabench Terms 원문: `raw/codabench/terms.md`)

1. 제출물은 루트에 `main.py`(`predict(frame, schema)`)를 둔 ZIP이다. `requirements.txt`, `models.txt`, 번들 파일은 선택이다. 로드 실패, 계약 위반, 시간 초과면 점수가 없다.
2. 런타임 네트워크는 차단된다. 가중치는 ZIP 번들(1GB 이하)이나 `models.txt`(HF repo id, 사전 다운로드)로 가져온다.
3. **Test 제출에는 4쪽 방법 설명서를 함께 내야 한다.** 상위 3팀은 비상업 연구 재현이 가능한 라이선스로 소스코드를 제출한다.
4. **외부 사전학습 모델은 Test 단계가 열리기 전에 참가자가 지정한 고정 commit hash로 공개 다운로드 가능해야 한다.** 라이선스도 비상업 연구 재현을 허용해야 한다. → HF 모델을 쓸 거면 revision 해시를 미리 고정하고 신고할 것.
5. 데이터 재식별과 유출(loss-curve 인코딩, 에러 메시지, 컨테이너 쓰기 등)은 금지되고, 위반 시 실격과 법적 책임을 진다. 반복되는 Dev 점수 피드백으로 라벨을 추론하는 것도 금지다(사이트 명시).
6. 한 사람은 한 팀에만 속할 수 있다. 팀을 나눠 협업 제출하면 전원 실격이다. 팀 구성은 Test 제출 시점에 동결된다.
7. 1위 동점 판정: paired-bootstrap 유의성 기준 안에 들면 상금을 나눈다.
8. 연락처: Terms에는 `neurips-un-benchmark@stanford.edu`, 사이트에는 `team@simulacrabench.org`로 서로 다르게 적혀 있다.

상금은 Grand $10,000(3개 skill 평균 1위), Roots of the Data $2,000(14개 데이터 수집국에 전원 소속된 팀), 기관별 트랙 각 $1,500이다. 한국 팀은 Roots 상금 대상이 아니다.

### 런타임 학습 허용 여부 (붙여준 분석에서 '미확정'이라 한 부분)
- 사이트 Rules 원문은 **"Code can train on the visible rows of the matrix before predicting."**
- README 원문은 "There is no runtime training hook, and training must happen **offline**."
- 공식 베이스라인 3개는 모두 `predict()` 안에서 visible 행으로 빈도와 MI를 계산한다(= 런타임 적합).
- 실데이터는 컨테이너 밖으로 나오지 않으므로 실데이터로 "offline 학습"하는 것은 원천적으로 불가능하다.
- 해석: README 문장은 "학습용 별도 훅이 없고 시간이 예산에 포함된다"는 뜻으로 읽힌다. 런타임 적합은 사이트가 명시적으로 허용하고 베이스라인도 그렇게 한다. 그래도 서면 확인은 이메일 한 통이면 된다.

## 3. 채점 (score.py 원문, Dev 채점 프로그램은 commit `a3e0a2a`에 고정)

- 셀마다 독립적으로 `log p[정답]`을 계산하고, 평균을 낸 뒤 `skill = 1 + mean_logp / U`로 환산한다.
- `U` = PREDICT 문항들의 `log K` 평균(셀 가중이 아니라 문항 평균). K는 gate가 있는 문항이면 sentinel `NA_GATED`를 포함한다.
- 제출 벡터는 정규화한 뒤 `p = 1e-3 + (1 - K·1e-3)·p`로 섞는다. 완전히 틀린 확신 셀 하나의 비용은 −6.9다.
- **gate가 있는 문항에만** 마지막 슬롯(`NA_GATED`)이 있다. gate가 없으면 슬롯도 없다.
- 종합 점수 = 세 기관 skill의 단순 평균.

**Dev 점수 노이즈는 사실상 0이다.** Laplace 스케일 = `6.91/(n_hidden·U)/ε(10)`이고, 계산하면 UNICEF 0.0003, WB 0.0007, UNHCR 0.0009다. 반올림 단위 0.01보다 훨씬 작으므로, Dev 리더보드는 **기관별 정확값을 0.01로 반올림한 값**으로 보면 된다. Grand 열은 반올림된 세 값의 평균이다(0.52+0.50+0.94)/3=0.6533.

검증된 스키마 기준 수치 (`raw/schema_stats.txt`):

| | UNICEF v1.1 | World Bank v1.0 | UNHCR v1.3 |
|---|---:|---:|---:|
| 행 수 (train/dev/test) | 19,846 (14,885/1,488/3,473) | 6,927 (5,195/520/1,212) | 10,349 (8,279/621/1,449) |
| GIVEN / PREDICT / EXCLUDE | 7 / 12 / 0 | 9 / 76 / 21 | 17 / 213 / 55 |
| gate 있는 PREDICT | 7 (부모가 GIVEN인 것 5: Q13→Q14~18) | 30 (부모 GIVEN 27: group·employmentStatus·age) | 156 (부모 GIVEN 1) |
| U (nats) | 1.562 | 2.018 | 1.262 |
| 문항 1개의 종합점수 가중 ∝ 1/(문항수·U) | 0.0534 | 0.0065 | 0.0037 |

UNICEF 문항 1개 ≈ UNHCR 문항 14.3개 ≈ WB 문항 8.2개 (같은 logloss 개선 기준). 붙여준 분석의 14.3배는 맞다.

## 4. 리더보드 현황 (2026-10-08 스냅샷, 참가자 149명 / 제출 654건 / 엔트리 92개)

| 순위 | 팀 | Grand | UNICEF | WB | UNHCR |
|---|---|---:|---:|---:|---:|
| 1 | SAPIENSQ (mgjeon, 한국) | 0.6533 | 0.52 | 0.50 | 0.94 |
| 2 | techsq | 0.6533 | 0.52 | 0.50 | 0.94 |
| 3~5 | in2ai(러시아), SAPIENSQ(다른 계정), riccardocadei | 0.6500 | 0.52 | 0.50 | 0.93 |
| ~40위 | 다수 | 0.640 | 0.52 | 0.49 | 0.91 |
| 84~89 | 6명 동일 점수 (공식 marginal 베이스라인으로 추정) | 0.3567 | 0.26 | 0.33 | 0.48 |

읽을 수 있는 것:
- **UNICEF 0.52와 WB 0.49~0.50은 상위 50팀이 거의 같다.** 0.01 해상도에서는 천장처럼 보인다. 순위 차이는 대부분 **UNHCR(0.91→0.94)**에서 난다.
- 상위 40팀이 0.640~0.653 안에 몰려 있다. 최종 순위는 0.00x 차이와 bootstrap 동점 판정으로 갈릴 가능성이 크다.
- marginal(0.36)에서 상위권(0.65)까지의 차이는 대부분 gate 구조와 GIVEN 조건부 분포에서 나온다. UNHCR에서 +0.46, UNICEF에서 +0.26.
- 같은 팀(SAPIENSQ)이 계정 두 개로 올라와 있다. 규정상 개인은 한 팀만 가능하다(우리와는 무관한 관찰).

## 5. 출처 간 불일치·변경 이력 (원문 대조로 확인)

1. **사이트 수치가 오래됐다.** WB PREDICT는 사이트 73 / 스키마 76, U는 사이트 1.758 / 실제 2.018이다. UNICEF U는 사이트 1.528 / 실제 1.562다. 스키마를 기준으로 삼을 것.
2. 사이트에는 "Dev는 30% 응답자로 채점"이라고 되어 있다. 실제 DEV 비율은 UNICEF 7.5%, WB 7.5%, UNHCR 6%다(스키마 split 기준).
3. **Dev 채점·ingestion 프로그램은 `a3e0a2a`(8/16)에 고정**돼 있다. 그 뒤 `unhcr.json`이 v1.2에서 v1.3으로 바뀌었다(`1bb2d46`, 8/17). 문항 4개(`s3_14_visit_importance_lbl`, `s3_23a/b_safety_rank*`, `s6_12_children_schooling_lbl`)는 `"Not recorded"` 레벨이 삭제되고 gate가 추가됐다. score.py는 바뀌지 않았다. → `predict()`는 반드시 **전달받은 schema에서** 옵션과 gate를 읽어야 한다. 하드코딩하면 안 된다.
4. **PR #4 (open, 미병합, 10/6)**: 기준을 uniform에서 visible-row crowd marginal로 바꾸는 제안이다. marginal 베이스라인이 0점이 된다. 병합되면 기관별 가중치가 바뀐다(분모가 marginal 손실이 되므로 marginal 대비 개선 비율로 평가). 병합 여부를 계속 확인할 것.
5. Issue #1에서 주최측은 "워커는 제출 간 재사용되지 않는다"고 답했다.
6. Runner 이미지: README는 Python 3.13이고 Codabench 메타는 `python:3.12-slim`이다. 3.12와 3.13 양쪽에서 돌아가는 코드로 작성할 것.
7. 메모리 "16 GB": README의 Docker 재현이 `--memory=16g`(호스트 RAM)이다. H100 VRAM 제한은 명시돼 있지 않다.

## 6. 붙여준 ChatGPT 분석 검증

| 주장 | 판정 |
|---|---|
| 행/문항 수(37,122행, 301 PREDICT, 분기 156/7/30) | ✅ 스키마와 일치 |
| skill 식, 기관 1/3 가중, UNICEF 14.3배 | ✅ |
| UNHCR gate 일치도 중앙값 0.83 (0.29–0.98), 59개 다중선택 문항의 round1=0 / round2=NA_GATED | ✅ unhcr.json description 원문 |
| PR #4 미병합 | ✅ |
| "리더보드·Terms 확인 못 함" | ➜ 이번에 둘 다 확보 (§2, §4) |
| "런타임 학습 허용 불확실" | ➜ 사이트 원문이 명시적으로 허용 (§2). 서면 확인만 남음 |
| Dev 점수 노이즈 언급 없음 | ➜ 사실상 0이고 0.01 반올림만 있음 (§3) |
| Q08(접종 여부)이 숨겨져 Q09/Q10 gate가 확률적 | ✅ Q08은 PREDICT |
| 일정 11/13, 11/14~18 | ⚠️ Codabench 실제 설정은 11/14 07:55 UTC, 11/15~11/19 07:55 UTC |

## 7. 원문에서 직접 나오는 전략적 사실

1. **셀마다 독립 채점이다.** 따라서 최적해는 문항별 조건부 주변분포 `P(y_j | GIVEN, visible rows)`다. 응답자 안에서 답들이 서로 일관될 필요는 없다. gate·joint 구조는 이 주변분포를 잘 추정하기 위한 도구일 뿐이다.
2. **gate를 하드 규칙으로 쓰면 위험하다.** 공식 `external_model` 베이스라인은 부모가 GIVEN이면 0/1로 확정한다. UNHCR에서는 gate와 실제 기록의 일치도가 0.29~0.98이다. UNICEF와 WB의 일치도는 공개되지 않았다. → `P(NA_GATED | 부모값)`은 학습 행에서 **경험적으로** 추정하고 gate는 사전정보로만 쓸 것. 확신 오답 1셀 = −6.9.
3. **실데이터를 볼 수 없으니 하이퍼파라미터는 런타임에 고른다.** `predict()` 안에서 visible 행으로 K-fold CV를 돌려 스무딩 강도, 모델 선택, temperature 같은 것을 자동으로 정하는 구조가 핵심이다. Dev 리더보드는 0.01 해상도에 하루 1회라 미세 튜닝 신호로는 부족하다. FAQ도 "random CV on TRAIN이 적절"하다고 말한다.
4. **Phase 2에서는 DEV 행이 학습에 추가된다**(UNICEF +10%, WB +10%, UNHCR +7.5%). 시간 예산은 3600s로 4배다. 런타임 적합 구조라면 자동으로 이득을 본다.
5. `check_submission_zip.py`의 smoke test는 **3행짜리 프레임**(학습 2행)이다. 표본이 극소일 때 fallback이 없으면 검사에서 실패한다.
6. 시간 예산은 import, 모델 로딩, `predict()` 3회를 합친 것이다(Dev 900s). 오류는 HSCC 코드만 돌려주고 traceback은 없다. 로컬 `score.py`와 `--docker`로 먼저 확인할 것.
7. 리더보드상 차별화 포인트는 UNHCR이다(213문항, 그중 73개가 `s3_11` 하나에 매달림). `s3_11`과 `s3_15`, `s6_10`, `s6_3` 같은 핵심 부모의 분포를 잘 맞히고, wave별 sentinel 코딩 차이(round1=0, round2=NA_GATED)를 `wave`(GIVEN) 조건으로 처리하는 것이 직접적인 레버다.
