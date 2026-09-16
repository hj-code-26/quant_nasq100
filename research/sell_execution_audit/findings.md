# 매도 실행 경로 감사 (2026-09-16)

기준 커밋: `6a0d843` (= 원래 분석 기준 = 수정 직전 기준). 수정본: 이 브랜치 HEAD.

```bash
python research/sell_execution_audit/scenarios.py    # 55 시나리오, 가짜 브로커·가짜 시계
python research/sell_execution_audit/ab_verify.py    # 같은 파일을 6a0d843 과 HEAD 에서 각각 실행해 비교
```
둘 다 `research.isolation.guard()` 아래에서 돈다 — 자격증명 제거, 운영 DB·로그 쓰기 차단,
토스 호스트 차단. `ab_verify.py` 는 끝에 운영 파일 해시를 다시 확인한다.

**A/B 결과 (2026-09-16): 수정으로 통과 전환 24 · 원래도 통과 31 · 여전히 실패 0 · 회귀 0.**

## 1. 운영 증거로 확인된 것

읽기 전용으로 본 자료: `autotrade.log`(2952줄, 09-09~09-16), `trading_decisions.db` 복사본.

| 관측 | 분류 | 근거 |
|---|---|---|
| run 52~64 에서 매도 주문 0건 | **A (조건 미충족)** | `.env` 에 전략값이 없다 → `STOP_LOSS_PCT=0`(끔), `MOMENTUM_EXIT=0`(끔). `equity` 5행 < `VOL_WINDOW+1=61` → `_vol_target_cap()` None → 노출 축소 상한 자체가 없다. 보유 7종목 전부 1~7 거래일 < `MAX_HOLD_DAYS=20`. **즉 지금 계좌에는 발동 가능한 매도 규칙이 하나도 없다.** |
| WDAY 매도 3회 연속 400 거절 | **D (브로커 거절)** | orders id 1·3·5, `소수점 수량 주문은 미국 주식 시장가 매도에만` — 현 체크아웃에는 `whole` 정수 주 가드가 이미 들어가 있어 **수정 완료** |
| TEAM 0.09306주 매도가 run 13·14 에 **두 번** 제출 | **E/G (접수됐으나 미체결 + 상태 불명)** | orders id 9·10, 서로 다른 brokerId. 같은 수량이 다음 사이클에도 남아 있었다 = 1차 매도 미체결. 시스템은 `submitted` 만 쓰고 **한 번도 대사하지 않는다** |
| run 60 전체 실패 (DNS) / run 27 (403 ip-not-allowed) / run 34 (LLM Connection error) | **C (의도 생성 전 중단)** | 지수·국면 조회가 위험관리 패스보다 앞에 있고 try 로 감싸여 있지 않아 사이클 전체가 죽었다 |
| run 64 `running` 채로 영구 방치 (재부팅) | 관측성 | `initialize_db()` 의 orphan 마감은 상태만 바꾸고 미체결 대사는 하지 않는다 |
| `autotrade.log` 의 AAPL 반복 블록 | 잡음 | pytest 가 같은 로그 파일에 쓴다 — 실계좌 사이클이 아니다 |

**F(체결됐는데 UI 미반영) 은 증거가 없다.** 애초에 체결 여부를 기록하는 필드가 없어서 판정할 수 없다 → **G**.

## 2. 코드 결함 (운영 장애로 확정된 것은 아님)

파일:행은 수정 후 기준.

| # | 위치 | 결함 |
|---|---|---|
| A | `autotrade.py` 노출 축소·만기 루프 | `sym in account["open_orders"]` 로 제외 → 강제 매도 의도가 뒤쪽 `cancel_first` 경로에 **도달조차 못 했다** |
| B | `free_position` / `place_order` | 조회 실패(None)를 '계획 수량 그대로' 로 읽었다 → 취소 완료도 매도 가능 수량도 모르는 채 제출 |
| C | `position_entry_map` | 로컬 대체가 `submitted` 를 진입/청산으로 취급, 브로커 진입일이 하나라도 있으면 나머지 보유 종목 누락을 보완하지 않음 |
| D | `find_order` / `toss._call` | 조회 실패와 부재를 둘 다 None 으로, POST 도 타임아웃 시 자동 재전송 |
| E | `run_cycle` | `risk_symbols` 를 **계획만으로** 확정 → 제출 실패한 종목이 다음 패스에서 "이미 주문함" 으로 제외 |
| F | `account_state` | 미체결 조회 실패를 `open_orders=[]` 로 = '미체결 없음' 추정 |

## 3. 매도가 안 될 때의 조치

1. `autotrade.log` 에서 해당 종목의 사유 코드를 찾는다 (`NOT_HELD` / `BELOW_MIN_ORDER` /
   `FRACTIONAL_SESSION` / `OPEN_ORDER_WAIT` / `PENDING_SELL_SUFFICIENT` /
   `DEFERRED_OPEN_UNKNOWN` / `ALREADY_ORDERED` / `ENTRY_UNVERIFIED` / `DEFERRED_SESSION`).
2. 코드가 없으면 **의도 자체가 안 생긴 것**이다 → 전략 스위치를 본다
   (`STOP_LOSS_PCT`, `MOMENTUM_EXIT`, `MAX_HOLD_DAYS`, `equity` 행 수).
   **여기서 스위치를 켜는 것은 전략 변경이라 별도 승인 사항이다.**
3. `orders.status` 가 `ACKNOWLEDGED` 면 **접수만 된 것**이다. 토스 앱/계좌에서 체결을 확인한다.
4. `UNKNOWN` 이면 계좌에서 그 `clientOrderId` 를 직접 찾는다. **재전송하지 않는다.**
5. `DEFERRED` 면 다음 사이클이 다시 대사한다. 반복되면 해당 조회 API(미체결/매도가능)를 점검한다.

## 4. 아직 확인하지 못한 외부 API 계약

- `clientOrderId` 의 멱등성 보장 범위와 보존 기간 (중복 키를 브로커가 정말 거절하는지)
- `GET /orders` 의 페이지네이션·조회 기간·일관성 지연, `CLOSED` 가 전 기간을 포함하는지
- 부분 체결·취소·거절의 정확한 status 값과 잔량 필드 이름
- 소수점 수량 정밀도(현재 6자리 가정)와 세션별 허용 주문 유형

→ 그래서 `order_state()` 의 매핑은 보수적이고(`CLOSED` 인데 체결도 거절도 아니면 UNKNOWN),
`position_entry_map` 은 이력을 믿는 대신 **holdings 수량과 대조**한다.

## 5. 현금 교착 산수 (시나리오 J) — 버그가 아니라 정책 + 계약 미확인

입력: 운영 DB **사본**의 `runs(id=63)`, `trading_decisions(run_id=63)`. 시나리오 파일에는
상수로 박혀 있어 실행 중 운영 DB 를 열지 않는다.

| 항목 | 코드 | 값 (run 63) |
|---|---|---|
| `cash` | `float(toss.buying_power("USD")["cashBuyingPower"])` | $0.01 |
| `total_value` | `cash + Σ(USD 보유 marketValue)` | 재구성 $568.93 / 기록 $568.83 (차 $0.10, 시세 스냅샷 시점차) |
| 유지선 | `total_value × CASH_RESERVE_PCT/100` (=10%) | $56.89 |
| **가용예산** | `cash − 유지선` (`validate_orders`) | **−$56.88** |
| 최소 주문 | `MIN_ORDER_USD` | $5 |

`cash $0.01` 과 `가용 −$2.12` 는 **서로 다른 시점의 값**이다. −$56.88 이 현재 상태이고,
−$2.12(재현값 −$2.13)는 **MU 를 팔아 체결됐다고 가정**했을 때의 값이다:
`(0.01 + 54.76) − 568.93×0.10 = −2.13`. 예수금과 준비금 차감 후 예산을 섞어 쓰면 안 된다.

체결 가정 후 예산 (시나리오 J2):
`SMCI($145.72)→$88.84 ✔ · MSTR($129.51)→$72.63 ✔ · TSLA($123.25)→$66.36 ✔ ·
MRNA($58.16)→$1.28 ✘ · MU($54.76)→−$2.13 ✘ · KDP($31.25)→−$25.63 ✘ · TEAM($26.28)→−$30.61 ✘`

**판정: 버그가 아니다.** `CASH_RESERVE_PCT` 는 총자산 대비 현금 하한이고, 전액 투자
상태에서 신규 매수를 막는 것은 그 정의 그대로의 동작이다. 이 시스템에는 **리밸런싱
기능이 없다** — 목표 비중으로 되돌리는 경로가 설계에 없으므로, 그 부재를 버그라고
부를 수 없다. 다만 다음 둘은 **실제 결함 후보**로 남는다(코드로 확인, 계약은 미확인):

- **J3** `total_value` 는 USD 보유만 센다. 원화 현금·국내주식이 있으면 분모가 작아져
  `CASH_RESERVE_PCT`·`MAX_POSITION_PCT` 가 의도보다 타이트해진다. *이 계좌에 원화 자산이
  있는지는 확인하지 못했다.*
- **J4** `cash` 는 `cashBuyingPower` 를 그대로 쓴다. 이 값이 미체결 매수 금액을 이미
  차감한 값인지 **API 문서로 확인하지 못했다.** 아니라면 예산이 과대계상된다.

## 6. 만기 도달 (시나리오 H) — 날짜는 확정하지 않는다

**진입일 원본을 확인하지 못했다.** 브로커 체결 이력은 런타임에만 조회되고 로컬 DB·로그에
남지 않는다. 로그에 있는 것은 코드가 계산한 `보유 거래일수`뿐이므로 그것을 입력으로 썼다.

- `trading_days_since(ts, cal)` 는 `start < d <= today` 인 달력 원소를 센다 — 진입일 당일은
  제외, 오늘은 포함.
- 달력(`TRADING_DAYS`)은 **런타임에 지수 일봉 인덱스에서** 채워진다. 오프라인에서는 재현할
  수 없으므로 **특정 만기 날짜를 확정하지 않는다.** H7 이 보여주듯 달력이 비면 평일 휴장
  일수만큼 만기가 **일찍** 온다.
- H6 결과 — 만기까지 남은 **거래일 수**(달력 기준):
  `SMCI+13, MSTR+13, KDP+13, MU+15, TSLA+15, TEAM+16, MRNA+19`

**성립 조건 / 실패 가능성**
| 조건 | 깨지면 |
|---|---|
| 그날 사이클이 실제로 돈다 (PC 가 켜져 있고 `schedule` 이 살아 있다) | 만기가 그만큼 미뤄진다 (run 64 재부팅 전례) |
| 브로커 체결 이력으로 진입일이 **검증**된다 | `ENTRY_UNVERIFIED` → 만기 판단 보류 (E6·E7) |
| 지수 일봉이 와서 달력이 채워진다 | 주말만 빼는 근사로 만기가 앞당겨진다 (H7) |
| 정규장 안에서 사이클이 돈다 | 소수점 잔량은 `FRACTIONAL_SESSION` 으로 보류 (H3) |
| `HOLD_EXTEND_TOP` 이 0 이다 | 상위 N위 안이면 매도하지 않는다 |
| 매도가 **체결**된다 | 접수만으로는 현금·슬롯이 생기지 않는다 (H5·G1) |

## 7. 변동성 타겟 자동 활성화 (시나리오 I)

- 경로: `exposure_cap()` → `_vol_target_cap()`. `VOL_TARGET_PCT=30`, `VOL_WINDOW=60`.
  `equity` 행이 `VOL_WINDOW+1 = 61` 개 미만이면 `None` = 기능 비활성 (I1).
- `equity` 행은 `log_equity()` 가 **미국 날짜당 한 줄**을 UPSERT 한다. 하루에 사이클이
  여러 번 돌아도 1행이고, 그날 모든 사이클이 실패하면 0행이다.
  현재 5행(09-09·10·11·14·15) → **56개의 "사이클이 성공한 미국 거래일"이 더 필요하다.**
  하루라도 통째로 실패하면 그만큼 밀린다. 달력 날짜는 확정하지 않는다.
- 수익률: `r = (v1 − cashflow1)/v0 − 1`. `cashflow` 는 **그 날의 입금(+)/출금(−)** 이고
  통화는 USD(=`total_value` 와 같은 축), 시점은 **그 날 행**이다.
- **I3 vs I4 가 핵심이다.** 시장가치를 고정한 채 입금 $200 만 넣으면:
  - `cashflow` 기재 → 변동성 0 유지, 상한 없음 (축소 주문 0건)
  - `cashflow` NULL 방치 → 허위 변동성 → **상한 73.2% 로 강제 축소가 걸린다**
- **I5**: 쿼리가 `COALESCE(cashflow, 0)` 이라 **NULL(미기재)과 0(입출금 없음)이 구분되지
  않는다.** 현재 5행 전부 NULL 이므로 "입출금이 없었다"가 아니라 "모른다"가 맞다.

### 활성화 전 승인 게이트 (제안 — 설정은 바꾸지 않았다)
1. `equity` 행이 55개를 넘으면 경보를 띄운다(현재 그런 장치가 없다).
2. 그 시점에 `cashflow IS NULL` 인 행을 전부 나열하고, 각 날짜에 입출금·환전이 있었는지
   **사람이 확인**한다. 확인된 날은 0 을, 있었던 날은 금액을 기록한다.
   → 운영 DB 직접 수정이 아니라, 감사 가능한 입력 스크립트(입력값·수정 전후를 로그로
   남기고 백업본을 뜨는 형태)로 하는 것을 권한다.
3. 확인이 끝나기 전에는 `VOL_TARGET_PCT=0` 으로 두어 **점등 자체를 막는 것**이 안전하다.
   (이것도 설정 변경이므로 승인 사항이다.)

## 8. 공식 스펙으로 확정된 계약 (2026-09-16 추가)

출처: `https://openapi.tossinvest.com/openapi-docs/latest/openapi.json` (**인증 없는 공개
문서**, `GET` 만. 계좌·주문 API 는 호출하지 않았다). 토스증권 Open API **1.2.17**.
§4 에 "미확인"으로 적었던 항목이 대부분 확정됐고, **새 확정 결함 2건**이 나왔다.

| 항목 | 스펙이 말하는 것 | 코드에 미친 영향 |
|---|---|---|
| `clientOrderId` | 멱등성 키. 전달 시 동일 값 재요청은 이전 주문 결과를 재반환. **유효 10분**, 이후 동일 값은 **새 주문**. 최대 36자, `^[a-zA-Z0-9\-_]+$` | `intent_key` 는 **30분 버킷** — 10~30분 사이 재제출은 브로커 멱등성 밖이다. 키를 바꾸지 않고, 중복 방어를 **미체결 잔량 대사**에 둔다 (L6·F4). 키 길이·패턴은 적합 |
| `GET /orders` 페이징 | `status=OPEN` 전량. **`status=CLOSED` 는 `limit` 기본 20 / 최대 100 + `cursor`** (`nextCursor`·`hasNext`) | **확정 결함 ①**: `toss.orders("CLOSED")` 가 최근 20건만 가져왔다 → 진입일 복원·대사가 20건 창에 갇혀 있었다. `orders_all()` 추가 (L1–L4) |
| 조회 범위 | "Open API 가 지원하는 호가 유형(지정가·시장가·장마감지정가)으로 접수된 주문만 반환. 장후·장전 시간외 종가 주문은 **목록과 상세 조회 모두에서 조회되지 않는다**" | **확정 결함 ②(설계 한계)**: 사용자가 앱에서 시간외로 산 종목은 진입일을 영영 복원할 수 없다. → `ENTRY_UNVERIFIED` 로 만기 판단에서 제외하는 것이 유일하게 맞는 처리 |
| `orders[].status` | `PENDING · PENDING_CANCEL · PENDING_REPLACE · PARTIAL_FILLED · FILLED · CANCELED · REJECTED · CANCEL_REJECTED · REPLACE_REJECTED · REPLACED`. 쿼리의 `OPEN/CLOSED` 와 **값 체계가 다르다**. `PARTIAL_FILLED` 는 양쪽 그룹에 나온다. "클라이언트는 unknown code 를 허용할 것" | `order_state()` 를 **실제 enum 으로 교체**. 모르는 코드는 UNKNOWN (C3) |
| 잔량 필드 | **없다.** `quantity` − `execution.filledQuantity` 로 계산 | `remaining_qty()` 로 명시 (C5) |
| 취소·정정 거부 | `CANCEL_REJECTED`/`REPLACE_REJECTED` 는 **별도 주문 레코드**로 생기고 원주문은 이전 상태로 복귀 | 원주문의 결말로 쓰지 않는다 |
| 소수점 수량 | US `MARKET`+`SELL` 만. **소수점 6자리**까지(초과 시 `fractional-quantity-scale-exceeded`). 정규장 시작~**종료 1시간 전**만 접수, 밖이면 `422 fractional-quantity-outside-regular-hours` | 기존 `f"{q:.6f}"` 와 `fractional_allowed()` 가 **정확히 일치**한다 (L5). 2026-09-08 WDAY 400 의 원인도 이것 |
| `price` 정밀도 | US: $1 이상 소수 2자리, $1 미만 4자리, 초과분 **절삭** | 현재 `round(x, 2)` — $1 미만 종목에서 정밀도 손실은 있으나 거절되진 않는다 |
| `timeInForce` | 미전달 시 `DAY` — **정규장 종료까지 미체결분은 자동 취소** | 코드는 미전달(=DAY). 미체결 매도는 다음 날로 넘어가지 않는다 |
| `cashBuyingPower` | "현금 기반 매수 가능 금액(미수 미발생 기준)" | **여전히 미확인**: 미체결 매수 금액을 이미 차감한 값인지 문서에 없다 (J4 유지) |

## 9. 접수 → 체결 대사 루프 (시나리오 M)

`reconcile_orders(toss)` 를 `run_cycle` 이 계좌 조회 직후 부른다. 이것이 **ACK 와 FILL 을
잇는 유일한 경로**다 — 없으면 DB 는 영원히 ACKNOWLEDGED 에 멈춘다 (TEAM 2026-09-09).

- 대상: `client_order_id` 가 있고 상태가 `INTENT_RECORDED / ACKNOWLEDGED / PARTIALLY_FILLED
  / CANCEL_PENDING / UNKNOWN` 인 행 (최근 100건).
- 비용: 사이클당 조회 2종(OPEN 전량 + CLOSED 페이징). 목록에서 못 찾고 brokerId 가 있을
  때만 `GET /orders/{orderId}` 폴백 (M7).
- **조회가 실패하면 아무 상태도 바꾸지 않는다** (M4). 목록을 다 읽었는데도 없으면
  `CANCELED` 가 아니라 `UNKNOWN: 계좌 이력에서 찾지 못함` 이다 (M5).
- 레거시 `submitted` 행은 `client_order_id` 가 없어 **대사 대상이 아니고 변환하지도 않는다**
  (M6). 대신 건수를 경고로 남긴다.

### DB 스키마 변경 (추가만)
`orders` 에 `client_order_id TEXT` · `filled_quantity REAL` · `reconciled_at TEXT` 를
`initialize_db()` 의 기존 `ALTER TABLE` 패턴으로 추가한다. 운영 DB **사본**으로 리허설:

```
행수 before/after: runs 65 / orders 51 / trading_decisions 608 / equity 5  — 동일
기존 orders 51행 (상태·order_id 포함) 동일: True
신규 컬럼 전부 NULL: True        재실행 멱등: True
```

롤백: 코드만 되돌리면 된다. 추가 컬럼은 남지만 이전 코드는 컬럼을 명시 지정해 읽으므로
무시된다. **레거시 `submitted` 를 `filled` 로 바꾸는 변환은 하지 않았다.**
