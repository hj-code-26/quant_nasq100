# 매도 실행 경로 감사 (2026-09-16)

기준 커밋: `6a0d843`. 재현: `python research/sell_execution_audit/test_sell_path.py`

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
