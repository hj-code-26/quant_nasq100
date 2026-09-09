# 인수인계 프롬프트 — 나스닥 100 Claude 자동매매 → 비트코인/암호화폐 이식

이 폴더는 새 프로젝트다. 아래는 원본 프로젝트 `C:\Users\ryan0\quant_nasq100` 의 내용을 그대로 옮긴 것이다.
원본 파일이 필요하면 그 경로에서 직접 읽어라 (읽기만, 수정 금지). 새 코드는 이 폴더에만 만든다.

---

## 1. 원본 프로젝트가 무엇인가

[youtube-jocoding/gpt-bitcoin](https://github.com/youtube-jocoding/gpt-bitcoin) 의 흐름(데이터 수집 → LLM 판단 → 주문 → 기록)을
**GPT → Claude**, **업비트 → 토스증권 Open API** 로 바꾸고, 단일 종목 대신 **나스닥 100 에서 종목을 골라 포트폴리오로** 굴리는 봇.
이번 목표는 이것을 다시 **암호화폐**로 되돌리되, 원본이 백테스트로 찾아낸 신호와 안전장치는 유지하는 것.

```
autotrade.py               매매 봇 (3단계 깔때기, 개장 후 하루 5회)  691줄
instructions_screen.md     1단계 프롬프트 — 후보 선별
instructions.md            2단계 프롬프트 — 종목별 판단
instructions_portfolio.md  3단계 프롬프트 — 포트폴리오 배분
nasdaq100.py               유니버스 티커 리스트 (TICKERS)
backtest.py                지표 조건 → 이후 5/20일 수익률 백테스트  257줄
evaluate.py                기록된 판단을 이후 가격과 대조한 성적표
toss.py                    토스증권 REST 클라이언트 (candles/prices/orderbook/holdings/buy/sell/orders)
streamlit_app.py           기록 대시보드 (Streamlit)
autotrade_server.py        웹 대시보드 (FastAPI + frontend_autotrade/)
Makefile                   make once(모의) / make live(실주문, '실주문' 문구 확인) / make backtest 등
requirements.txt           anthropic requests pandas pandas_ta numpy schedule streamlit fastapi uvicorn pydantic
```

## 2. 한 사이클 흐름 (run_cycle)

```
휴장 확인 → 계좌 상태(현금·보유·미체결)
 1) 스크리닝   유니버스 전 종목 일봉 90개 + 지표 → 20일 수익률 내림차순 표
              → 상위 SCREEN_N(40) 개를 Claude 에 → 후보 TOP_N(5) 개          [Claude 1회]
 2) 종목별 판단 후보 + 보유 종목 각각: 일봉 30 + 시간봉 24 + 지표 + 호가 + 뉴스 + 지난 판단
              → {decision: buy|sell|hold, percentage: 0~100, reason}       [Claude 종목당 1회, 병렬 WORKERS=3]
 3) 배분       종목별 판단 + 계좌 + 규칙 → 최종 주문 목록 + 요약            [Claude 1회]
              → 코드 검증: 최대 종목 수 · 종목당 비중 · 현금 유지 · 최소 주문 · 미체결 · 정규장 외 정수 주
              → 매도 먼저, 매수 나중. clientOrderId 로 중복 주문 방지
 기록          runs / trading_decisions / orders (SQLite trading_decisions.db) + autotrade.log + 토큰 비용
```

설계 원칙:
- 2단계는 그 종목만 본다. **현금 사정은 일부러 안 보여준다** (보여주면 hold 편향 생김). 3단계가 계좌 전체를 본다.
- **Claude 는 거부권만 갖는다.** 종목 순서와 매수 금액은 코드가 정한다. Claude 가 낸 주문은 반드시 코드 규칙 검증을 거친다.
- 모든 Claude 호출은 JSON 스키마 강제 (structured output). 게이트웨이(OmniRoute) 경유 시엔 스키마 없이 호출하고 본문에서 JSON 을 꺼낸다.
- DRY_RUN=1 이 기본. 실주문은 DRY_RUN=0 + 확인 문구.

## 3. 검증된 알고리즘 (이게 핵심이다)

나스닥 101종목 × 3년 일봉, 시간순 앞 2/3 탐색 · 뒤 1/3 검증. 매일 상위 5종목 → 20일 보유:

| 선별 규칙 | 유니버스 대비 초과수익 (탐색 / 검증) | 양(+)인 날 |
|---|---|---|
| **20일 수익률 상위** (적용) | +2.7%p / +4.7%p | 67% / 65% |
| 20일 수익률 > 20% 만 | +3.7%p / +6.0%p | 61% / 66% |
| 20일 모멘텀 + 구간 가중 사이즈 (적용) | +2.1%p / +4.1%p | 67% / 64% |
| 이평 정배열·MACD·RSI·볼린저 교과서 규칙 점수 (구버전) | +0.3%p / +1.1%p | 61% / 61% |
| 모멘텀 + 5일 눌림 우선 | +0.3%p / +0.0%p | 60% / 57% |
| 20일 수익률 하위 (역발상) | +0.3%p / −1.4%p | 60% / 49% |

결론:
- **신호 = 횡단면 20일 수익률 순위.** 여러 종목을 20일 수익률로 줄 세워 상위를 산다.
- RSI 70↑, 볼린저 상단 이탈, 스토캐스틱 80↑ 같은 "과열" 은 이후 수익률을 **낮추지 않았다.** 과열 제외는 성적을 깎았다. RSI 80↑, 20일 +20%↑ 구간이 가장 많이 올랐다.
- 20일 −5 ~ +5% 의 어중간한 종목이 가장 못 올랐다.
- 이평 정배열·MACD·RSI 같은 교과서 지표는 기준선과 차이 없음. 지표는 Claude 에 **참고 열로만** 준다.
- t값은 1 안팎 → "확실한 예측"이 아니라 "일관된 경향". 강세장 3년 표본이므로 국면 바뀌면 재검증.

매수 금액 = 코드가 정함 (`planned_buy_amount`):
```python
MOMENTUM_TIERS = [(20, 1.0, "강(>20%)"), (10, 0.7, "중(10~20%)"), (0, 0.4, "약(0~10%)")]  # 음수 → 0.0 매수 안 함
cap = 총자산 × MAX_POSITION_PCT/100 − 그 종목 기존 평가액
planned_buy = max(0, cap × size_factor)
```

지표 (pandas_ta, `add_indicators`): SMA10/20/50, EMA10, RSI14, 스토캐스틱(14,3,3), MACD(12,26,9), 볼린저(20,2).
스크리닝 표 열: symbol, close, ret_20d_pct, momentum_tier, size_factor, ret_5d_pct, ret_60d_pct, vs_sma20_pct, rsi14, bb_pct, vol_ratio_20d, atr_pct.

## 4. 리스크 규칙 (.env, validate_orders 가 강제)

```
SCREEN_N=40  TOP_N=5  MAX_POSITIONS=5  MAX_POSITION_PCT=30  CASH_RESERVE_PCT=10  MIN_ORDER_USD=5  WORKERS=3
TRADE_TIMES=22:30,23:30,00:00,02:00,04:00   (KST, 미국 정규장 기준)
DRY_RUN=1   CLAUDE_MODEL=claude-sonnet-5   ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL(선택, OmniRoute)   SERPAPI_API_KEY(선택, 뉴스)
```
- 매도 먼저, 매수 나중. 미체결 주문 있는 종목은 건드리지 않음.
- 종목당 비중 초과분은 깎고, 현금 유지선 밑으로 내려가면 매수 버림. 최소 주문 미만 버림.
- 매도는 sell_pct(보유 수량 중 %) 로만. 매수는 코드의 planned_buy 금액을 Claude 가 승인/거부만.

## 5. 세 프롬프트의 요지 (원본 파일을 그대로 복사해 "미국 주식"→"암호화폐" 로 바꿔 써라)

- **instructions_screen.md (선별)**: 표 순서(20일 수익률)가 기본 답. Claude 는 (1) 이벤트로 만들어진 일회성 급등, 거래량 급감, atr_pct 극단 → 거부, (2) 같은 업종/테마 쏠림 → 분산, (3) 보유 종목은 자동 포함이니 다시 뽑지 않음, (4) 상위권을 뺐으면 reason 에 이유 필수. 백테스트 표를 프롬프트 안에 넣어 "과열은 거부 사유 아님" 을 못 박는다.
- **instructions.md (종목 판단)**: 모멘텀 구간이 출발점 (강·중 → buy/hold, 약·음 → hold/sell). 뉴스로만 거부권. 시간봉은 타이밍 참고만, 시간봉 과열로 일봉 모멘텀을 부정하지 말 것. 지난 판단 기록을 보고 새 정보 없이 뒤집지 말 것. 보유 종목은 20일 수익률 양수면 hold, 음수로 꺾이면 sell, "많이 올랐다"는 매도 이유 아님. percentage 는 확신의 기록이지 금액이 아님.
- **instructions_portfolio.md (배분)**: 매도 먼저. buy 판단이면서 planned_buy > 0 인 종목을 ret_20d_pct 높은 순으로 자리·현금이 허락하는 만큼 승인. 거부 사유는 정성 요인(이벤트·규제·시장 급락)뿐. 승인할 게 없으면 빈 목록.

## 6. 암호화폐로 옮길 때 바뀌는 것

**결정 필요 (먼저 물어볼 것):**
- (A) **BTC 단일**: 횡단면 순위가 성립 안 함 → 시계열 모멘텀(20일 수익률 > 0 이면 매수, 구간 사이즈)으로 바꿔야 하고 **검증 안 된 다른 신호**다. 스크리닝·배분 단계 삭제, 종목 판단 1단계만 남음.
- (B) **코인 유니버스** (업비트 KRW 마켓 상위 30~50 등): 원본 알고리즘과 가장 가깝다. TICKERS 만 코인 목록으로 바꾸면 파이프라인 대부분 유지. **권장.**

**어느 쪽이든:**
- `toss.py` → 거래소 클라이언트로 교체 (pyupbit 또는 ccxt). 나머지 코드는 `candles`, `account_state`, `get_current_status`(prices/orderbook), `place_order`, `market_session` 지점으로만 토스를 부른다.
- 24시간 시장: `market_session`, 휴장일 캘린더, 정규장 외 정수 주 로직 삭제. 소수점 주문 항상 가능. TRADE_TIMES 는 원본 gpt-bitcoin 처럼 00:01/08:01/16:01 등 고정 간격.
- 시간봉: 토스는 1시간봉이 없어 1분봉을 리샘플했지만 거래소는 1시간봉을 직접 준다 → 단순화.
- 통화 USD → KRW (또는 USDT). MIN_ORDER 는 업비트 5,000원. 수수료 0.05%.
- 모멘텀 구간 임계값(20%/10%/0%)은 주식 변동성 기준 → 코인은 재조정. **backtest.py 를 코인 일봉으로 먼저 돌리고 정할 것.** 3년 이상 일봉으로 탐색/검증 분할 유지.
- 뉴스: SerpAPI 코인 헤드라인, 또는 원본처럼 alternative.me 공포·탐욕 지수 추가.
- SQLite 기록, DRY_RUN 기본, 확인 문구 후 실주문, 대시보드는 그대로 가져와도 됨.

## 7. 작업 순서 제안

1. 거래소 클라이언트 + `candles` 로 코인 일봉 수집 → 캐시
2. `backtest.py` 이식 → 20일 모멘텀 횡단면이 코인에서도 유지되는지, 구간 임계값 확인
3. 결과 보고 `MOMENTUM_TIERS` 와 규칙 확정
4. `autotrade.py` 이식 (세션 로직 제거, 통화 변경) + 프롬프트 3개 수정
5. DRY_RUN 으로 여러 사이클 → 실주문

주의: 실계좌 주문이 나가는 코드다. 검증 없이 임계값을 가져다 쓰지 말고, 코인 변동성(주식의 3~5배)에 맞춰 낙폭을 먼저 확인할 것.
