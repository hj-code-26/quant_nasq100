# Claude 자동매매 (토스증권 · 나스닥 100)

[youtube-jocoding/gpt-bitcoin](https://github.com/youtube-jocoding/gpt-bitcoin) 의 흐름(데이터 수집 → LLM 판단 → 주문 → 기록)을
**GPT → Claude**, **업비트 → 토스증권 Open API** 로 바꾸고, 단일 종목 대신 **나스닥 100 에서 종목을 골라 포트폴리오로** 굴리도록 확장한 것.

```
autotrade.py               매매 봇 (3단계 깔때기, 개장 후 하루 5회)
instructions_screen.md     1단계 프롬프트 — 후보 선별
instructions.md            2단계 프롬프트 — 종목별 판단
instructions_portfolio.md  3단계 프롬프트 — 포트폴리오 배분
nasdaq100.py               스크리닝 유니버스
streamlit_app.py           기록 대시보드 (Streamlit)
autotrade_server.py        웹 대시보드 (FastAPI + frontend_autotrade/)
toss.py                    토스증권 REST 클라이언트
```

## 흐름 (한 사이클)

```
휴장 확인 → 계좌 상태(현금·보유·미체결)
 1) 스크리닝   나스닥 100 전 종목 일봉 지표 + 규칙 점수(추세·모멘텀·MACD·비과열·거래량)
              → 상위 SCREEN_N 개 표를 Claude 에 → 후보 TOP_N 개          [Claude 1회]
 2) 종목별 판단 후보 + 보유 종목 각각: 일봉 30 + 시간봉 24 + 지표 + 호가 + 뉴스 + 지난 판단
              → {decision, percentage, reason}                            [Claude 종목당 1회, 병렬]
 3) 배분       종목별 판단 + 계좌 + 규칙 → 최종 주문 목록 + 요약            [Claude 1회]
              → 코드 검증: 최대 종목 수 · 종목당 비중 · 현금 유지 · 최소 주문 · 미체결 · 정규장 외 정수 주
              → 매도 먼저, 매수 나중. clientOrderId 로 중복 주문 방지
 기록          runs / trading_decisions / orders (SQLite) + autotrade.log
```

2단계 종목별 판단은 그 종목만 보고(현금 사정은 일부러 안 보여줌), 3단계가 계좌 전체(현금·보유·분산)를 보고 최종 결정한다.
Claude 가 낸 주문은 반드시 코드의 규칙 검증을 거친다. 규칙은 `.env` 에서 조정.

## 신호와 사이즈는 백테스트로 정했다 (`backtest.py`)

나스닥 101종목 × 3년 일봉(토스), 시간순 앞 2/3 탐색 · 뒤 1/3 검증. 매일 규칙대로 상위 5종목을 뽑아 20일 보유했을 때:

| 선별 규칙 | 유니버스 대비 (탐색 / 검증) | 양(+)인 날 |
|---|---|---|
| **20일 수익률 상위** (적용) | +2.7%p / +4.7%p | 67% / 65% |
| 20일 모멘텀 + 구간 가중 사이즈 (적용, 일부 현금 보유) | +2.1%p / +4.1%p | 67% / 64% |
| 이평 정배열·MACD·RSI·볼린저 교과서 규칙 점수 (구버전) | +0.3%p / +1.1%p | 61% / 61% |
| 모멘텀 + 5일 눌림 우선 | +0.3%p / +0.0%p | 60% / 57% |
| 20일 수익률 하위 (역발상) | +0.3%p / −1.4%p | 60% / 49% |

- RSI 70↑, 볼린저 상단 이탈 같은 "과열" 조건은 이후 수익률을 낮추지 않았다. 과열 제외는 성적을 깎았다.
- 그래서 **선별 = 20일 수익률 순**, **매수 금액 = 코드가 모멘텀 구간으로 결정**(>20%: 한도의 100%, 10~20%: 70%, 0~10%: 40%, 음수: 매수 안 함), **Claude = 뉴스·이벤트·분산에 대한 거부권**.
- t값은 1 안팎이라 "확실한 예측"이 아니라 "일관된 경향"이다. 강세장 3년 표본이므로 국면이 바뀌면 재검증할 것: `python backtest.py --strategy`.
- 판단 성적표: `python evaluate.py` (기록된 판단을 이후 가격과 대조).

## 실행

처음이면 `make install` 후 `cp .env.example .env` 로 키를 넣는다. 그 다음은 `make` 한 줄이면 된다.

```bash
make            # 웹 대시보드 실행 + 브라우저 열기 → http://127.0.0.1:8877
make help       # 전체 명령 보기
```

| 명령 | 하는 일 |
|---|---|
| `make` | 웹 대시보드 (FastAPI). 이미 떠 있으면 브라우저만 연다 |
| `make dash` | Streamlit 대시보드 → http://127.0.0.1:8501 |
| `make once` | 1회 모의 실행 (`DRY_RUN=1` 강제, 주문 안 나감) |
| `make live` | 1회 실주문 — 확인 문구 「실주문」 을 입력해야 진행 |
| `make status` / `make stop` / `make logs` | 상태 · 종료 · 로그 |
| `make check` | 토스 연결 확인 (읽기 전용) |
| `make eval` / `make backtest` | 판단 성적표 · 선별 규칙 백테스트 |

직접 실행할 때는:

```bash
python toss.py               # 토스 연결 확인 (읽기 전용) → accountSeq 를 .env 에
python autotrade.py --once   # 1회 실행 (DRY_RUN=1 이면 주문 없음)
python autotrade.py          # 즉시 1회 + TRADE_TIMES 마다 반복
```

`DRY_RUN=1`(기본) 이면 스크리닝·판단·배분·검증까지 다 하고 주문만 내지 않는다. 실주문은 `DRY_RUN=0`.

### 대시보드에서 실행하기

`streamlit run streamlit_app.py` 로 띄운 대시보드에서 봇을 직접 돌릴 수 있다. 봇은 대시보드 프로세스 안의 백그라운드 스레드로 돈다.

- **지금 1회 실행**: 한 사이클 즉시 실행. 로그가 3초마다 갱신된다.
- **자동 실행 시작/중지**: `TRADE_TIMES` 마다 반복. 대시보드를 끄면 같이 멈춘다.
- **실주문 토글**: 대시보드는 `.env` 와 무관하게 **항상 모의로 시작**한다. 토글을 켜고 확인 문구 「실주문」을 입력해야 실제 주문이 나간다.

터미널의 `python autotrade.py` 와 대시보드 자동 실행을 **동시에 켜지 말 것** — 같은 시각에 두 번 주문한다.

### OmniRoute 게이트웨이로 호출하기 (선택)

Anthropic 직접 호출 대신 [OmniRoute](https://github.com/diegosouzapw/OmniRoute) 를 거칠 수 있다.

```bash
npm install -g omniroute
omniroute serve                # http://localhost:20128 대시보드
```

1. 대시보드 > Providers 에서 Claude(구독 OAuth) 또는 Anthropic(API 키) 공급자 추가
2. `.env` 에 `ANTHROPIC_BASE_URL=http://localhost:20128`, `CLAUDE_MODEL=claude/claude-sonnet-5` (모델 ID 는 `curl localhost:20128/v1/models` 로 확인)

게이트웨이 경유 시 봇은 베타 파라미터(대체 모델·JSON 스키마 강제) 없이 호출하고, 응답 본문에서 JSON 을 꺼낸다.

## 원본과 다른 점

| 항목 | gpt-bitcoin | 이 저장소 |
|---|---|---|
| 모델 | gpt-4-turbo `json_object` | `claude-sonnet-5` structured output (JSON 스키마 강제) |
| 거래소 | 업비트 KRW-BTC | 토스증권 미국 주식 |
| 종목 | 1개 고정 | 나스닥 100 스크리닝 → 후보 → 포트폴리오 (최대 `MAX_POSITIONS`) |
| 시간봉 | 업비트 1시간봉 | 토스에 1시간봉이 없어 1분봉(약 1.3일치만 제공)을 1시간으로 묶음. MACD 등 긴 지표는 시간봉에서 빠질 수 있음 |
| 매수 | KRW × % 시장가 | USD 금액 기반 시장가 (소수점 주식, 정규장 종료 1시간 전까지). 그 외 시간은 정수 주 |
| 리스크 규칙 | 없음 | 최대 종목 수 · 종목당 비중 · 현금 유지 · 최소 주문 · 미체결 종목 제외 |
| 공포·탐욕 지수 | alternative.me (암호화폐) | 주식용 공개 API 가 없어 제외 |
| 실행 시각 | 00:01 / 08:01 / 16:01 | 개장 22:30 → 23:30 → 00:00 부터 2시간 간격 (`TRADE_TIMES`, KST). 장 시작 전·마감 후·휴장일은 건너뜀 |
| 휴장 | — | 토스 장운영 캘린더로 휴장일 건너뜀 |

⚠️ 실계좌 주문이 나간다. 충분히 `DRY_RUN=1` 로 돌려본 뒤 사용할 것. 투자 손실은 본인 책임.
