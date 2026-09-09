# Claude 자동매매 — 한 줄 실행
#
#   make            웹 대시보드 실행 (http://127.0.0.1:8877) + 브라우저 열기
#   make help       전체 명령 보기
#
# macOS(lsof/open) · Windows Git Bash(netstat/taskkill) 양쪽에서 동작한다.

SHELL := /bin/bash
.SHELLFLAGS := -c
PY    ?= python3
PORT  ?= 8877
DPORT ?= 8501
URL   := http://127.0.0.1:$(PORT)
LOG   := autotrade.log

# 포트를 LISTEN 중인 PID — lsof(mac) 와 netstat(Windows) 결과를 합쳐서 중복 제거
PIDS = { lsof -ti tcp:$(PORT) -sTCP:LISTEN 2>/dev/null; netstat -ano 2>/dev/null | grep -iE 'listen' | grep -E '[:.]$(PORT)[[:space:]]' | awk '{print $$NF}'; } | grep -E '^[0-9]+$$' | sort -u

# 포트 점유 프로세스 정리 (mac: kill, Windows: taskkill — 서로 상대 PID 를 못 죽이므로 둘 다 시도)
KILL_PORT = pids=$$($(PIDS)); if [ -z "$$pids" ]; then echo "  포트 $(PORT) 비어 있음"; else for p in $$pids; do if taskkill //F //PID $$p >/dev/null 2>&1 || kill -9 $$p 2>/dev/null; then echo "  포트 $(PORT) 점유 프로세스 종료: PID $$p"; else echo "  PID $$p 종료 실패 (권한?)"; fi; done; sleep 1; fi

# 브라우저 열기 (mac / Windows / Linux)
BROWSE = { open $(URL) || cmd //c start "" "$(URL)" || xdg-open $(URL); } >/dev/null 2>&1 || true

.DEFAULT_GOAL := run
.PHONY: run help install dash once live stop restart status logs check eval backtest backtest-daily test clean

# ---------- 실행 ----------
run:  ## 대시보드 실행 — 포트 정리 후 시작 + 브라우저 열기 (기본, Ctrl+C 로 종료)
	@$(KILL_PORT)
	@echo "대시보드 시작 -> $(URL)   (Ctrl+C 로 종료)"
	@( sleep 2; $(BROWSE) ) &
	@PORT=$(PORT) $(PY) autotrade_server.py

dash:  ## Streamlit 대시보드 실행 (http://127.0.0.1:8501)
	@echo "Streamlit 시작 -> http://127.0.0.1:$(DPORT)   (Ctrl+C 로 종료)"
	@$(PY) -m streamlit run streamlit_app.py --server.port $(DPORT)

once:  ## 1회 모의 실행 (DRY_RUN 강제, 주문 안 나감)
	@echo "모의 1회 실행 — 주문은 나가지 않습니다."
	@DRY_RUN=1 $(PY) autotrade.py --once

live:  ## 1회 실주문 (확인 문구 '실주문' 입력 필요)
	@echo "⚠️  실제 주문이 나갑니다. 계속하려면 '실주문' 을 입력하세요 (그 외에는 취소)."
	@read -r a; \
	if [ "$$a" = "실주문" ]; then \
	  echo "실주문 모드로 1회 실행합니다."; DRY_RUN=0 $(PY) autotrade.py --once; \
	else echo "취소됨."; fi

# ---------- 서버 관리 ----------
stop:  ## 대시보드 종료 (해당 포트 점유 프로세스)
	@$(KILL_PORT)

restart:  ## 대시보드 재시작 (run 이 이미 포트를 정리한다)
	@$(MAKE) --no-print-directory run

status:  ## 실행 상태 + 봇 상태 확인
	@pids=$$($(PIDS)); \
	if [ -z "$$pids" ]; then echo "대시보드: 꺼짐"; else \
	  echo "대시보드: 켜짐 -> $(URL)  (PID $$pids)"; \
	  curl -sf --max-time 5 $(URL)/api/state \
	    | PYTHONIOENCODING=utf-8 $(PY) -c "import json,sys; d=json.load(sys.stdin); c=d['config']; print('  \ubaa8\ub4dc    : ' + ('\ubaa8\uc758(DRY_RUN)' if d['dry_run'] else '\uc2e4\uc8fc\ubb38')); print('  \uc790\ub3d9\uc2e4\ud589: ' + ('\ucf1c\uc9d0' if d['auto'] else '\uaebc\uc9d0') + '   \ub2e4\uc74c ' + str(d['next_run'] or '\u2014')); print('  \ub9c8\uc9c0\ub9c9  : ' + str(d['last_run'] or '\u2014')); print('  \ubaa8\ub378    : ' + c['model'])" 2>/dev/null || echo "  (상태 조회 실패)"; \
	fi

logs:  ## 로그 따라보기 (Ctrl+C 로 중단)
	@tail -f $(LOG)

# ---------- 점검 · 분석 (조회 전용, 주문 없음) ----------
check:  ## 토스 연결 확인 (읽기 전용)
	@$(PY) toss.py

eval:  ## 지난 판단 성적표 (기록 vs 이후 가격)
	@$(PY) evaluate.py

backtest:  ## 선별 규칙 백테스트 (data_cache 사용, 네트워크 불필요)
	@$(PY) backtest.py --strategy

backtest-daily:  ## 계좌 단위 일별 시뮬 — 청산·진입 규칙 조합 비교 (data_cache 사용)
	@$(PY) backtest.py --daily

test:  ## 자체 점검 (지표 · 주문 검증 규칙, 네트워크 불필요)
	@$(PY) test_indicators.py && $(PY) test_validate.py

# ---------- 설치 · 정리 ----------
install:  ## 의존성 설치
	@$(PY) -m pip install -r requirements.txt

clean:  ## 파이썬 임시파일 정리 (.env · DB · 캐시는 보존)
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; \
	 find . -name '*.pyc' -delete 2>/dev/null; echo "정리 완료"

help:  ## 사용 가능한 명령 보기
	@echo "Claude 자동매매   [$(PY)]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  그냥 'make' 만 치면 웹 대시보드가 뜬다."
