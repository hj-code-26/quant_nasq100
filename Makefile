# GAF-VAE Quant — 실행/종료 정리
# 사용법: make          (도움말)
#         make run      (서버 기동)
#         make stop     (서버 종료)
SHELL := /usr/bin/bash
.SHELLFLAGS := -c
PORT := 8899
PY   := python
URL  := http://127.0.0.1:$(PORT)
LOG  := server.log
TICKER ?= AAPL

.DEFAULT_GOAL := help
.PHONY: help install run start stop restart status logs open \
        account plan journal scan stats clean clean-cache

help:  ## 사용 가능한 명령 보기
	@echo "GAF-VAE Quant"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  ex) make plan TICKER=MSFT"

install:  ## 의존성 설치
	$(PY) -m pip install -r requirements.txt

# ---------- 서버 ----------
run: stop  ## 서버 기동 (포그라운드, Ctrl+C 로 종료)
	$(PY) run.py

start:  ## 서버 기동 (백그라운드, 로그는 server.log)
	@$(MAKE) --no-print-directory stop
	@nohup $(PY) run.py > $(LOG) 2>&1 & \
	  for i in $$(seq 1 30); do \
	    curl -sf -o /dev/null $(URL)/api/scan/status && \
	      { echo "started -> $(URL)"; exit 0; }; \
	    sleep 1; \
	  done; echo "start failed - see $(LOG)"; exit 1

stop:  ## 서버 종료 (포트 8899 점유 프로세스)
	@pids=$$(netstat -ano 2>/dev/null | grep ":$(PORT) .*LISTENING" | awk '{print $$NF}' | sort -u); \
	if [ -z "$$pids" ]; then echo "not running"; else \
	  for p in $$pids; do taskkill //F //PID $$p >/dev/null 2>&1 && echo "stopped PID $$p"; done; \
	fi

restart:  ## 서버 재시작 (백그라운드)
	@$(MAKE) --no-print-directory start

status:  ## 서버 상태 확인
	@curl -sf -o /dev/null $(URL)/api/scan/status && echo "running  -> $(URL)" || echo "stopped"
	@netstat -ano 2>/dev/null | grep ":$(PORT) .*LISTENING" || true

logs:  ## 서버 로그 따라보기
	@tail -f $(LOG)

open: start  ## 서버 기동 후 브라우저 열기
	@start $(URL) 2>/dev/null || echo "open $(URL) in your browser"

# ---------- 토스 / 매매 (전부 조회 · 모의) ----------
account:  ## 토스 계좌 연결 확인 (읽기 전용)
	$(PY) -m backend.toss

plan:  ## 매매 계획 산출, 주문 없음 (make plan TICKER=MSFT)
	$(PY) -m backend.trader $(TICKER)

journal:  ## 거래 일지 + 현재가 대조
	$(PY) -m backend.trader --report

# ---------- 분석 ----------
scan:  ## 250종목 전체 스캔 (약 20분)
	$(PY) -c "from backend import nasdaq100 as n; n._run(n.TICKERS, 8)"

stats:  ## 스캔 결과 통계적 유의성 검정
	@$(PY) -c "import json,sys; sys.stdout.reconfigure(encoding='utf-8'); \
	from backend import nasdaq100 as n; print(json.dumps(n.scan_stats(), ensure_ascii=False, indent=2))"

# ---------- 정리 ----------
clean-cache:  ## 학습 모델 캐시 삭제 (스캔 결과는 유지)
	@rm -f models_cache/*.pkl models_cache/*.pt 2>/dev/null; echo "model cache cleared"

clean: stop  ## 서버 종료 + 임시파일 정리 (.env·일지·스캔결과는 보존)
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; \
	 rm -f $(LOG) 2>/dev/null; echo "cleaned"
