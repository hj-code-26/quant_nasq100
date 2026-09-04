"""토스증권 Open API 클라이언트 (REST).

준비: 토스증권 WTS > 설정 > Open API 에서 client_id/secret 발급 + 허용 IP 등록.
환경변수: TOSS_CLIENT_ID, TOSS_CLIENT_SECRET, TOSS_ACCOUNT_SEQ(주문·자산용, GET /accounts 의 accountSeq)

사용:
    from toss import TossClient
    t = TossClient()
    t.prices("AAPL")                       # 현재가
    t.candles("AAPL", interval="1d")       # 일봉 200개
    t.holdings()                           # 보유 주식
    t.buy("AAPL", quantity="1", price="200.00")   # 지정가 매수
    t.sell("AAPL", quantity="1")                  # 시장가 매도
"""
import json
import os
import pathlib
import sys
import time
import uuid

import requests

BASE = "https://openapi.tossinvest.com"


def _load_dotenv(path=pathlib.Path(__file__).resolve().parent / ".env"):
    """의존성 없이 .env 를 os.environ 에 주입. 이미 있는 값은 덮어쓰지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv()


class TossError(RuntimeError):
    def __init__(self, status, code, message, data=None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.data = status, code, data


class TossClient:
    def __init__(self, client_id=None, client_secret=None, account_seq=None):
        self.client_id = client_id or os.environ["TOSS_CLIENT_ID"]
        self.client_secret = client_secret or os.environ["TOSS_CLIENT_SECRET"]
        self.account_seq = account_seq or os.environ.get("TOSS_ACCOUNT_SEQ")
        self._token, self._token_exp = None, 0.0
        self._s = requests.Session()

    # ---- auth -------------------------------------------------------------
    # 토스는 토큰을 새로 발급하면 이전 토큰을 무효화한다. 봇·대시보드·백테스트가 각자 발급하면
    # 서로 죽이므로, 발급한 토큰을 파일에 저장해 같은 머신의 모든 프로세스가 공유한다.
    TOKEN_FILE = pathlib.Path(__file__).resolve().parent / ".toss_token.json"

    def _load_token_file(self):
        try:
            j = json.loads(self.TOKEN_FILE.read_text())
            if j.get("client_id") == self.client_id and time.time() < j["exp"] - 60:
                return j["token"], j["exp"]
        except (OSError, ValueError, KeyError):
            pass
        return None, 0.0

    def _access_token(self, force=False):
        if not force and self._token and time.time() < self._token_exp - 60:
            return self._token
        if not force:
            tok, exp = self._load_token_file()
            if tok:
                self._token, self._token_exp = tok, exp
                return tok
        r = self._s.post(f"{BASE}/oauth2/token", data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=10)
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        self._token_exp = time.time() + j["expires_in"]
        try:
            tmp = self.TOKEN_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"client_id": self.client_id, "token": self._token,
                                       "exp": self._token_exp}))
            tmp.replace(self.TOKEN_FILE)     # 원자적 교체
        except OSError:
            pass
        return self._token

    # ---- core -------------------------------------------------------------
    def _call(self, method, path, *, params=None, json=None, account=False):
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        if account:
            if not self.account_seq:
                raise RuntimeError("TOSS_ACCOUNT_SEQ 미설정 — accounts() 로 확인 후 설정하세요")
            headers["X-Tossinvest-Account"] = str(self.account_seq)
        last = None
        for attempt in range(5):
            try:
                r = self._s.request(method, f"{BASE}{path}", params=params, json=json,
                                    headers=headers, timeout=15)
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                    continue
                if r.status_code == 401 and attempt < 3:
                    # 다른 프로세스가 토큰을 새로 받았을 수 있다 → 먼저 공유 파일의 토큰을 써 보고,
                    # 그것도 아니면 새로 발급한다 (발급하면 상대 토큰이 죽으므로 마지막 수단)
                    used = self._token
                    tok, exp = self._load_token_file()
                    if tok and tok != used:
                        self._token, self._token_exp = tok, exp
                    else:
                        self._access_token(force=True)
                    headers["Authorization"] = f"Bearer {self._token}"
                    time.sleep(0.3)
                    continue
                if r.ok:
                    return r.json().get("result")
                err = r.json().get("error", {})
                raise TossError(r.status_code, err.get("code"), err.get("message"), err.get("data"))
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                last = e          # 끊긴 응답·잘린 JSON 은 잠깐 쉬고 재시도
                time.sleep(1.0 * (attempt + 1))
        raise TossError(503, "retry-exhausted", f"재시도 초과: {last}")

    # ---- 시세·종목 (토큰만 필요) -----------------------------------------
    def prices(self, *symbols):
        return self._call("GET", "/api/v1/prices", params={"symbols": ",".join(symbols)})

    def orderbook(self, symbol):
        return self._call("GET", "/api/v1/orderbook", params={"symbol": symbol})

    def candles(self, symbol, interval="1d", count=200, before=None, adjusted=True):
        return self._call("GET", "/api/v1/candles", params={
            "symbol": symbol, "interval": interval, "count": count,
            "before": before, "adjusted": adjusted})

    def stocks(self, *symbols):
        return self._call("GET", "/api/v1/stocks", params={"symbols": ",".join(symbols)})

    def exchange_rate(self):
        return self._call("GET", "/api/v1/exchange-rate",
                          params={"baseCurrency": "USD", "quoteCurrency": "KRW"})

    def us_market_calendar(self, date=None):
        return self._call("GET", "/api/v1/market-calendar/US", params={"date": date})

    # ---- 계좌·자산 ---------------------------------------------------------
    def accounts(self):
        return self._call("GET", "/api/v1/accounts")

    def holdings(self, symbol=None):
        return self._call("GET", "/api/v1/holdings", params={"symbol": symbol}, account=True)

    def buying_power(self, currency="USD"):
        return self._call("GET", "/api/v1/buying-power", params={"currency": currency}, account=True)

    def commissions(self):
        return self._call("GET", "/api/v1/commissions", account=True)

    def sellable_quantity(self, symbol):
        return self._call("GET", "/api/v1/sellable-quantity", params={"symbol": symbol}, account=True)

    # ---- 주문 --------------------------------------------------------------
    def create_order(self, symbol, side, order_type, quantity=None, price=None,
                     order_amount=None, time_in_force=None, client_order_id=None):
        body = {"symbol": symbol, "side": side, "orderType": order_type,
                "clientOrderId": client_order_id or uuid.uuid4().hex}
        if quantity is not None:
            body["quantity"] = str(quantity)
        if price is not None:
            body["price"] = str(price)
        if order_amount is not None:
            body["orderAmount"] = str(order_amount)
        if time_in_force:
            body["timeInForce"] = time_in_force
        return self._call("POST", "/api/v1/orders", json=body, account=True)

    def buy(self, symbol, quantity=None, price=None, order_amount=None):
        """price 있으면 지정가, 없으면 시장가. order_amount 는 US 금액 기반 시장가 매수."""
        return self.create_order(symbol, "BUY", "LIMIT" if price else "MARKET",
                                 quantity=quantity, price=price, order_amount=order_amount)

    def sell(self, symbol, quantity, price=None):
        return self.create_order(symbol, "SELL", "LIMIT" if price else "MARKET",
                                 quantity=quantity, price=price)

    def cancel_order(self, order_id):
        return self._call("POST", f"/api/v1/orders/{order_id}/cancel", json={}, account=True)

    def modify_order(self, order_id, order_type, price=None, quantity=None):
        body = {"orderType": order_type}
        if price is not None:
            body["price"] = str(price)
        if quantity is not None:
            body["quantity"] = str(quantity)
        return self._call("POST", f"/api/v1/orders/{order_id}/modify", json=body, account=True)

    def orders(self, status="OPEN", symbol=None):
        return self._call("GET", "/api/v1/orders",
                          params={"status": status, "symbol": symbol}, account=True)

    def order(self, order_id):
        return self._call("GET", f"/api/v1/orders/{order_id}", account=True)


_shared = None
_shared_lock = __import__("threading").Lock()


def shared_client():
    """프로세스당 하나의 클라이언트. 여러 스레드가 각자 토큰을 받으면 서로 무효화하므로 공유한다."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = TossClient()
        return _shared


if __name__ == "__main__":
    # 읽기 전용 연결 점검 — 주문은 내지 않습니다.
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔 한글 깨짐 방지
    t = TossClient()
    print("계좌:", t.accounts())
    print("AAPL 현재가:", t.prices("AAPL"))
    if t.account_seq:
        print("보유:", t.holdings())
        print("매수 가능(USD):", t.buying_power("USD"))
