"""§8 운영 격리 하네스 — mtime 확인을 넘어 **실제로 못 건드리게** 막고, 사후에 검증한다.

`python research/isolation.py -- python research/xxx.py` 로 감싸 실행하거나,
연구 스크립트 맨 위에서 `from research.isolation import guard; guard()` 로 켠다.

막는 것
  1. 자격증명 격리 — TOSS_*/ANTHROPIC_* 환경변수를 프로세스 안에서 지운다.
     .env 는 **읽기 자체를 차단**한다. 비밀값은 어디에도 출력하지 않는다.
  2. 운영 경로 쓰기 차단 — 운영 DB(+WAL/SHM)·토큰 파일·.env·로그·data_cache 에
     쓰기 모드로 여는 모든 경로(open / pathlib / sqlite3.connect)를 예외로 막는다.
  3. 실브로커 요청 차단 — requests.Session.request 에서 토스 호스트를 막는다.
     yfinance(라이선스 범위 내 무료 시세)는 ALLOW_HOSTS 에 명시했을 때만 통과.
  4. 변경 감시 — 보호 대상의 (크기, mtime_ns, sha256) 을 실행 전후로 비교한다.
     SQLite 는 -wal/-shm 까지 본다 (본 파일이 그대로여도 WAL 에 쓰였을 수 있다).
"""
import builtins
import hashlib
import io
import json
import os
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
(RESEARCH / "out").mkdir(parents=True, exist_ok=True)

# 운영 자산 — 읽기만 허용(일부는 읽기도 금지), 쓰기는 전면 차단
PROTECTED = [
    ROOT / "trading_decisions.db", ROOT / "trading_decisions.db-wal",
    ROOT / "trading_decisions.db-shm", ROOT / "trading_decisions.db-journal",
    ROOT / ".toss_token.json", ROOT / ".env", ROOT / "autotrade.log",
    ROOT / "server.log",
]
PROTECTED_DIRS = [ROOT / "data_cache", ROOT / "models_cache", ROOT / "backend"]
NO_READ = [ROOT / ".env", ROOT / ".toss_token.json"]        # 비밀값 — 읽기도 막는다
# 로그는 막는 대신 **연구 폴더로 돌린다.** autotrade 를 import 하면 모듈 로드 시점에
# autotrade.log 를 append 모드로 연다 — 막으면 import 자체가 안 되고, 그냥 두면
# 운영 로그가 연구 실행으로 오염된다.
REDIRECT = {ROOT / "autotrade.log": RESEARCH / "out" / "research_autotrade.log",
            ROOT / "server.log": RESEARCH / "out" / "research_server.log"}
SOURCES = sorted(ROOT.glob("*.py"))                          # 운영 소스 변경 감시
CRED_PREFIXES = ("TOSS_", "ANTHROPIC_", "AWS_", "OPENAI_")
BROKER_HOSTS = ("toss", "tossbank", "tossinvest")
ALLOW_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com",
               "fc.yahoo.com", "finance.yahoo.com")

_installed = False
_REAL_OPEN = builtins.open          # 격리를 켜기 전의 open — 지문 계산 전용


def real_bytes(p):
    return _REAL_OPEN(p, "rb")


_writes_blocked = []            # (경로, 모드) — 시도 기록. 비밀값은 담지 않는다.


class IsolationError(RuntimeError):
    pass


def _empty(mode):
    """비밀 파일 읽기 시 돌려주는 빈 스트림. 값이 프로세스로 들어오지 않는다."""
    return io.BytesIO(b"") if "b" in str(mode) else io.StringIO("")


def _redirect(p):
    """운영 로그 경로면 연구 폴더 경로로 바꿔 준다. 아니면 None."""
    if isinstance(p, int):
        return None
    try:
        rp = pathlib.Path(p).resolve()
    except (OSError, ValueError):
        return None
    for src, dst in REDIRECT.items():
        if rp == src.resolve() or rp == src:
            return dst
    return None


def _is_protected(p, write):
    if isinstance(p, int):          # 파일 디스크립터 — 경로가 아니다 (subprocess 파이프 등)
        return None
    try:
        rp = pathlib.Path(p).resolve()
    except (OSError, ValueError):
        return None
    if not write and rp in [x.resolve() for x in NO_READ if x.exists()]:
        return "비밀 파일 읽기 차단(빈 내용으로 대체)"
    if not write:
        return None
    if rp in [x.resolve() for x in PROTECTED] or any(
            str(rp).startswith(str(d.resolve())) for d in PROTECTED_DIRS):
        return "운영 경로 쓰기 금지"
    if rp.parent == ROOT and rp.suffix == ".py":
        return "운영 소스 쓰기 금지"
    return None


def _fingerprint():
    out = {}
    for p in [*PROTECTED, *SOURCES] + [f for d in PROTECTED_DIRS if d.exists()
                                       for f in sorted(d.glob("*")) if f.is_file()]:
        if p.exists():
            st = p.stat()
            if p in NO_READ or st.st_size >= 50_000_000:
                # 비밀 파일은 **내용을 읽지 않는다.** 크기·mtime 만으로 변경을 감시한다.
                h = "no-read(size+mtime only)"
            else:
                with real_bytes(p) as f:
                    h = hashlib.sha256(f.read()).hexdigest()
            out[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns, h]
        else:
            out[str(p.relative_to(ROOT))] = None
    return out


def guard(allow_network=()):
    """격리를 켠다. allow_network 에 준 호스트만 HTTP 를 허용한다."""
    global _installed
    if _installed:
        return
    _installed = True

    # 1) 자격증명 격리 — 값은 절대 출력/저장하지 않는다.
    dropped = [k for k in list(os.environ) if k.startswith(CRED_PREFIXES)]
    for k in dropped:
        os.environ.pop(k, None)
    os.environ["DRY_RUN"] = "1"

    # 2) 쓰기 차단
    real_open = builtins.open
    real_io_open = io.open

    def _open(file, mode="r", *a, **kw):
        w = any(c in str(mode) for c in "wxa+")
        r = _redirect(file)
        if r is not None:
            _writes_blocked.append((str(file), str(mode), "로그 리다이렉트"))
            return real_open(r, mode, *a, **kw)
        why = _is_protected(file, w)
        if why:
            _writes_blocked.append((str(file), str(mode), why))
            if not w:                       # 비밀 파일: 예외 대신 **빈 내용**을 준다.
                return _empty(mode)         # 값이 프로세스에 들어오지 않으면서 import 는 산다.
            raise IsolationError(f"{why}: {file} (mode={mode})")
        return real_open(file, mode, *a, **kw)

    builtins.open = _open
    io.open = _open

    real_wt, real_wb = pathlib.Path.write_text, pathlib.Path.write_bytes
    real_popen, real_replace, real_unlink = (pathlib.Path.open, pathlib.Path.replace,
                                             pathlib.Path.unlink)

    def _wrap(fn, write=True):
        def inner(self, *a, **kw):
            mode = kw.get("mode", a[0] if (a and fn is real_popen) else "w")
            w = write if fn is not real_popen else any(c in str(mode) for c in "wxa+")
            r = _redirect(self)
            if r is not None:
                _writes_blocked.append((str(self), str(mode), "로그 리다이렉트"))
                return fn(r, *a, **kw)
            why = _is_protected(self, w)
            if why:
                _writes_blocked.append((str(self), str(mode), why))
                if not w:
                    return _empty(mode)
                raise IsolationError(f"{why}: {self}")
            return fn(self, *a, **kw)
        return inner

    pathlib.Path.write_text = _wrap(real_wt)
    pathlib.Path.write_bytes = _wrap(real_wb)
    pathlib.Path.open = _wrap(real_popen)
    pathlib.Path.replace = _wrap(real_replace)
    pathlib.Path.unlink = _wrap(real_unlink)

    real_connect = sqlite3.connect

    def _connect(database, *a, **kw):
        s = str(database)
        ro = "mode=ro" in s and kw.get("uri")
        why = _is_protected(s.split("?")[0].replace("file:", ""), not ro)
        if why:
            _writes_blocked.append((s, "sqlite", why))
            raise IsolationError(f"{why}: {s}")
        return real_connect(database, *a, **kw)

    sqlite3.connect = _connect

    # 3) 실브로커 요청 차단
    try:
        import requests

        real_req = requests.sessions.Session.request

        def _request(self, method, url, *a, **kw):
            host = str(url).split("//")[-1].split("/")[0].lower()
            if any(b in host for b in BROKER_HOSTS):
                raise IsolationError(f"실브로커 요청 차단: {method} {host}")
            if host not in tuple(allow_network) + ALLOW_HOSTS:
                raise IsolationError(f"허용되지 않은 호스트: {host}")
            return real_req(self, method, url, *a, **kw)

        requests.sessions.Session.request = _request
    except ImportError:
        pass
    return dropped


def snapshot(path=None):
    path = pathlib.Path(path or RESEARCH / "out" / "isolation_baseline.json")
    fp = _fingerprint()
    path.parent.mkdir(parents=True, exist_ok=True)
    _raw_write(path, json.dumps(fp, indent=1))
    return fp


def _raw_write(path, text):
    """격리가 켜진 뒤에도 research/out 에는 써야 한다 — 보호 대상이 아니므로 통과한다."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def verify(baseline_path=None):
    p = pathlib.Path(baseline_path or RESEARCH / "out" / "isolation_baseline.json")
    if not p.exists():
        raise IsolationError("baseline 이 없다 — snapshot() 을 먼저 부를 것")
    with open(p, encoding="utf-8") as f:
        before = json.load(f)
    after = _fingerprint()
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    return {"changed": sorted(changed), "blocked_attempts": len(_writes_blocked),
            "watched": len(after), "ok": not changed}


def main(argv):
    import subprocess
    if "--" not in argv:
        snapshot()
        print(f"baseline 기록: {len(_fingerprint())}개 파일")
        return 0
    cmd = argv[argv.index("--") + 1:]
    snapshot()
    env = {k: v for k, v in os.environ.items() if not k.startswith(CRED_PREFIXES)}
    env["RESEARCH_ISOLATED"] = "1"
    r = subprocess.run(cmd, env=env)
    v = verify()
    print(f"\n[격리 검증] 감시 {v['watched']}개 · 변경 {len(v['changed'])}개 · "
          f"{'OK' if v['ok'] else 'FAIL: ' + ', '.join(v['changed'])}")
    return r.returncode or (0 if v["ok"] else 2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
