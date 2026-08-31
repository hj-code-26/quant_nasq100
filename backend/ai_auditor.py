"""
AI 코드 감사 에이전트 — 내부 진단 루틴을 실행해 파이프라인의 수학적
불변식과 모형 건전성을 검증하고, 실패 항목이 있으면 Claude API 로
원인 분석과 코드 패치를 받아 적용 → 재검증 → 실패 시 롤백한다.

루프:  진단 → (실패 시) AI 분석/패치 제안 → 백업 후 적용
       → 컴파일 + 재진단 → 악화되면 롤백
"""
import importlib
import json
import pathlib
import py_compile
import re
import shutil
import threading
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
# AI 패치 허용 파일 (화이트리스트)
PATCHABLE = ["gaf.py", "vae.py", "models.py", "quant.py", "backtest.py",
             "pipeline.py", "nasdaq100.py", "data.py"]

_STATE = {"status": "idle", "log": [], "checks": [], "ai_available": None,
          "last_run": None, "patches_applied": []}
_lock = threading.Lock()


def get_state():
    with _lock:
        return json.loads(json.dumps(_STATE, ensure_ascii=False, default=str))


def _log(msg):
    with _lock:
        _STATE["log"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})
        _STATE["log"] = _STATE["log"][-200:]


def _set(**kw):
    with _lock:
        _STATE.update(kw)


# ---------------- 진단 루틴 ----------------

def _sample_df():
    """진단용 데이터: 캐시/네트워크 실패 시 합성 랜덤워크로 대체."""
    try:
        from . import data as data_mod
        return data_mod.history("AAPL", "1y")
    except Exception:
        rng = np.random.default_rng(7)
        n = 260
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        idx = pd.bdate_range("2025-08-01", periods=n)
        return pd.DataFrame({
            "Open": c * (1 + rng.normal(0, 0.004, n)),
            "High": c * (1 + abs(rng.normal(0, 0.008, n))),
            "Low": c * (1 - abs(rng.normal(0, 0.008, n))),
            "Close": c, "Volume": np.full(n, 1e6)}, index=idx)


def run_diagnostics() -> list:
    """핵심 불변식/건전성 검사 모음. 각 항목 {name, passed, detail}."""
    checks = []

    def add(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"예외: {type(e).__name__}: {e}"
        checks.append({"name": name, "passed": bool(ok), "detail": str(detail)})

    # 0) 코드 컴파일
    def c_compile():
        bad = []
        # ablation.py 는 컴파일 검사만 하고 AI 자동 패치 대상에서는 제외한다
        # (성능 비교를 수행하는 코드까지 자동 수정되면 검증의 의미가 없다)
        for f in PATCHABLE + ["server.py", "ai_auditor.py", "ablation.py",
                              "indicators.py", "toss.py"]:
            try:
                py_compile.compile(str(ROOT / f), doraise=True)
            except py_compile.PyCompileError as e:
                bad.append(f"{f}: {e.msg}")
        return (not bad, "; ".join(bad) or "모든 모듈 컴파일 통과")
    add("코드 컴파일", c_compile)

    from . import gaf, models, vae  # 컴파일 확인 후 임포트
    df = _sample_df()

    # 1) GAF 수학적 불변식
    def c_gaf():
        x = np.linspace(1, 2, 20)
        g = gaf.gadf(x)
        sym = np.allclose(g, -g.T, atol=1e-8)          # GADF 반대칭
        diag = np.allclose(np.diag(g), 0, atol=1e-8)   # sin(0)=0
        rng_ok = g.min() >= -1 - 1e-9 and g.max() <= 1 + 1e-9
        gs = gaf.gasf(x)
        sym2 = np.allclose(gs, gs.T, atol=1e-8)        # GASF 대칭
        ok = sym and diag and rng_ok and sym2
        return (ok, f"반대칭={sym}, 대각0={diag}, 범위={rng_ok}, GASF대칭={sym2}")
    add("GAF 불변식", c_gaf)

    # 2) 데이터셋 무결성
    X, y, dates = gaf.build_dataset(df, 20)

    def c_dataset():
        n_nan = int(np.isnan(X).sum())
        in_range = X.min() >= 0 and X.max() <= 1
        bal = float(y[y >= 0].mean())
        fresh = (pd.Timestamp.now() - pd.Timestamp(dates[-1])).days <= 7
        ok = n_nan == 0 and in_range and 0.25 <= bal <= 0.75 and len(X) > 100
        return (ok, f"NaN={n_nan}, 범위[0,1]={in_range}, 상승비율={bal:.2f}, "
                    f"최신성(7일)={fresh}, 표본={len(X)}")
    add("데이터셋 무결성", c_dataset)

    # 3) VAE 건전성 (소규모 학습)
    def c_vae():
        m = vae.train_vae(X[y >= 0][:150], epochs=3)
        Z = vae.extract_latents(m, X[:150])
        collapse = float(Z.std(axis=0).min())
        finite = np.isfinite(Z).all()
        ok = finite and collapse > 1e-4
        return (ok, f"잠재변수 유한={finite}, 최소 std={collapse:.4f} "
                    f"(사후분포 붕괴 여부)")
    add("VAE 건전성", c_vae)

    # 4) 분류기·검증 루틴
    def c_clf():
        m = vae.train_vae(X[y >= 0][:200], epochs=3)
        Z = vae.extract_latents(m, X)
        yl = y[y >= 0]
        Zl = Z[y >= 0]
        dl = [d for d, ok_ in zip(dates, y >= 0) if ok_]
        b = models.fit_classifiers(Zl, yl)
        p = models.predict_proba(b, Zl[:50])["ensemble"]
        wf = models.walk_forward(Zl, yl, dl, n_walks=2, test_days=15)
        acc = wf["average"]["ensemble"].get("accuracy", 0)
        probs_ok = (0 <= p.min()) and (p.max() <= 1) and 0.05 < p.mean() < 0.95
        ok = probs_ok and acc > 0.30 and np.isfinite(acc)
        return (ok, f"확률범위 OK={probs_ok}, WF정확도={acc:.3f}")
    add("분류기·Walk-forward", c_clf)

    # 5) gs-quant / 예상주가 루틴
    def c_quant():
        from . import quant
        ind = quant.analytics(df["Close"])
        exp = quant.expected_price(df["Close"], 0.6)
        band_ok = exp["band_low"] < exp["expected_price"] < exp["band_high"]
        rsi = ind.get("rsi_14", float("nan"))
        rsi_ok = 0 <= rsi <= 100 if rsi == rsi else False
        return (band_ok and rsi_ok,
                f"밴드순서={band_ok}, RSI범위={rsi_ok}({rsi:.1f})")
    add("gs-quant·예상주가", c_quant)

    # 6) 검증 위생 — 누수 방지 불변식
    def c_leak():
        from . import backtest as bt_
        from . import pipeline as pl
        # (a) VAE 학습 구간이 walk-forward 검증 구간을 침범하지 않는가
        n_lab, nw, td = 800, 6, 21
        end, lf = pl.vae_fit_end(n_lab, nw, td)
        a_ok = bool(lf) and end == n_lab - nw * td
        # (b) 임계값이 '직전 walk 들'로만 결정되는가
        rng = np.random.default_rng(3)
        rows = [{"date": f"d{i}", "walk": i // 10 + 1,
                 "prob": float(rng.uniform(0.3, 0.7)),
                 "actual": int(rng.integers(0, 2))} for i in range(40)]
        cal = bt_.rolling_calibration(rows)
        b_ok = all(abs(r["threshold"] - 0.5) < 1e-9
                   for r in rows if r["walk"] == 1)
        exp_t, _ = bt_._search_threshold([r for r in rows if r["walk"] <= 2])
        c_ok = all(abs(r["threshold"] - exp_t) < 1e-9
                   for r in rows if r["walk"] == 3)
        # (c) 첫 walk 은 채점에서 제외
        d_ok = cal["n_eval"] == 30 and cal["walks_scored"] == 3
        ok = a_ok and b_ok and c_ok and d_ok
        return (ok, f"VAE 구간분리={a_ok}, 첫walk=0.5 {b_ok}, "
                    f"과거전용 임계={c_ok}, 채점표본={cal['n_eval']}")
    add("검증 위생(누수 방지)", c_leak)

    # 7) 특징 인과성 — 미래 봉을 바꿔도 과거 특징이 변하면 안 된다
    def c_causal():
        from . import quant
        F1 = quant.feature_matrix(df)
        d2 = df.copy()
        d2.iloc[-5:] = d2.iloc[-5:] * 1.25
        F2 = quant.feature_matrix(d2)
        k = len(df) - 5
        same = bool(np.allclose(F1.to_numpy()[:k], F2.to_numpy()[:k],
                                atol=1e-10))
        n_feat = int(F1.shape[1])
        return (same and n_feat >= 8,
                f"과거 특징 불변={same}, 특징 수={n_feat}")
    add("특징 인과성", c_causal)

    return checks


# ---------------- Claude AI 수정 루프 ----------------

def _client():
    try:
        import anthropic
        c = anthropic.Anthropic()  # env/프로필에서 자격 증명 자동 해석
        if not c.api_key and not getattr(c, "auth_token", None):
            return None
        return c
    except Exception:
        return None


def _ask_ai(client, failed, checks):
    """실패 진단 + 소스를 Claude 에 보내 분석/패치(JSON)를 받는다."""
    import anthropic
    sources = []
    for f in PATCHABLE:
        try:
            sources.append(f"### {f}\n```python\n{(ROOT / f).read_text(encoding='utf-8')}\n```")
        except OSError:
            pass
    report = json.dumps(checks, ensure_ascii=False, indent=1)
    prompt = f"""당신은 퀀트 트레이딩 파이프라인의 코드 감사 AI입니다.
아래 진단 결과 중 실패 항목의 근본 원인을 소스에서 찾아 최소 수정 패치를 제안하세요.

## 진단 결과
{report}

## 실패 항목
{json.dumps(failed, ensure_ascii=False)}

## 소스 코드
{chr(10).join(sources)}

## 응답 형식 (JSON만, 다른 텍스트 금지)
{{"analysis": "원인 분석 요약(한국어)",
  "patches": [{{"file": "파일명.py", "find": "정확히 일치하는 원본 코드", "replace": "수정 코드"}}]}}
패치가 불필요하면 patches 를 빈 배열로 하세요. find 는 해당 파일에서 유일하게
일치해야 합니다. 수학적 불변식(GADF 반대칭, VAE 손실식 등)은 논문 정의를 따르세요."""
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {"analysis": text, "patches": []}


def _apply_patches(patches) -> list:
    """화이트리스트 파일에 백업 후 패치 적용. 적용 내역 반환."""
    applied = []
    for p in patches:
        f = pathlib.Path(p.get("file", "")).name
        if f not in PATCHABLE:
            _log(f"거부(화이트리스트 외): {f}")
            continue
        path = ROOT / f
        src = path.read_text(encoding="utf-8")
        find = p.get("find", "")
        if not find or src.count(find) != 1:
            _log(f"거부(find 불일치 {src.count(find)}회): {f}")
            continue
        bak = path.with_suffix(".py.bak")
        shutil.copy2(path, bak)
        path.write_text(src.replace(find, p.get("replace", "")), encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
            applied.append({"file": f, "backup": str(bak)})
            _log(f"패치 적용: {f}")
        except py_compile.PyCompileError as e:
            shutil.copy2(bak, path)
            _log(f"롤백(컴파일 실패): {f} — {e.msg[:80]}")
    return applied


def _rollback(applied):
    for a in applied:
        shutil.copy2(a["backup"], ROOT / a["file"])
        _log(f"롤백: {a['file']}")


def _reload_modules():
    from . import backtest, data, gaf, models, nasdaq100, pipeline, quant, vae
    for m in (gaf, vae, models, quant, backtest, data, nasdaq100, pipeline):
        importlib.reload(m)


def run_audit(auto_fix: bool = True):
    """전체 감사 루프 (백그라운드 스레드에서 실행)."""
    try:
        _set(status="running", log=[], checks=[], patches_applied=[])
        _log("진단 루틴 시작…")
        checks = run_diagnostics()
        _set(checks=checks)
        failed = [c for c in checks if not c["passed"]]
        _log(f"진단 완료: {len(checks) - len(failed)}/{len(checks)} 통과")

        if not failed:
            _set(status="passed", last_run=time.time())
            return
        client = _client() if auto_fix else None
        _set(ai_available=client is not None)
        if client is None:
            _log("실패 항목 있음 — ANTHROPIC_API_KEY 미설정으로 AI 수정 생략")
            _set(status="failed", last_run=time.time())
            return

        for it in range(2):  # 최대 2회 수정 시도
            _log(f"Claude 분석 요청 (시도 {it + 1}/2)…")
            ai = _ask_ai(client, failed, checks)
            _log(f"AI 분석: {ai.get('analysis', '')[:300]}")
            patches = ai.get("patches", [])
            if not patches:
                _log("AI가 코드 패치 불필요로 판단")
                _set(status="failed", last_run=time.time())
                return
            applied = _apply_patches(patches)
            if not applied:
                _set(status="failed", last_run=time.time())
                return
            _reload_modules()
            _log("재진단 중…")
            new_checks = run_diagnostics()
            new_failed = [c for c in new_checks if not c["passed"]]
            if len(new_failed) < len(failed):
                _set(checks=new_checks,
                     patches_applied=_STATE["patches_applied"] + applied)
                _log(f"개선됨: 실패 {len(failed)} → {len(new_failed)}")
                if not new_failed:
                    _set(status="fixed", last_run=time.time())
                    return
                failed, checks = new_failed, new_checks
            else:
                _rollback(applied)
                _reload_modules()
                _log("개선 없음 — 롤백 완료")
                _set(status="failed", last_run=time.time())
                return
        _set(status="failed", last_run=time.time())
    except Exception as e:  # noqa: BLE001
        _log(f"감사 오류: {type(e).__name__}: {e}")
        _set(status="error", last_run=time.time())


def start_audit(auto_fix: bool = True):
    if _STATE["status"] == "running":
        return get_state()
    threading.Thread(target=run_audit, args=(auto_fix,), daemon=True).start()
    time.sleep(0.05)
    return get_state()
