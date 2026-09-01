"""
VAE 잠재변수 접목 분류 모형 (논문 3.3절) + Walk-forward 검증 (논문 3.4.4절).

  잠재 변수 15차원 (+ 기술적 지표 특징) -> 로지스틱 / SVM / 랜덤포레스트
  앙상블 확률 = 세 모형 확률의 평균
  평가지표: Accuracy, F1-Score, AUC (논문 3.4절)

표시되는 정확도는 모두 고정 임계값 0.5 기준이다. 임계값을 탐색해 얻은
정확도는 같은 표본으로 고르고 같은 표본으로 채점한 값이므로 여기서
보고하지 않는다 (backtest.rolling_calibration 참조).
"""
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def fit_classifiers(Z: np.ndarray, y: np.ndarray):
    """잠재변수(+지표) Z 에 대해 논문의 세 가지 분류 모형 학습."""
    scaler = StandardScaler().fit(Z)
    Zs = scaler.transform(Z)
    logit = LogisticRegression(max_iter=2000).fit(Zs, y)
    svm = CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, gamma="scale"),
                              ensemble=False).fit(Zs, y)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Zs, y)
    return {"scaler": scaler, "logit": logit, "svm": svm, "rf": rf}


def predict_proba(bundle, Z: np.ndarray):
    """모형별 상승 확률 + 앙상블 평균."""
    Zs = bundle["scaler"].transform(Z)
    p = {
        "logistic": bundle["logit"].predict_proba(Zs)[:, 1],
        "svm": bundle["svm"].predict_proba(Zs)[:, 1],
        "rf": bundle["rf"].predict_proba(Zs)[:, 1],
    }
    p["ensemble"] = (p["logistic"] + p["svm"] + p["rf"]) / 3.0
    return p


def majority_baseline(actuals) -> float:
    """
    '아무 것도 학습하지 않은' 기준선 = 다수 클래스 비율.

    라벨이 불균형하면(거래비용 임계 라벨은 하락이 더 많다) 항상 다수
    클래스만 찍어도 50% 를 훌쩍 넘는다. 정확도는 0.5 가 아니라 이 값과
    비교해야 의미가 있다.
    """
    a = np.asarray(list(actuals), dtype=float)
    if a.size == 0:
        return 0.5
    up = float((a == 1).mean())
    return max(up, 1.0 - up)


def accuracy_stats(hits: int, n: int, p0: float = 0.5) -> dict:
    """
    정확도의 95% 신뢰구간과 기준선(p0) 대비 양측 p-value.

    p0 에는 다수 클래스 비율(majority_baseline)을 넣는다. 표본이 작은
    walk-forward 결과를 순위표에 그대로 노출하면 노이즈를 실력으로
    오독하게 되므로, 정확도와 함께 항상 표본 수·구간·기준선·유의성을
    같이 보고한다. (정규근사, scipy 의존 없음)
    """
    if n <= 0:
        nan = float("nan")
        return {"accuracy": nan, "baseline": float(p0), "n": 0,
                "ci_low": nan, "ci_high": nan,
                "p_value": nan, "significant": False, "se": nan}
    acc = hits / n
    se_obs = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    se_null = math.sqrt(max(p0 * (1 - p0), 1e-12) / n)
    z = (acc - p0) / se_null
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "accuracy": float(acc),
        "baseline": float(p0),
        "n": int(n),
        "ci_low": float(max(0.0, acc - 1.96 * se_obs)),
        "ci_high": float(min(1.0, acc + 1.96 * se_obs)),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05 and acc > p0),
        "se": float(se_null),
    }


def _metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n": int(len(y_true)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def walk_forward(Z: np.ndarray, y: np.ndarray, dates, n_walks: int = 6,
                 test_days: int = 21, embargo: int = 0):
    """
    Walk-forward Validation (논문 [그림 3.6], [표 3.6]).

    마지막 n_walks * test_days 구간을 1개월(≈21거래일) 단위로 나누어,
    각 walk 마다 그 이전 전체 데이터로 분류기를 학습하고 다음 구간을 검증.
    분류기와 스케일러는 walk 마다 학습 구간만으로 재적합한다.
    (VAE 는 pipeline 에서 검증 구간을 제외하고 미리 1회 학습된 것을 사용)

    embargo: 학습 구간 끝에서 잘라낼 표본 수. horizon>1 인 라벨은 미래 h일
    종가를 참조하므로, 학습 구간 마지막 h개 표본의 라벨은 테스트 구간의
    가격을 이미 알고 있다. embargo=horizon-1 로 그 구간을 버려 누수를 막는다.
    """
    n = len(y)
    results = []
    per_day = []  # 시기 분석용: 각 테스트일의 (날짜, walk, 확률, 실제라벨, 적중)
    for w in range(n_walks):
        test_end = n - (n_walks - 1 - w) * test_days
        test_start = test_end - test_days
        if test_start <= 40:
            continue
        train_end = test_start - embargo          # 라벨 누수 구간 제외
        if train_end <= 40:
            continue
        bundle = fit_classifiers(Z[:train_end], y[:train_end])
        probs = predict_proba(bundle, Z[test_start:test_end])
        yt = y[test_start:test_end]
        row = {
            "walk": w + 1,
            "train_end": str(dates[train_end - 1].date()),
            "test_start": str(dates[test_start].date()),
            "test_end": str(dates[test_end - 1].date()),
        }
        for name in ("logistic", "svm", "rf", "ensemble"):
            row[name] = _metrics(yt, probs[name])
        results.append(row)
        ens = probs["ensemble"]
        for i in range(len(yt)):
            per_day.append({
                "date": str(dates[test_start + i].date()),
                "walk": w + 1,
                "prob": float(ens[i]),
                "actual": int(yt[i]),
                "hit": bool((ens[i] >= 0.5) == (yt[i] == 1)),
            })
    # 전체 평균 (walk 별 지표의 산술평균)
    avg = {}
    for name in ("logistic", "svm", "rf", "ensemble"):
        avg[name] = {
            k: float(np.nanmean([r[name][k] for r in results]))
            for k in ("accuracy", "f1", "auc")
        } if results else {}
    # 표본 전체를 모은 정확도 + 신뢰구간 + 유의성 (임계값 0.5 고정)
    hits = sum(1 for d in per_day if d["hit"])
    base = majority_baseline(d["actual"] for d in per_day)
    pooled = accuracy_stats(hits, len(per_day), p0=base)
    return {"walks": results, "average": avg, "per_day": per_day,
            "pooled": pooled}

# ---------------------------------------------------------------------------
# 특징 구성 A/B 비교 (ablation)
#
# 조병호(2021)의 주장 — 여러 기술적 기법을 결합하면 단일 기법보다 낫다 —
# 을 이 시스템에서 실제로 검증하려면, 같은 walk·같은 VAE·같은 라벨 위에서
# 입력 구성만 바꿔 비교해야 한다. 세 변형을 같은 날짜에 대해 평가하므로
# 두 모형의 우열은 짝지은 표본 검정(McNemar)으로 판정한다.
# ---------------------------------------------------------------------------

VARIANTS = (
    ("latent", "VAE 잠재변수만 (조예서 2022)"),
    ("indicator", "기술적 지표만 (조병호 2021)"),
    ("combined", "결합 (본 구현)"),
)


def mcnemar(rows_a, rows_b, key_a="a", key_b="b") -> dict:
    """
    같은 날짜에 대한 두 모형의 적중 여부를 짝지어 비교 (McNemar).

    두 모형이 같은 표본을 쓰므로 정확도 차이를 독립 표본처럼 비교하면
    안 된다. 한쪽만 맞힌 날의 개수(불일치 쌍)만으로 검정한다.
    연속성 보정 카이제곱 근사 (자유도 1).
    """
    da = {r["date"]: r for r in rows_a}
    db = {r["date"]: r for r in rows_b}
    common = [d for d in da if d in db]
    a_only = sum(1 for d in common if da[d]["hit"] and not db[d]["hit"])
    b_only = sum(1 for d in common if db[d]["hit"] and not da[d]["hit"])
    disc = a_only + b_only
    if disc == 0:
        p = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(a_only - b_only) - 1.0) ** 2 / disc
        chi2 = max(chi2, 0.0)
        p = math.erfc(math.sqrt(chi2 / 2.0))
    if p < 0.05 and a_only != b_only:
        verdict = f"{key_a} 우세" if a_only > b_only else f"{key_b} 우세"
    else:
        verdict = "차이 없음"
    return {"a": key_a, "b": key_b, "n": len(common),
            "a_only": a_only, "b_only": b_only, "discordant": disc,
            "chi2": float(chi2), "p_value": float(p), "verdict": verdict,
            "significant": bool(p < 0.05 and a_only != b_only)}


def ablation(Z: np.ndarray, y: np.ndarray, dates, n_latent: int,
             n_walks: int = 6, test_days: int = 21, progress=None) -> dict:
    """
    입력 구성만 바꿔 같은 walk-forward 로 비교한다.

    Z 는 [잠재변수 n_latent | 기술적 지표] 순으로 결합된 행렬이어야 한다.
    반환: 변형별 성능(기준선·CI·p-value 포함) + 변형 간 McNemar 비교.
    """
    total = Z.shape[1]
    cols = {
        "latent": np.arange(0, n_latent),
        "indicator": np.arange(n_latent, total),
        "combined": np.arange(0, total),
    }
    variants, per_day_by = [], {}
    for i, (key, label) in enumerate(VARIANTS):
        idx = cols[key]
        if len(idx) == 0:
            continue
        if progress:
            progress(i, len(VARIANTS), label)
        wf = walk_forward(Z[:, idx], y, dates, n_walks=n_walks,
                          test_days=test_days)
        pooled = wf["pooled"]
        ens = wf["average"]["ensemble"]
        variants.append({
            "key": key,
            "label": label,
            "dims": int(len(idx)),
            "accuracy": pooled["accuracy"],
            "baseline": pooled["baseline"],
            "ci_low": pooled["ci_low"],
            "ci_high": pooled["ci_high"],
            "p_value": pooled["p_value"],
            "significant": pooled["significant"],
            "n": pooled["n"],
            "auc": ens.get("auc", float("nan")),
            "f1": ens.get("f1", float("nan")),
            "walk_acc": [w["ensemble"]["accuracy"] for w in wf["walks"]],
        })
        per_day_by[key] = wf["per_day"]

    pairs = []
    for a, b in (("combined", "latent"), ("combined", "indicator"),
                 ("latent", "indicator")):
        if a in per_day_by and b in per_day_by:
            pairs.append(mcnemar(per_day_by[a], per_day_by[b], a, b))

    best = max(variants, key=lambda v: v["accuracy"]) if variants else None
    return {"variants": variants, "pairs": pairs,
            "best": best["key"] if best else None,
            "n_walks": n_walks, "test_days": test_days}
