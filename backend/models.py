"""
VAE 잠재변수 접목 분류 모형 (논문 3.3절) + Walk-forward 검증 (논문 3.4.4절).

  잠재 변수 15차원 -> 로지스틱 회귀 / SVM / 랜덤포레스트
  앙상블 확률 = 세 모형 확률의 평균
  평가지표: Accuracy, F1-Score, AUC (논문 3.4절)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def fit_classifiers(Z: np.ndarray, y: np.ndarray):
    """잠재변수 Z 에 대해 논문의 세 가지 분류 모형 학습."""
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


def _metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
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
    (실시간 구동을 위해 VAE 는 고정하고 분류기만 walk 별 재학습)

    embargo: 학습 구간 끝에서 잘라낼 표본 수. horizon>1 인 라벨은 미래 h일
    종가를 참조하므로, 학습 구간 마지막 h개 표본의 라벨은 테스트 구간의
    가격을 이미 알고 있다. embargo=horizon-1 로 그 구간을 버려 누수를 막는다.
    """
    n = len(y)
    results = []
    per_day = []  # 시기 분석용: 각 테스트일의 (날짜, 앙상블확률, 실제라벨, 적중여부)
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
                "prob": float(ens[i]),
                "actual": int(yt[i]),
                "hit": bool((ens[i] >= 0.5) == (yt[i] == 1)),
            })
    # 전체 평균
    avg = {}
    for name in ("logistic", "svm", "rf", "ensemble"):
        avg[name] = {
            k: float(np.nanmean([r[name][k] for r in results]))
            for k in ("accuracy", "f1", "auc")
        } if results else {}
    return {"walks": results, "average": avg, "per_day": per_day}
