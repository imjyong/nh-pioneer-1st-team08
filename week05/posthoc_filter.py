"""7단계 사후 응용: 방법 1 확률 < 0.30인 날만 QQQ 정규장을 보유한다. 결과를 본 뒤 만든 규칙을 추가로 점검한다.
보유일과 회피일을 나누고, 문턱 민감도(0.30 동결, 나머지는 표로만)와 거래비용을 확인한다.
reality check는 아시아 특징을 원형으로 밀어 가짜 표본을 만들고, 검토했던 모델·문턱 조합 중 최대 샤프의 분포와 비교한다.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier

from second_tier import RES, load
from vol_ml import BASE, INIT, M1, STEP, build, logit

TH = 0.30
THS = [0.25, 0.30, 0.35, 0.40]
COST = 3e-4  # 왕복 3bp 가정 (MNQ 1틱 ≈ 1bp × 양쪽 + 수수료, 추정)
rng = np.random.default_rng(7)


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


def mdd(x):
    eq = np.cumsum(x)
    return (eq - np.maximum.accumulate(eq)).min()


def perf(r, hold, cost=0.0):
    x = r * hold - cost * hold
    neg = x[x < 0]
    return dict(보유일=int(hold.sum()), 일평균bp=x.mean() * 1e4, 연변동성=x.std() * np.sqrt(252),
                샤프=sharpe(x), MDD=mdd(x), 하방편차=np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(252),
                CVaR5=np.mean(np.sort(x)[: max(1, int(0.05 * len(x)))]))


def fast_walk(D):
    """HMM 제외: base, m1, tree, iforest 예측."""
    raw = [c for c in D.columns if c.startswith("r_")]
    y, n = D["y"].values, len(D)
    P = {k: np.full(n, np.nan) for k in ["base", "m1", "tree", "m3_iforest"]}
    Xb, X1, Xr = D[BASE].values, D[BASE + M1].values, D[raw].values
    for i in range(INIT, n, STEP):
        tr, te = slice(0, i), slice(i, min(i + STEP, n))
        P["base"][te] = logit().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
        P["m1"][te] = logit().fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
        P["tree"][te] = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0) \
            .fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
        iso = IsolationForest(n_estimators=100, random_state=0).fit(Xr[tr])
        sc = -iso.score_samples(Xr[: te.stop])
        X3 = np.column_stack([Xb[: te.stop], sc])
        P["m3_iforest"][te] = logit().fit(X3[tr], y[tr]).predict_proba(X3[te])[:, 1]
    return pd.DataFrame(P, index=D.index).iloc[INIT:]


def max_sharpe(P, r):
    return max(sharpe(r * (P[m].values < t)) for m in P.columns for t in THS)


def main():
    D = build("A_QQQ장중")
    q = load("QQQ")
    rq = np.log(q["Close"] / q["Open"])
    Pr = pd.read_csv(RES / "7_vol_ml_preds.csv", index_col=0, parse_dates=True)
    Pr = Pr[Pr["target"] == "A_QQQ장중"]
    r = rq.reindex(Pr.index).values
    hold = (Pr["m1"].values < TH).astype(float)
    y = Pr["y"].values

    # 보유·회피 분해
    up, dn = r > 0, r < 0
    H, A = hold == 1, hold == 0
    dec = pd.DataFrame({
        "보유": [int((up & H).sum()), int((dn & H).sum()), r[up & H].sum(), r[dn & H].sum(), r[H].mean() * 1e4, r[H].std() * np.sqrt(252)],
        "회피": [int((up & A).sum()), int((dn & A).sum()), r[up & A].sum(), r[dn & A].sum(), r[A].mean() * 1e4, r[A].std() * np.sqrt(252)],
    }, index=["상승일 수", "하락일 수", "상승일 수익 합", "하락일 손실 합", "일평균 bp", "연변동성"])
    dec.loc["큰 움직임일 수"] = [int(((y == 1) & H).sum()), int(((y == 1) & A).sum())]
    dec.loc["큰 움직임일 평균 bp"] = [r[(y == 1) & H].mean() * 1e4, r[(y == 1) & A].mean() * 1e4]
    dec.to_csv(RES / "7p_decomposition.csv", encoding="utf-8-sig")
    print("== 보유/회피 분해 ==")
    print(dec.round(4).to_string())
    print(f"하락일 회피율 {(dn & A).sum() / dn.sum():.2f}, 상승일 회피율 {(up & A).sum() / up.sum():.2f}")

    # 문턱 민감도와 비용
    rows = [dict(규칙="매일 보유", 비용="없음", **perf(r, np.ones_like(r)))]
    rows.append(dict(규칙="매일 보유", 비용="3bp", **perf(r, np.ones_like(r), COST)))
    for t in THS:
        h = (Pr["m1"].values < t).astype(float)
        rows.append(dict(규칙=f"m1<{t:.2f}" + (" (동결)" if t == TH else ""), 비용="없음", **perf(r, h)))
        rows.append(dict(규칙=f"m1<{t:.2f}" + (" (동결)" if t == TH else ""), 비용="3bp", **perf(r, h, COST)))
    S = pd.DataFrame(rows)
    S.to_csv(RES / "7p_threshold_cost.csv", index=False, encoding="utf-8-sig")
    print("\n== 문턱·비용 ==")
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # reality check
    obs_P = Pr[["base", "m1", "tree", "m3_iforest"]]
    obs = max_sharpe(obs_P, r)
    asia = [c for c in D.columns if c.startswith("r_")] + M1
    n = len(D)
    shifts = rng.choice(np.arange(60, n - 60), size=200, replace=False)
    fake = []
    for k in shifts:
        Df = D.copy()
        Df[asia] = np.roll(D[asia].values, k, axis=0)
        Pf = fast_walk(Df)
        fake.append(max_sharpe(Pf, rq.reindex(Pf.index).values))
    fake = np.array(fake)
    p = (1 + (fake >= obs).sum()) / (1 + len(fake))
    pd.Series(fake).to_csv(RES / "7p_reality_check.csv", index=False)
    print(f"\n== reality check (200 원형 이동, 모델 4 × 문턱 4 중 최대 샤프) ==")
    print(f"실제 최대 샤프 {obs:.2f}, 가짜 중앙값 {np.median(fake):.2f}, 95% {np.quantile(fake, .95):.2f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
