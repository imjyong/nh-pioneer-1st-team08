"""가설 5(문서 7단계): 아시아 AI 종목 정보로 다음 세션의 큰 움직임을 예측한다."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from second_tier_delay import COUNTRY, FIG, RES, load, stock_returns

END = pd.Timestamp("2026-11-11")
INIT, STEP, B, BLOCK = 504, 21, 10_000, 10
rng = np.random.default_rng(20261119)

BASE = ["log_m20", "log_last"]
M1 = ["disp", "breadth_dev", "ctry_gap", "ai_abs"]
HMM_IN = ["ai_mean", "disp"]


def asia_features():
    cals = {c: load(g["cal"]).index for c, g in COUNTRY.items()}
    cals = {c: x[x <= END] for c, x in cals.items()}
    common = cals["KR"].intersection(cals["JP"]).intersection(cals["TW"])
    R, ctry = {}, {}
    for c, g in COUNTRY.items():
        cr = {t: stock_returns(t, cals[c]).reindex(common) for t in g["sec"] + g["large"]}
        R.update(cr)
        ctry[c] = pd.DataFrame(cr).mean(axis=1)
    R = pd.DataFrame(R)
    cnt = R.notna().sum(axis=1)
    f = pd.DataFrame(index=common)
    f["disp"] = R.std(axis=1)
    f["breadth_dev"] = ((R > 0).sum(axis=1) / cnt - 0.5).abs()
    f["ctry_gap"] = pd.DataFrame(ctry).max(axis=1) - pd.DataFrame(ctry).min(axis=1)
    f["ai_mean"] = R.mean(axis=1)
    f["ai_abs"] = f["ai_mean"].abs()
    f = f[cnt >= 10]
    raw = R.loc[f.index].fillna(0.0)
    raw.columns = [f"r_{c}" for c in raw.columns]
    return pd.concat([f, raw], axis=1).iloc[1:]  # 첫날은 수익률 없음


def series_feats(s):
    """대상 값 s의 날짜마다 라벨 기준값(직전 60개의 70% 분위)과 기본 특징(직전 값)을 구한다."""
    p = s.shift(1)
    return pd.DataFrame({
        "v": s,
        "thr": p.rolling(60).quantile(0.7),
        "log_m20": np.log(p.rolling(20).mean()),
        "log_last": np.log(p + 1e-4),
    })


def build(target):
    F = asia_features()
    if target == "A_QQQ장중":
        q = load("QQQ")
        s = np.log(q["Close"] / q["Open"]).abs()
        sf = series_feats(s[s.index <= END])
        tdate = pd.Series(F.index, index=F.index)  # 같은 날짜의 미국 정규장
        ok = tdate.isin(sf.index)
    else:
        tw = load("^TWII")["Close"]
        tw = tw[tw.index <= END]
        s = np.log(tw).diff().abs()
        sf = series_feats(s)
        idx = sf.index
        pos = idx.searchsorted(F.index, side="right")
        tdate = pd.Series([idx[p] if p < len(idx) else pd.NaT for p in pos], index=F.index)
        ok = tdate.notna()
    F, tdate = F[ok], tdate[ok]
    T = sf.reindex(tdate.values)
    T.index = F.index
    D = pd.concat([F, T], axis=1).dropna()
    D["y"] = (D["v"] > D["thr"]).astype(int)
    D["target_date"] = tdate.reindex(D.index)
    return D


def _emis(X, mu, var):
    ll = -0.5 * (((X[:, None, :] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var[None])).sum(-1)
    m = ll.max(axis=1, keepdims=True)
    return np.exp(ll - m)  # 행마다 상수배는 정규화에서 상쇄됨


def _forward(Bm, pi, A):
    T, K = Bm.shape
    al, c = np.zeros((T, K)), np.zeros(T)
    a = pi * Bm[0]
    c[0] = a.sum(); al[0] = a / c[0]
    for t in range(1, T):
        a = (al[t - 1] @ A) * Bm[t]
        c[t] = a.sum(); al[t] = a / c[t]
    return al, c


def fit_hmm(X, K=3, iters=100):
    """대각 공분산을 쓰는 3상태 가우시안 HMM을 EM으로 추정한다."""
    T, d = X.shape
    order = np.argsort(np.abs(X).sum(axis=1))
    groups = np.array_split(order, K)
    mu = np.array([X[g].mean(0) for g in groups])
    var = np.array([X[g].var(0) + 1e-3 for g in groups])
    pi, A = np.full(K, 1 / K), np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * (0.9 - 0.1 / (K - 1))
    for _ in range(iters):
        Bm = _emis(X, mu, var)
        al, c = _forward(Bm, pi, A)
        be = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (Bm[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        g /= g.sum(1, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(T - 1):
            x = al[t][:, None] * A * (Bm[t + 1] * be[t + 1])[None] / c[t + 1]
            xi += x / x.sum()
        pi = g[0]
        A = xi / xi.sum(1, keepdims=True)
        w = g.sum(0)
        mu = (g.T @ X) / w[:, None]
        var = np.maximum((g.T @ X ** 2) / w[:, None] - mu ** 2, 1e-3)
    return pi, A, mu, var


def hmm_filtered(Xtr, Xall):
    """학습 구간에서 HMM을 추정하고, 각 시점까지의 정보만 반영한 필터링 확률을 계산한다."""
    m, s = Xtr.mean(0), Xtr.std(0)
    pi, A, mu, var = fit_hmm((Xtr - m) / s)
    al, _ = _forward(_emis((Xall - m) / s, mu, var), pi, A)
    order = np.argsort(var[:, 0])  # 재학습마다 상태 번호가 바뀌지 않도록 평균 수익률 분산 순으로 정렬한다
    return al[:, order][:, 1:]     # 세 확률의 합이 1이라 가장 조용한 상태를 기준으로 두고 나머지 두 개만 쓴다


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def walk(D):
    """21일 단위 확장형 워크포워드를 수행한다. 모든 모델(로지스틱, HMM, Isolation Forest)은 각 시험 구간 이전 자료로만 추정한다."""
    raw_cols = [c for c in D.columns if c.startswith("r_")]
    y = D["y"].values
    n = len(D)
    P = {k: np.full(n, np.nan) for k in ["base", "m1", "m2_hmm", "m3_iforest", "tree"]}
    for i in range(INIT, n, STEP):
        tr, te = slice(0, i), slice(i, min(i + STEP, n))
        Xb = D[BASE].values
        P["base"][te] = logit().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]

        X1 = D[BASE + M1].values
        P["m1"][te] = logit().fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0)
        P["tree"][te] = tree.fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]

        Xh = D[HMM_IN].values
        filt = hmm_filtered(Xh[tr], Xh[: te.stop])
        X2 = np.column_stack([Xb[: te.stop], filt])
        P["m2_hmm"][te] = logit().fit(X2[tr], y[tr]).predict_proba(X2[te])[:, 1]

        Xr = D[raw_cols].values
        iso = IsolationForest(n_estimators=200, random_state=0).fit(Xr[tr])
        sc = -iso.score_samples(Xr[: te.stop])
        X3 = np.column_stack([Xb[: te.stop], sc])
        P["m3_iforest"][te] = logit().fit(X3[tr], y[tr]).predict_proba(X3[te])[:, 1]
    out = pd.DataFrame(P, index=D.index)
    out["y"] = y
    return out.iloc[INIT:]


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_p(delta):
    """평균 개선이 0이라는 귀무가설 아래, 중심화한 손실 차이를 원형 블록으로 재표집해 단측 p값을 구한다."""
    delta = np.asarray(delta)
    n, obs = len(delta), delta.mean()
    e = delta - obs
    nb = int(np.ceil(n / BLOCK))
    idx = ((rng.integers(0, n, (B, nb))[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]) % n
    return (1 + (e[idx].mean(1) >= obs).sum()) / (1 + B)


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    rows, preds, side = [], [], []
    for target in ["A_QQQ장중", "B_대만다음날"]:
        D = build(target)
        W = walk(D)
        W["target"] = target
        preds.append(W)
        y = W["y"].values
        Lb = logloss(W["base"].values, y)
        clim = logloss(np.full(len(y), 0.3), y).mean()
        side.append(dict(대상=target, 시험일=len(y), 기간=f"{W.index[0].date()}~{W.index[-1].date()}",
                         큰움직임비율=y.mean(), 기본_logloss=Lb.mean(), 상수30_logloss=clim,
                         기본_AUC=roc_auc_score(y, W["base"])))
        for m in ["m1", "m2_hmm", "m3_iforest", "tree"]:
            Lm = logloss(W[m].values, y)
            d = Lb - Lm
            bs = (W["base"] - y) ** 2 - (W[m] - y) ** 2
            rows.append(dict(대상=target, 방법=m, 판정대상=m != "tree", 평균Δlogloss=d.mean(),
                             p_단측=boot_p(d), 평균ΔBrier=bs.mean(), AUC=roc_auc_score(y, W[m]),
                             기본AUC=roc_auc_score(y, W["base"])))
    R = pd.DataFrame(rows)
    main_ = R["판정대상"]
    R.loc[main_, "BH_q"] = bh(R.loc[main_, "p_단측"])
    R["통과"] = main_ & (R["평균Δlogloss"] > 0) & (R["BH_q"] < 0.10)
    R.to_csv(RES / "7_vol_ml_tests.csv", index=False, encoding="utf-8-sig")
    S = pd.DataFrame(side)
    S.to_csv(RES / "7_vol_ml_baseline.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds).to_csv(RES / "7_vol_ml_preds.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    lab = [f"{t[:1]}-{m}" for t, m in zip(R["대상"], R["방법"])]
    ax.barh(lab, R["평균Δlogloss"], color=["#2a4d8f" if k else "#9aa7bf" for k in R["판정대상"]])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("mean log-loss improvement vs baseline (>0 = better)")
    ax.set_title("A = QQQ US session, B = TAIEX next day")
    fig.tight_layout()
    fig.savefig(FIG / "7_vol_ml.png", dpi=150)


if __name__ == "__main__":
    main()
