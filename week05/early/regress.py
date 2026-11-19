"""3단계: 사전 고정(3단계_사전고정_1117.md, 수정 절 포함) 그대로 회귀·순열 검정·워크포워드를 돌린다."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "market" / "week05_yahoo"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

CUTOFF = "2026-11-16"
HOLDOUT_FROM = "2026-11-12"  # 젠슨 황 사임 전후로 넉넉히 잡아 학습·검증에서 빼고 사례로만 사용 (발표일 확인 전)
ASIA = {"^TWII": "대만", "^N225": "닛케이", "^HSI": "항셍", "^HSCE": "H주"}
TARGETS = {"NVDA": "엔비디아", "QQQ": "QQQ"}
N_PERM, MIN_SHIFT, SEED = 1000, 20, 0


def load(t, cols=("Open", "Close")):
    df = pd.read_csv(RAW / f"{t.replace('^', '_')}_1d_adj.csv", index_col=0, parse_dates=True)
    return df.loc[:CUTOFF, list(cols)].dropna()


def asia_returns():
    px = {n: load(t) for t, n in ASIA.items()}
    kinds = {}
    for n, df in px.items():
        o, c = np.log(df["Open"]), np.log(df["Close"])
        kinds.setdefault("장중", {})[n] = c - o
        kinds.setdefault("갭", {})[n] = o - c.shift()
        kinds.setdefault("종가→종가", {})[n] = c.diff()
    kinds = {k: pd.DataFrame(v) for k, v in kinds.items()}
    common = kinds["장중"].dropna().index
    for k in kinds:
        common = common.intersection(kinds[k].dropna().index)
    return {k: v.loc[common] for k, v in kinds.items()}


def features(r):
    """z = 수익률 ÷ 직전 60거래일 표준편차(당일 제외). 공통 관측일 순서로 계산."""
    z = r / r.rolling(60).std().shift(1)
    supply = z[["대만", "닛케이"]].mean(axis=1)
    hk = z[["항셍", "H주"]].mean(axis=1)
    f = pd.DataFrame({"갈림": supply - hk, "공통": (supply + hk) / 2})
    f["|갈림|"], f["|공통|"] = f["갈림"].abs(), f["공통"].abs()
    return f.dropna()


def us_frames():
    out = {}
    for t, n in TARGETS.items():
        df = load(t)
        out[n] = pd.DataFrame({"y": np.log(df["Close"] / df["Open"]), "cc": np.log(df["Close"]).diff()})
    return out


def build(feat, us):
    """아시아 D 행에 미국 D 목표와 미국 D-1 통제를 붙인다."""
    us_dates = us.dropna().index
    rows = []
    for d in feat.index:
        if d not in us_dates:
            continue
        prev = us_dates[us_dates < d]
        if len(prev) == 0:
            continue
        rows.append({"date": d, "y": us.at[d, "y"], "us_prev": us.at[prev[-1], "cc"]})
    df = pd.DataFrame(rows).set_index("date").join(feat, how="inner")
    df["|y|"], df["|us_prev|"] = df["y"].abs(), df["us_prev"].abs()
    return df


def ols_b1(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def run_test(df, problem):
    if problem == "수익률":
        ycol, xcols = "y", ["갈림", "us_prev", "공통"]
    else:
        ycol, xcols = "|y|", ["|갈림|", "|us_prev|", "|공통|"]
    X = sm.add_constant(df[xcols])
    m = sm.OLS(df[ycol], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    b1, p_hac = m.params[xcols[0]], m.pvalues[xcols[0]]

    # 순열: 아시아 특징 묶음 전체를 같이 원형 이동
    rng = np.random.default_rng(SEED)
    asia_block = df[xcols[0::2]].values  # 갈림·공통 (또는 절댓값)
    us_part = df[[xcols[1]]].values
    y = df[ycol].values
    n = len(df)
    hits = 0
    for _ in range(N_PERM):
        k = rng.integers(MIN_SHIFT, n - MIN_SHIFT)
        a = np.roll(asia_block, k, axis=0)
        Xp = np.column_stack([np.ones(n), a[:, 0], us_part[:, 0], a[:, 1]])
        hits += abs(ols_b1(y, Xp)) >= abs(b1)
    p_perm = (1 + hits) / (1 + N_PERM)

    # 워크포워드 (확장형)
    init, step = 504, 63
    err_with, err_without = [], []
    for start in range(init, n, step):
        tr, te = df.iloc[:start], df.iloc[start:start + step]
        for cols, store in [(xcols, err_with), (xcols[1:], err_without)]:
            mm = sm.OLS(tr[ycol], sm.add_constant(tr[cols])).fit()
            pred = mm.predict(sm.add_constant(te[cols], has_constant="add"))
            store.append(te[ycol].values - pred.values)
    ew, eo = np.concatenate(err_with), np.concatenate(err_without)
    if problem == "수익률":
        wf = 1 - (ew ** 2).sum() / (eo ** 2).sum()
        wf_name = "표본 외 R²"
    else:
        wf = np.abs(eo).mean() - np.abs(ew).mean()
        wf_name = "MAE 개선"
    return {"b1": b1, "HAC p": p_hac, "순열 p": p_perm, "지표": wf_name, "워크포워드": wf, "n": n, "시험 n": len(ew)}


def main():
    rets = asia_returns()
    us = us_frames()
    rows = []
    for kind, r in rets.items():
        feat = features(r)
        for name, u in us.items():
            df = build(feat, u)
            df = df[df.index < HOLDOUT_FROM]
            for problem in ["수익률", "변동성"]:
                res = run_test(df, problem)
                rows.append({"수익률 종류": kind, "대상": name, "문제": problem, **res})
    tab = pd.DataFrame(rows)
    tab["BH 유의(10%)"] = multipletests(tab["HAC p"], alpha=0.10, method="fdr_bh")[0]
    tab["BH 유의(10%, 순열 p)"] = multipletests(tab["순열 p"], alpha=0.10, method="fdr_bh")[0]
    tab.to_csv(OUT / "3_regression_daily.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
