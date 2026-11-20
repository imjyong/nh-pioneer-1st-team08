"""가설 6(문서 8단계): 아시아 AI 종목 변동폭이 그날 밤 QQQ 변동폭 예측을 개선하는지 검정한다."""
import numpy as np
import pandas as pd

from second_tier_delay import COUNTRY, RES, load
from vol_ml import END, INIT, STEP

B, BLOCK = 10_000, 10
rng = np.random.default_rng(20261120)
HAR = ["har1", "har5", "har22"]


def rng_series(ticker, cal):
    df = load(ticker).reindex(cal)
    r = np.log(df["High"] / df["Low"])
    ok = (r > 0) & ((df["Volume"] > 0) | ticker.startswith("^"))
    return r.where(ok)


def build():
    cals = {c: load(g["cal"]).index for c, g in COUNTRY.items()}
    cals = {c: x[x <= END] for c, x in cals.items()}
    common = cals["KR"].intersection(cals["JP"]).intersection(cals["TW"])
    R, grp = {}, {}
    for c, g in COUNTRY.items():
        for t in g["sec"] + g["large"]:
            R[t] = rng_series(t, cals[c]).reindex(common)
            grp[t] = "large" if t in g["large"] else "sec"
    R = pd.DataFrame(R)
    cnt = R.notna().sum(axis=1)
    A = pd.DataFrame(index=common)
    A["asia"] = np.log(R.mean(axis=1)).where(cnt >= 10)
    A["asia_large"] = np.log(R[[t for t in R if grp[t] == "large"]].mean(axis=1))
    A["asia_sec"] = np.log(R[[t for t in R if grp[t] == "sec"]].mean(axis=1))
    idx = {}
    for t, c in [("^KS11", "KR"), ("^N225", "JP"), ("^TWII", "TW")]:
        idx[t] = rng_series(t, cals[c]).reindex(common)
    A["asia_index"] = np.log(pd.DataFrame(idx).mean(axis=1))

    q = load("QQQ")
    q = q[q.index <= END]
    s = np.log(q["High"] / q["Low"])
    s = s.where(s > 0)
    p = s.shift(1)
    U = pd.DataFrame({"y": np.log(s), "har1": np.log(p), "har5": np.log(p.rolling(5).mean()),
                      "har22": np.log(p.rolling(22).mean()), "ret": np.log(q["Close"] / q["Open"])})
    D = A.join(U, how="inner").dropna(subset=["asia", "y"] + HAR)
    return D


def walk(D, extra):
    n = len(D)
    pred = {k: np.full(n, np.nan) for k in ["base", "full"]}
    for i in range(INIT, n, STEP):
        tr, te = slice(0, i), slice(i, min(i + STEP, n))
        for k, cols in [("base", HAR), ("full", HAR + extra)]:
            sub = D.iloc[tr].dropna(subset=cols)
            X = np.column_stack([np.ones(len(sub)), sub[cols].values])
            b = np.linalg.lstsq(X, sub["y"].values, rcond=None)[0]
            Xt = np.column_stack([np.ones(te.stop - te.start), D[cols].iloc[te].values])
            pred[k][te] = Xt @ b
    P = pd.DataFrame(pred, index=D.index).iloc[INIT:]
    P["y"] = D["y"].iloc[INIT:]
    return P.dropna(), b


def boot_p(delta):
    """평균 MSE 개선이 0이라는 귀무가설 아래, 중심화한 손실 차이를 원형 블록으로 재표집해 단측 p값을 구한다."""
    n, obs = len(delta), delta.mean()
    e = delta - obs
    nb = int(np.ceil(n / BLOCK))
    idx = ((rng.integers(0, n, (B, nb))[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]) % n
    return (1 + (e[idx].mean(1) >= obs).sum()) / (1 + B)


def evaluate(D, extra):
    P, b = walk(D, extra)
    eb, ef = (P["y"] - P["base"]) ** 2, (P["y"] - P["full"]) ** 2
    d = (eb - ef).values
    r2 = 1 - ef.sum() / eb.sum()
    return dict(특징="+".join(extra), 시험일=len(P), 기간=f"{P.index[0].date()}~{P.index[-1].date()}",
                기본MSE=eb.mean(), 비교MSE=ef.mean(), 평균Δ=d.mean(), 기본대비R2=r2,
                p_단측=boot_p(d), 마지막c=b[-1]), P


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


def mdd(x):
    e = np.cumsum(x)
    return (e - np.maximum.accumulate(e)).min()


def main():
    D = build()
    rows, P_main = [], None
    for extra, kind in [(["asia"], "본선"), (["asia_large"], "보조"), (["asia_sec"], "보조"), (["asia_index"], "보조")]:
        res, P = evaluate(D, extra)
        res["구분"] = kind
        rows.append(res)
        if kind == "본선":
            P_main = P
    R = pd.DataFrame(rows)
    R["통과"] = (R["구분"] == "본선") & (R["평균Δ"] > 0) & (R["p_단측"] < 0.05)
    R.to_csv(RES / "8_vol_spill_tests.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 매매 연결 (보조): 예측 변동폭 역비중
    ret = D["ret"].reindex(P_main.index).values
    out = [dict(규칙="매일 비중 1", **_perf(ret))]
    for k, name in [("base", "HAR 예측 역비중"), ("full", "HAR+아시아 예측 역비중")]:
        w = np.empty(len(P_main))
        pv = np.exp(P_main[k].values)
        # 학습 구간 예측 중앙값 대신, 각 시점까지의 예측 중앙값(과거만) 사용
        for i in range(len(pv)):
            med = np.median(pv[: i + 1]) if i > 0 else pv[0]
            w[i] = np.clip(med / pv[i], 0.5, 1.5)
        out.append(dict(규칙=name, 평균비중=w.mean(), **_perf(ret * w)))
    T = pd.DataFrame(out)
    T.to_csv(RES / "8_vol_spill_trading.csv", index=False, encoding="utf-8-sig")
    print()
    print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _perf(x):
    return dict(샤프=sharpe(x), 연변동성=x.std() * np.sqrt(252), MDD=mdd(x),
                하방편차=np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(252))


if __name__ == "__main__":
    main()
