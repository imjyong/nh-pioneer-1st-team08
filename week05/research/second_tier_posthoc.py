"""가설 4 사후 분석(문서 6단계): 아시아 대형 AI주의 D일 수익률과 2차 수혜 기업의 D+1~D+3 반응을 분석한다. 본선 판정에는 쓰지 않는다."""
import numpy as np
import pandas as pd

from second_tier_delay import COUNTRY, END, MKT, RES, START, load, stock_returns

B, BLOCK = 10_000, 10
rng = np.random.default_rng(20261119)


def country_frame(c):
    g = COUNTRY[c]
    cal = load(g["cal"]).index
    cal = cal[(cal >= START - pd.Timedelta(days=10)) & (cal <= END)]
    adj = {t: stock_returns(t, cal) - stock_returns(MKT[t], cal) for t in g["sec"] + g["large"]}
    A = pd.DataFrame(adj)
    S, L = A[g["sec"]].mean(axis=1), A[g["large"]].mean(axis=1)
    Y = S.shift(-1) + S.shift(-2) + S.shift(-3)  # 현지 거래일 기준 D+1~D+3이며, 하나라도 결측이면 NaN으로 둔다
    f = pd.DataFrame({"L": L, "S": S, "Y": Y})
    return f[(f.index >= START)].dropna()


def gamma(F):
    X = np.column_stack([np.ones(len(F)), F["L"], F["S"]])
    return np.linalg.lstsq(X, F["Y"].values, rcond=None)[0][1]


def main():
    frames = {c: country_frame(c) for c in COUNTRY}
    common = frames["KR"].index.intersection(frames["JP"].index).intersection(frames["TW"].index)
    frames = {c: f.loc[common] for c, f in frames.items()}
    n = len(common)
    g_obs = {c: gamma(f) for c, f in frames.items()}
    g_bar = np.mean(list(g_obs.values()))

    # 귀무가설 γ=0을 강제한 제한모형 Y = α + δS의 잔차로 부트스트랩 표본을 만든다
    fit, res = {}, {}
    for c, f in frames.items():
        X0 = np.column_stack([np.ones(n), f["S"]])
        b0 = np.linalg.lstsq(X0, f["Y"].values, rcond=None)[0]
        fit[c], res[c] = X0 @ b0, f["Y"].values - X0 @ b0

    nb = int(np.ceil(n / BLOCK))
    count = 0
    pinv = {c: np.linalg.pinv(np.column_stack([np.ones(n), f["L"], f["S"]])) for c, f in frames.items()}
    for _ in range(B):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(BLOCK)).ravel()[:n]) % n
        gs = [(pinv[c] @ (fit[c] + res[c][idx]))[1] for c in frames]
        count += np.mean(gs) >= g_bar
    p = (1 + count) / (1 + B)

    sd = {c: frames[c]["L"].std() for c in frames}
    out = pd.DataFrame([dict(나라=c, n=n, gamma=g_obs[c], L_표준편차=sd[c]) for c in frames]
                       + [dict(나라="평균", n=n, gamma=g_bar, 귀무부트스트랩_단측p=p)])
    out.to_csv(RES / "h4_posthoc_large_to_second.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"기간 {common[0].date()} ~ {common[-1].date()}")


if __name__ == "__main__":
    main()
