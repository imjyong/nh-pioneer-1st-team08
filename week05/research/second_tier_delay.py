"""가설 4: 엔비디아 고유 충격 뒤 AI 2차 수혜 기업이 대형주보다 늦게 따라가는지 검정한다.
다른 가설 코드가 쓰는 종목 목록, 데이터 읽기, 결과 경로도 여기 있다.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "market" / "week05_yahoo"
HERE = Path(__file__).resolve().parent
RES, FIG = HERE.parent / "results", HERE.parent / "figs"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

START, END = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-11-11")
B, BLOCK = 10_000, 10
rng = np.random.default_rng(20261118)

COUNTRY = {
    "KR": dict(cal="^KS11", sec=["083450.KQ", "007660.KS"], large=["000660.KS", "005930.KS"]),
    "JP": dict(cal="1306.T", sec=["6787.T", "5805.T"], large=["8035.T", "6857.T"]),
    "TW": dict(cal="^TWII", sec=["2421.TW", "6412.TW", "2368.TW", "6213.TW"], large=["2330.TW"]),
}
MKT = {"083450.KQ": "^KQ11", "007660.KS": "^KS11", "000660.KS": "^KS11", "005930.KS": "^KS11",
       "6787.T": "1306.T", "5805.T": "1306.T", "8035.T": "1306.T", "6857.T": "1306.T"}
for t in COUNTRY["TW"]["sec"] + COUNTRY["TW"]["large"]:
    MKT[t] = "^TWII"


def load(ticker):
    df = pd.read_csv(DATA / f"{ticker.replace('^', '_')}_1d_adj.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def stock_returns(ticker, cal):
    """현지 거래일 기준 일 로그수익률을 구한다. 거래량이 0인 날은 종가를 결측으로 둔다."""
    df = load(ticker).reindex(cal)
    close = df["Close"].where(df["Volume"] > 0) if not ticker.startswith("^") else df["Close"]
    return np.log(close).diff()


def rolling_beta(y, x, win=120):
    """t−win ~ t−1 자료로 β를 추정한다. 당일은 넣지 않고, 두 값이 모두 있는 날만 쓴다."""
    out = pd.Series(np.nan, index=y.index)
    both = pd.concat([y, x], axis=1).dropna()
    yy, xx = both.iloc[:, 0].values, both.iloc[:, 1].values
    for i in range(win, len(both)):
        xs, ys = xx[i - win:i], yy[i - win:i]
        out[both.index[i]] = np.cov(xs, ys)[0, 1] / np.var(xs, ddof=1)
    return out


def nvda_signal():
    nv, qq = load("NVDA")["Close"], load("QQQ")["Close"]
    r = pd.concat([np.log(nv).diff(), np.log(qq).diff()], axis=1, keys=["n", "q"]).dropna()
    beta = rolling_beta(r["n"], r["q"])
    return (r["n"] - beta * r["q"]).dropna()


def country_paths(c, sec, large, resid=False):
    """나라 c의 (2차 − 대형) 포트폴리오 일 수익률을 현지 거래일 기준으로 구한다."""
    cal = load(COUNTRY[c]["cal"]).index
    cal = cal[(cal >= START - pd.Timedelta(days=400)) & (cal <= END)]
    rets = {}
    for t in sec + large:
        r = stock_returns(t, cal)
        if resid:
            m = stock_returns(MKT[t], cal)
            r = r - rolling_beta(r, m) * m
        rets[t] = r
    R = pd.DataFrame(rets)
    return R[sec].mean(axis=1, skipna=True) - R[large].mean(axis=1, skipna=True)


def align(signal, diff, h0, h1):
    """미국 t 다음 첫 현지 거래일을 D로 두고, D+h0 ~ D+h1 현지 거래일 수익률을 누적한다."""
    cal = diff.index
    pos = cal.searchsorted(signal.index, side="right")  # t보다 엄격히 뒤의 첫 현지 거래일
    out = pd.Series(np.nan, index=signal.index)
    vals = diff.values
    for k, (t, p) in enumerate(zip(signal.index, pos)):
        a, b = p + h0, p + h1
        if b >= len(cal) or cal[p] < START or cal[b] > END:
            continue
        seg = vals[a:b + 1]
        if np.isnan(seg).any():
            continue
        out.iloc[k] = seg.sum()
    return out


def build_Y(h0, h1, resid=False, groups=None):
    sig = nvda_signal()
    per = {}
    for c, g in COUNTRY.items():
        sec, large = (groups[c] if groups else (g["sec"], g["large"]))
        per[c] = align(sig, country_paths(c, sec, large, resid), h0, h1)
    P = pd.DataFrame(per)
    ok = P.notna().all(axis=1)
    return sig[ok], P[ok].mean(axis=1), P[ok]


def ols_gamma(N, Y):
    x = N - N.mean()
    return (x * (Y - Y.mean())).sum() / (x ** 2).sum()


def null_bootstrap(N, Y, block=BLOCK, b=B):
    N, Y = np.asarray(N), np.asarray(Y)
    g_obs = ols_gamma(N, Y)
    e = Y - Y.mean()
    n = len(Y)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(b, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, -1)[:, :n] % n
    xs = N - N.mean()
    g_star = (e[idx] * xs[None, :]).sum(axis=1) / (xs ** 2).sum()
    return g_obs, (1 + (g_star >= g_obs).sum()) / (1 + b)


def nw_p_onesided(N, Y, lag=5):
    from scipy.stats import norm
    N, Y = np.asarray(N), np.asarray(Y)
    X = np.column_stack([np.ones_like(N), N])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    u = Y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (G + G.T)
    se = np.sqrt((XtX_inv @ S @ XtX_inv)[1, 1])
    return 1 - norm.cdf(beta[1] / se)


def main():
    rows = []

    def record(name, N, Y, block=BLOCK, kind="보조"):
        g, p = null_bootstrap(N, Y, block)
        rows.append(dict(분석=name, 구분=kind, n=len(Y), gamma=g, boot_p_단측=p,
                         블록=block, NW_p_단측=nw_p_onesided(N, Y)))

    N, Y, P = build_Y(1, 3)
    record("주 검정: D+1~D+3", N, Y, kind="본선")
    for blk in (5, 20):
        record(f"D+1~D+3 블록 {blk}일", N, Y, block=blk)
    for c in COUNTRY:
        record(f"D+1~D+3 {c}만", N, P[c])
    for name, (h0, h1) in {"D 당일": (0, 0), "D+1": (1, 1), "D+1~D+5": (1, 5)}.items():
        n2, y2, _ = build_Y(h0, h1)
        record(name, n2, y2)
    n3, y3, _ = build_Y(1, 3, resid=True)
    record("D+1~D+3 시장모형 잔차", n3, y3)

    # 같은 나라 안 라벨 순열 (가능한 조합 전부)
    pools = {c: g["sec"] + g["large"] for c, g in COUNTRY.items()}
    choices = {c: list(combinations(pools[c], len(g["sec"]))) for c, g in COUNTRY.items()}
    perm = []
    for kr in choices["KR"]:
        for jp in choices["JP"]:
            for tw in choices["TW"]:
                grp = {c: (list(s), [t for t in pools[c] if t not in s])
                       for c, s in zip(COUNTRY, (kr, jp, tw))}
                n4, y4, _ = build_Y(1, 3, groups=grp)
                perm.append(ols_gamma(n4, y4))
    g_obs = rows[0]["gamma"]
    perm = np.array(perm)
    rows.append(dict(분석="같은 나라 라벨 순열(180조합)", 구분="보조", n=len(Y), gamma=g_obs,
                     boot_p_단측=(perm >= g_obs - 1e-12).mean(), 블록=np.nan, NW_p_단측=np.nan))

    out = pd.DataFrame(rows)
    out.to_csv(RES / "h4_second_tier_tests.csv", index=False, encoding="utf-8-sig")
    pd.Series(perm).to_csv(RES / "h4_label_perm_gammas.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n표본 t: {N.index[0].date()} ~ {N.index[-1].date()}, N 표준편차 {N.std():.4f}")

    # 반응 경로: D+h 하루씩의 γ (보조, 그림용)
    path = []
    for h in range(0, 6):
        n5, y5, _ = build_Y(h, h)
        path.append(ols_gamma(n5, y5))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(range(6), path, color="#2a4d8f")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xticks(range(6), ["D"] + [f"D+{h}" for h in range(1, 6)])
    ax[0].set_title("날짜별 γ: 엔비디아 충격 뒤 (2차 수혜 - 대형)")
    ax[1].hist(perm, bins=30, color="#9aa7bf")
    ax[1].axvline(g_obs, color="#c0392b", lw=2, label=f"실제 γ {g_obs:.4f}")
    ax[1].set_title("같은 나라 라벨 순열 180조합, D+1~D+3")
    ax[1].set_xlabel("가짜 포트폴리오의 γ")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "h4_second_tier.png", dpi=150)
    pd.Series(path, index=["D"] + [f"D+{h}" for h in range(1, 6)]).to_csv(RES / "h4_gamma_path.csv")


if __name__ == "__main__":
    main()
