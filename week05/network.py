"""공급망 지도 전략: 5단계_공급망지도_사전고정_1118.md의 검정 1·2와 민감도 분석."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2] / "data" / "market" / "week05_yahoo"
OUT, FIG = HERE / "results", HERE / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

END = pd.Timestamp("2026-11-11")
FIRST = {"KR": "09:00", "JP": "09:00", "TW": "10:00"}
MKT = {"KR": "^KS11", "JP": "1306.T", "TW": "^TWII"}
TW_BASKET = ["2454.TW", "2317.TW", "2382.TW", "3231.TW", "2408.TW"]  # 사전 고정의 대만 비교 바스켓
CUSTOMERS = ["2330.TW", "2303.TW", "3711.TW"]
N_PERM, SEED = 10000, 0

nodes = pd.read_csv(HERE / "map" / "nodes.csv", dtype=str).fillna("")
edges = pd.read_csv(HERE / "map" / "edges.csv", dtype=str)
edges["on"] = pd.to_datetime(edges["공개일"]) + pd.Timedelta(days=1)  # 공개 다음 날부터 (주말이면 다음 거래일에 자연히 켜짐)
NAME = dict(zip(nodes["ticker"], nodes["이름"]))


def first_hour(ticker, market):
    """야후 60분봉의 그날 첫 봉 시가·종가와 전날 마지막 봉 종가."""
    df = pd.read_csv(RAW / f"{ticker.replace('^', '_')}_60m_adj.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Seoul")
    df["date"] = df.index.normalize().tz_localize(None)
    last_close = df.groupby("date")["Close"].last()
    first = df[df.index.strftime("%H:%M") == FIRST[market]].set_index("date")
    out = pd.DataFrame({"r": np.log(first["Close"] / first["Open"]),
                        "g": np.log(first["Open"] / last_close.shift(1).reindex(first.index))})
    return out[out.index <= END].dropna()


RAWFH = {}


def fh(ticker, market):
    if ticker not in RAWFH:
        RAWFH[ticker] = first_hour(ticker, market)
    return RAWFH[ticker]


def market_frame(market, tw_mode, exclude=None):
    if market == "TW" and tw_mode == "바스켓":
        mem = [t for t in TW_BASKET if t != exclude]
        return pd.concat([fh(t, "TW") for t in mem], axis=1, keys=mem).T.groupby(level=1).mean().T
    return fh(MKT[market], market)


def adjusted(ticker, market, tw_mode="지수"):
    m = market_frame(market, tw_mode, exclude=ticker)
    a = fh(ticker, market).join(m, rsuffix="_m", how="inner")
    return pd.DataFrame({"y": a["r"] - a["r_m"], "gap": a["g"] - a["g_m"]})


def fe_fit(df, ycol, xcols, fe):
    d = df[[ycol] + xcols].sub(df.groupby(fe)[[ycol] + xcols].transform("mean"))
    return np.linalg.lstsq(d[xcols].values, d[ycol].values, rcond=None)[0]


def fe_cluster(df, ycol, xcols, fe):
    X = sm.add_constant(pd.get_dummies(df[fe], drop_first=True, dtype=float).join(df[xcols]))
    return sm.OLS(df[ycol], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["date"])[0]})


# 검정 1: 엔비디아 밤 수익률과 다음 날 아시아 아침
def test1(tw_mode="지수"):
    nv = pd.read_csv(RAW / "NVDA_1d_adj.csv", index_col=0, parse_dates=True)["Close"]
    nv_r = np.log(nv).diff().dropna()
    nv_r = nv_r[nv_r.index <= END]
    g1 = nodes[nodes["고리1_업종"] != ""]
    on = {}
    for _, e in edges[edges["고리"] == "1"].iterrows():
        firm = e["to"] if e["from"] == "NVDA" else e["from"]
        on[firm] = min(on.get(firm, e["on"]), e["on"])
    rows = []
    for _, n in g1.iterrows():
        a = adjusted(n["ticker"], n["시장"], tw_mode)
        a["firm"], a["group"] = n["ticker"], n["고리1_업종"]
        rows.append(a)
    df = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    prev = nv_r.index.searchsorted(df["date"].values) - 1
    df["N"] = np.where(prev >= 0, nv_r.values[np.clip(prev, 0, None)], np.nan)
    df = df.dropna().reset_index(drop=True)
    df["N"] = df["N"] / nv_r.std()  # 분석 표본(11/11까지)의 표준편차. 단위만 바꿈

    firms = df["firm"].values
    dvals = df["date"].values.astype("datetime64[ns]")
    never = np.datetime64("2100-01-01")

    def make_c(assign):
        act = np.array([np.datetime64(assign[f]) if f in assign else never for f in firms], dtype="datetime64[ns]")
        return (dvals >= act).astype(float)

    df["NC"] = df["N"] * make_c(on)
    fit = fe_cluster(df, "y", ["N", "NC", "gap"], "firm")
    b, c = fe_fit(df, "y", ["N", "NC", "gap"], "firm")[:2]

    rng = np.random.default_rng(SEED)
    groups = {g: list(sub["ticker"]) for g, sub in g1.groupby("고리1_업종")}
    perm = []
    for _ in range(N_PERM):
        assign = {}
        for g, members in groups.items():
            dates = [on.get(m) for m in members]
            for m, k in zip(members, rng.permutation(len(members))):
                if dates[k] is not None:
                    assign[m] = dates[k]
        df["NCp"] = df["N"] * make_c(assign)
        perm.append(fe_fit(df, "y", ["N", "NCp", "gap"], "firm")[1])
    perm = np.array(perm)
    p = (1 + (np.abs(perm) >= abs(c) - 1e-12).sum()) / (1 + N_PERM)

    res = {"검정": "1 엔비디아→아시아 아침", "대만 시장 조정": tw_mode, "귀무분포": "업종 안 연결 라벨 이동",
           "계수": c, "날짜군집 p": fit.pvalues["NC"], "순열 p": p, "가짜 지도 수": N_PERM,
           "b(비연결 기준)": b, "b+c(연결)": b + c, "관측": len(df), "날짜": df["date"].nunique()}
    by_firm = df.groupby("firm")[["N", "y"]].apply(lambda g: np.polyfit(g["N"], g["y"], 1)[0]).rename("엔비디아 반응")
    return res, perm, by_firm


# 검정 2: 한국·일본 공급사와 대만 고객
def test2(tw_mode="지수"):
    sup = nodes[(nodes["고리2_업종"] != "") & (nodes["고리2_업종"] != "고객")]
    x = pd.DataFrame({t: adjusted(t, m)["y"] for t, m in zip(sup["ticker"], sup["시장"])})
    cust = {c: adjusted(c, "TW", tw_mode) for c in CUSTOMERS}
    group_of = dict(zip(sup["ticker"], sup["고리2_업종"]))
    groups = sorted(set(group_of.values()))
    gmem = {g: [s for s in group_of if group_of[s] == g] for g in groups}
    e2 = edges[(edges["고리"] == "2") & edges["to"].isin(CUSTOMERS)]
    base = [(r["from"], r["to"], r["on"]) for _, r in e2.iterrows()]
    dates = x.index.intersection(cust[CUSTOMERS[0]].index)
    dv = dates.values.astype("datetime64[ns]")
    xv = {g: x.reindex(dates)[gmem[g]].values for g in groups}
    obs = {g: ~np.isnan(xv[g]) for g in groups}  # 그날 값이 있는 공급사만 분자·분모에 넣는다
    cust_df = {c: cust[c].reindex(dates) for c in CUSTOMERS}

    def signal(edge_list):
        rows = []
        for c in CUSTOMERS:
            per_group = []
            for g in groups:
                mem = gmem[g]
                act = np.zeros((len(dv), len(mem)))
                for s, cc, d in edge_list:
                    if cc == c and s in mem:
                        act[:, mem.index(s)] = dv >= np.datetime64(d)
                on_m, off_m = act * obs[g], (1 - act) * obs[g]
                n_on, n_off = on_m.sum(1), off_m.sum(1)
                xg = np.nan_to_num(xv[g])
                with np.errstate(invalid="ignore", divide="ignore"):
                    sg = (xg * on_m).sum(1) / n_on - (xg * off_m).sum(1) / n_off
                sg[(n_on == 0) | (n_off == 0)] = np.nan
                per_group.append(sg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                S = np.nanmean(np.vstack(per_group), 0)
            rows.append(cust_df[c].assign(S=S, cust=c))
        return pd.concat(rows).reset_index().rename(columns={"index": "date"}).dropna()

    df = signal(base)
    fit = fe_cluster(df, "y", ["S", "gap"], "cust")
    beta = fe_fit(df, "y", ["S", "gap"], "cust")[0]

    # (가) 사전 고정 귀무분포: 업종 안 연결 수 유지 재배선. 만들 수 있는 지도를 모두 모아 한 번씩만 센다
    rng = np.random.default_rng(SEED)
    graphs, cur = {}, list(base)
    for _ in range(20000):
        i, j = rng.choice(len(cur), 2, replace=False)
        (s1, c1, d1), (s2, c2, d2) = cur[i], cur[j]
        if s1 != s2 and c1 != c2 and group_of[s1] == group_of[s2]:
            pairs = {(s, c) for s, c, _ in cur}
            if (s1, c2) not in pairs and (s2, c1) not in pairs:
                cur[i], cur[j] = (s1, c2, d1), (s2, c1, d2)
        key = frozenset((s, c) for s, c, _ in cur)
        graphs.setdefault(key, list(cur))
    rew = np.array([fe_fit(signal(g), "y", ["S", "gap"], "cust")[0] for g in graphs.values()])
    p_rew = (np.abs(rew) >= abs(beta) - 1e-12).mean()

    # (나) 민감도: 업종 안에서 공급사 라벨(그 공급사의 연결 전체)을 통째로 바꾼다
    perm = []
    for _ in range(N_PERM):
        relabel = {}
        for g in groups:
            mem = gmem[g]
            for a, b in zip(mem, rng.permutation(mem)):
                relabel[a] = b
        perm.append(fe_fit(signal([(relabel[s], c, d) for s, c, d in base]), "y", ["S", "gap"], "cust")[0])
    perm = np.array(perm)
    p_lab = (1 + (np.abs(perm) >= abs(beta) - 1e-12).sum()) / (1 + N_PERM)

    heat = pd.DataFrame(index=sup["ticker"], columns=CUSTOMERS, dtype=float)
    for s in sup["ticker"]:
        for c in CUSTOMERS:
            j = pd.concat([x[s], cust[c]["y"]], axis=1, join="inner").dropna()
            heat.loc[s, c] = np.polyfit(j.iloc[:, 0], j.iloc[:, 1], 1)[0]
    common = {"검정": "2 공급사→대만 고객", "대만 시장 조정": tw_mode, "계수": beta, "날짜군집 p": fit.pvalues["S"],
              "관측": len(df), "날짜": df["date"].nunique()}
    res = [{**common, "귀무분포": "재배선(가능한 지도 전부)", "순열 p": p_rew, "가짜 지도 수": len(graphs)},
           {**common, "귀무분포": "공급사 라벨 순열(민감도)", "순열 p": p_lab, "가짜 지도 수": N_PERM}]
    return res, perm, heat, base


def plots(perm1, r1, perm2, r2, heat, base):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, perm, r, t in [(ax[0], perm1, r1, "검정 1: 엔비디아 → 아시아 아침 (연결 라벨 이동)"),
                          (ax[1], perm2, r2, "검정 2: 공급사 → 대만 고객 (공급사 라벨 순열)")]:
        a.hist(perm, bins=60, color="#bbb")
        a.axvline(r["계수"], color="#1f2a5a", lw=2, label=f"진짜 지도 {r['계수']:.4f} (p={r['순열 p']:.3f})")
        a.set_title(t, fontsize=10)
        a.set_xlabel(f"가짜 지도 {len(perm):,}개의 계수")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "5_permutation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 7))
    v = np.nanmax(np.abs(heat.values.astype(float)))
    ax.imshow(heat.values.astype(float), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(3), [NAME[c] for c in heat.columns])
    ax.set_yticks(range(len(heat)), [NAME[s] for s in heat.index], fontsize=8)
    for i, s in enumerate(heat.index):
        for j, c in enumerate(heat.columns):
            ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
            if any(s == es and c == ec for es, ec, _ in base):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.8))
    ax.set_title("공급사 09~10시 → 고객 10~11시 단순 기울기 (기술통계)\n검은 테두리 = 공개된 공급 관계", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "5_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = []
    r1, perm1, by_firm = test1("지수")
    r2s, perm2, heat, base = test2("지수")
    out += [r1] + r2s
    plots(perm1, r1, perm2, r2s[1], heat, base)
    heat.to_csv(OUT / "5_heatmap.csv")
    by_firm.to_frame().join(nodes.set_index("ticker")[["이름", "고리1_업종"]]).to_csv(OUT / "5_test1_by_firm.csv")
    r1b, _, _ = test1("바스켓")
    r2b, _, _, _ = test2("바스켓")
    out += [r1b] + r2b
    tab = pd.DataFrame(out)
    main = tab[(tab["대만 시장 조정"] == "지수") & (tab["귀무분포"] != "공급사 라벨 순열(민감도)")]
    tab["BH q(본선 2개)"] = np.nan
    tab.loc[main.index, "BH q(본선 2개)"] = multipletests(main["순열 p"], method="fdr_bh")[1]
    tab.to_csv(OUT / "5_network_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
