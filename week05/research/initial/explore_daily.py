"""초기 탐색(문서 2단계): 야후 일봉을 한국 시간 순서로 맞추고 시장 간 상관과 계층적 군집을 계산한다.
군집에는 아시아 D와 그보다 먼저 끝난 미국 D-1 수익률을 쓴다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data" / "market" / "week05_yahoo"
FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

MAIN_ASIA = {"^TWII": "대만 가권", "^N225": "닛케이", "3033.HK": "항셍테크 ETF", "^HSI": "항셍", "^HSCE": "H주"}
AUX_ASIA = {"2330.TW": "TSMC", "6857.T": "어드밴테스트", "8035.T": "도쿄일렉트론", "4062.T": "이비덴",
            "9988.HK": "알리바바", "0700.HK": "텐센트", "0981.HK": "SMIC"}
US = {"NVDA": "엔비디아", "QQQ": "나스닥100 ETF"}
ALL = {**MAIN_ASIA, **AUX_ASIA, **US}


def load(ticker):
    path = RAW / f"{ticker.replace('^', '_')}_1d_adj.csv"
    if not path.exists():
        raise SystemExit(f"{path.name} 없음. 먼저 fetch_yahoo.py를 실행하세요")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "Close"]].dropna()


px = {t: load(t) for t in ALL}

# 아시아: 종가→종가 로그수익률 (각 거래소 자기 거래일 기준)
asia = pd.DataFrame({ALL[t]: np.log(px[t]["Close"]).diff() for t in {**MAIN_ASIA, **AUX_ASIA}})
# 미국: 종가→종가, 시가→종가
us_cc = pd.DataFrame({ALL[t]: np.log(px[t]["Close"]).diff() for t in US})
us_oc = pd.DataFrame({ALL[t] + " 시가→종가": np.log(px[t]["Close"] / px[t]["Open"]) for t in US})

asia_dates = asia.dropna(how="all").index
us_dates = us_cc.dropna(how="all").index

# 아시아 D에 붙일 미국 D-1: 아시아 D보다 앞선 마지막 미국 거래일
prev_us = pd.Series([us_dates[us_dates < d].max() if (us_dates < d).any() else pd.NaT for d in asia_dates], index=asia_dates)
same_us = pd.Series([d if d in us_dates else pd.NaT for d in asia_dates], index=asia_dates)

panel = asia.loc[asia_dates].copy()
for col in us_cc:
    panel[col + " (D-1)"] = us_cc[col].reindex(prev_us.values).values
for col in us_oc:
    panel[col + " (D)"] = us_oc[col].reindex(same_us.values).values

main_cols = list(MAIN_ASIA.values()) + ["엔비디아 (D-1)", "나스닥100 ETF (D-1)"]
aux_cols = main_cols + list(AUX_ASIA.values())


def cluster_plot(cols, name, title):
    c = panel[cols].corr(min_periods=200)
    d = np.sqrt(np.clip(2 * (1 - c.values), 0, None))
    np.fill_diagonal(d, 0)
    z = linkage(squareform(d, checks=False), method="average")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax[0].imshow(c.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[0].set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax[0].text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    dendrogram(z, labels=cols, orientation="right", ax=ax[1], color_threshold=0.9 * max(z[:, 2]))
    ax[1].set_xlabel("거리 = √(2(1 - 상관))")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    return c


c_main = cluster_plot(main_cols, "2_cluster_main.png", "같은 창 상관과 계층적 군집 (지수·ETF, 2021-11 ~ 2026-11 일봉)")
c_aux = cluster_plot(aux_cols, "2_cluster_aux.png", "같은 창 상관과 계층적 군집 (개별 종목 포함, 보조)")

print("관측일:", len(panel), panel.index[0].date(), "~", panel.index[-1].date())
print("\n[같은 창 상관, 지수·ETF]")
print(c_main.round(2).to_string())

# 연도별 안정성: 주요 쌍
pairs = [("대만 가권", "닛케이"), ("대만 가권", "항셍테크 ETF"), ("닛케이", "항셍테크 ETF"),
         ("항셍테크 ETF", "항셍"), ("대만 가권", "나스닥100 ETF (D-1)"), ("항셍테크 ETF", "나스닥100 ETF (D-1)")]
rows = {}
for y, g in panel.groupby(panel.index.year):
    rows[y] = {f"{a}-{b}": g[a].corr(g[b]) for a, b in pairs}
print("\n[연도별 같은 창 상관]")
print(pd.DataFrame(rows).round(2).to_string())

# 선행 비교: 아시아 D → 미국 D 시가→종가
lead_cols = list(MAIN_ASIA.values())
lead = panel[lead_cols + ["엔비디아 시가→종가 (D)", "나스닥100 ETF 시가→종가 (D)"]].corr().loc[lead_cols, ["엔비디아 시가→종가 (D)", "나스닥100 ETF 시가→종가 (D)"]]
print("\n[선행 상관: 아시아 D 수익률 → 미국 D 시가→종가]")
print(lead.round(3).to_string())

panel.to_csv(RAW / "panel_daily.csv")
