"""초기 탐색 보완(문서 2단계): 공통 관측일의 군집 안정성과 3033.HK 시가 품질을 점검한다.
날짜 iid와 5거래일 블록 부트스트랩을 비교하고, 수집 중이던 11/17은 제외한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data" / "market" / "week05_yahoo"
CUTOFF = "2026-11-16"  # 이 날짜까지만 사용 (11/17 아시아 장은 수집 시점에 진행 중)
ASIA = {"^TWII": "대만", "^N225": "닛케이", "3033.HK": "항셍테크", "^HSI": "항셍", "^HSCE": "H주"}
US = {"NVDA": "엔비디아", "QQQ": "QQQ"}


def load(t):
    df = pd.read_csv(RAW / f"{t.replace('^', '_')}_1d_adj.csv", index_col=0, parse_dates=True)
    return df.loc[:CUTOFF, ["Open", "Close"]].dropna()


px = {t: load(t) for t in {**ASIA, **US}}
cc, gap, intra = {}, {}, {}
for t, n in ASIA.items():
    o, c = np.log(px[t]["Open"]), np.log(px[t]["Close"])
    cc[n], gap[n], intra[n] = c.diff(), o - c.shift(), c - o
cc, gap, intra = pd.DataFrame(cc), pd.DataFrame(gap), pd.DataFrame(intra)
us_cc = pd.DataFrame({n: np.log(px[t]["Close"]).diff() for t, n in US.items()})

# 3033 시가 품질
h = px["3033.HK"]
flat_open = (h["Open"] == h["Close"].shift()).reindex(gap.index)
hsi_gap_big = gap["항셍"].abs() > 0.001
bad3033 = (flat_open & hsi_gap_big).fillna(False)
print(f"[3033 시가 점검] 시가 = 직전 종가인 날 {int(flat_open.sum())}일, 그중 항셍 갭 0.1% 초과 {int(bad3033.sum())}일")
for n in ["대만", "닛케이", "항셍", "H주"]:
    t = [k for k, v in ASIA.items() if v == n][0]
    f = (px[t]["Open"] == px[t]["Close"].shift()).sum()
    print(f"  {n}: 시가 = 직전 종가인 날 {int(f)}일")

asia_cols = list(ASIA.values())
common = cc[asia_cols].dropna().index
common = common.intersection(intra[asia_cols].dropna().index).intersection(gap[asia_cols].dropna().index)
print(f"\n다섯 지수 공통 관측일: {len(common)}일 ({common[0].date()} ~ {common[-1].date()})")

us_dates = us_cc.dropna(how="all").index
prev = [us_dates[us_dates < d].max() for d in common]
us_prev = us_cc.reindex(prev)
us_prev.index = common
us_prev.columns = [c + " D-1" for c in us_prev.columns]

print("\n[공통 관측일, 미국 D-1 QQQ와 상관]")
rows = {}
for name, df in [("종가→종가", cc), ("갭", gap), ("장중", intra)]:
    j = df.loc[common, asia_cols].join(us_prev)
    rows[name] = {n: j[n].corr(j["QQQ D-1"]) for n in asia_cols}
j = intra.loc[common, ["항셍테크"]].join(us_prev)[~bad3033.reindex(common).values]
rows["장중(3033 시가 이상일 제외)"] = {"항셍테크": j["항셍테크"].corr(j["QQQ D-1"])}
print(pd.DataFrame(rows).round(2).to_string())


def two_groups(df, cols):
    c = df[cols].corr()
    d = np.sqrt(np.clip(2 * (1 - c.values), 0, None))
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), 2, "maxclust")
    return frozenset(frozenset(cols[i] for i in range(len(cols)) if lab[i] == k) for k in set(lab))


def block_sample(n, block, rng):
    starts = rng.integers(0, n - block + 1, n // block + 1)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


target = frozenset([frozenset(["대만", "닛케이"]), frozenset(["항셍테크", "항셍", "H주"])])
target_us = frozenset([frozenset(["대만", "닛케이", "엔비디아 D-1", "QQQ D-1"]), frozenset(["항셍테크", "항셍", "H주"])])
rng = np.random.default_rng(0)
print("\n[2묶음 군집, 공통 관측일]")
for label, df, cols, tgt in [
    ("아시아만, 종가→종가", cc.loc[common], asia_cols, target),
    ("아시아만, 장중", intra.loc[common], asia_cols, target),
    ("아시아만, 장중, 3033 시가 이상일 제외", intra.loc[common][~bad3033.reindex(common).values], asia_cols, target),
    ("아시아만, 갭", gap.loc[common], asia_cols, target),
    ("아시아 + 미국 D-1, 종가→종가", cc.loc[common, asia_cols].join(us_prev).dropna(), asia_cols + ["엔비디아 D-1", "QQQ D-1"], target_us),
]:
    got = two_groups(df, cols)
    n = len(df)
    iid = sum(two_groups(df.iloc[rng.integers(0, n, n)], cols) == tgt for _ in range(1000))
    blk = sum(two_groups(df.iloc[block_sample(n, 5, rng)], cols) == tgt for _ in range(1000))
    groups = " / ".join("{" + ", ".join(sorted(g)) + "}" for g in got)
    print(f"{label} (n={n}): {groups} | 같은 2묶음 비율: 날짜 iid {iid / 10:.1f}%, 5일 블록 {blk / 10:.1f}%")

bad3033.to_frame("bad_open").to_csv(RAW / "3033_bad_open_days.csv")
