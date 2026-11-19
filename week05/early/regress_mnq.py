"""3단계 (거래 상품 기준): 장중 갈림 점수 → MNQ=F 한국 18:00 이후 수익률. 사전 고정 수정 절 그대로."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from regress import HOLDOUT_FROM, MIN_SHIFT, N_PERM, RAW, OUT, SEED, asia_returns, features, ols_b1

KST, NY = "Asia/Seoul", "America/New_York"

m = pd.read_csv(RAW / "MNQ=F_60m_adj.csv", index_col=0)
m.index = pd.to_datetime(m.index, utc=True)
m = m[m.index.minute == 0]
open_at = m["Open"]


def price_at(ts_utc):
    return open_at.get(ts_utc, np.nan)


daily = pd.read_csv(RAW / "MNQ=F_1d_adj.csv", index_col=0, parse_dates=True)
trade_days = daily.index


def roll_days():
    out = set()
    for y in range(2024, 2027):
        for mth in (3, 6, 9, 12):
            d = pd.Timestamp(y, mth, 15)
            tf = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
            pos = trade_days.searchsorted(tf)
            for k in range(pos - 1, pos + 3):
                if 0 <= k < len(trade_days):
                    out.add(trade_days[k].normalize())
    return out


ROLL = roll_days()

feat = features(asia_returns()["장중"])
rows = []
for d in feat.index:
    if d < pd.Timestamp("2024-06-27") or d >= pd.Timestamp(HOLDOUT_FROM):
        continue
    entry = pd.Timestamp(d.date()).tz_localize(KST).replace(hour=18).tz_convert("UTC")
    exit_exec = pd.Timestamp(d.date()).tz_localize(KST).replace(hour=0) + pd.Timedelta(days=1)
    exit_exec = exit_exec.tz_convert("UTC")
    ny_close = pd.Timestamp(d.date()).tz_localize(NY).replace(hour=16).tz_convert("UTC")
    prev_ny = pd.Timestamp((d - pd.Timedelta(days=1)).date()).tz_localize(NY).replace(hour=16).tz_convert("UTC")
    # 전날 뉴욕 16:00 봉이 없으면(주말·휴장) 더 앞으로
    for back in range(1, 5):
        cand = pd.Timestamp((d - pd.Timedelta(days=back)).date()).tz_localize(NY).replace(hour=16).tz_convert("UTC")
        if not np.isnan(price_at(cand)):
            prev_ny = cand
            break
    p_in, p_ex, p_cl, p_prev = price_at(entry), price_at(exit_exec), price_at(ny_close), price_at(prev_ny)
    if d.normalize() in ROLL:  # 사전 정의대로 아시아 날짜만 검사
        continue
    rows.append({"date": d, "실행 18→24": np.log(p_ex / p_in), "연구 18→뉴욕16시": np.log(p_cl / p_in),
                 "us_prev": np.log(p_in / p_prev)})
df = pd.DataFrame(rows).set_index("date").join(feat, how="inner")

res = []
for target in ["실행 18→24", "연구 18→뉴욕16시"]:
    for problem in ["수익률", "변동성"]:
        d2 = df.dropna(subset=[target, "us_prev"]).copy()
        if problem == "수익률":
            y, xcols = d2[target], ["갈림", "us_prev", "공통"]
        else:
            d2["|us_prev|"] = d2["us_prev"].abs()
            y, xcols = d2[target].abs(), ["|갈림|", "|us_prev|", "|공통|"]
        mdl = sm.OLS(y, sm.add_constant(d2[xcols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        b1 = mdl.params[xcols[0]]
        rng = np.random.default_rng(SEED)
        a, u, yy, n = d2[[xcols[0], xcols[2]]].values, d2[xcols[1]].values, y.values, len(d2)
        hits = 0
        for _ in range(N_PERM):
            k = rng.integers(MIN_SHIFT, n - MIN_SHIFT)
            ar = np.roll(a, k, axis=0)
            hits += abs(ols_b1(yy, np.column_stack([np.ones(n), ar[:, 0], u, ar[:, 1]]))) >= abs(b1)
        res.append({"목표": target, "문제": problem, "b1": b1, "HAC p": mdl.pvalues[xcols[0]],
                    "순열 p": (1 + hits) / (1 + N_PERM), "n": n})
tab = pd.DataFrame(res)
tab["BH 유의(10%)"] = multipletests(tab["HAC p"], alpha=0.10, method="fdr_bh")[0]
tab["BH 유의(10%, 순열 p)"] = multipletests(tab["순열 p"], alpha=0.10, method="fdr_bh")[0]
tab.to_csv(OUT / "3_regression_mnq.csv", index=False)
print(f"롤오버로 뺀 날 포함 후보 {len(feat[(feat.index >= '2024-06-27') & (feat.index < HOLDOUT_FROM)])}일 → 사용 {len(df)}일")
print(tab.round(4).to_string(index=False))
