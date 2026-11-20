"""가설 2(문서 4단계): 아시아 오전장 갈림이 오후 MNQ 수익률과 변동성을 예측하는지 검정한다. 정의는 4단계_오전장_사전고정_1117.md 그대로."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

RAW = Path(__file__).resolve().parents[4] / "data" / "market" / "week05_yahoo"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
HOLDOUT_FROM = pd.Timestamp("2026-11-12")
KST, NY = "Asia/Seoul", "America/New_York"
N_BOOT, SEED = 1000, 0

# 시장: (파일, 시가 봉 시각, 종가 봉 시각) — 한국 시각
MORNING = {"대만": ("_TWII", "10:00", "12:00"), "닛케이": ("_N225", "09:00", "11:00"),
           "항셍": ("_HSI", "10:30", "12:30"), "H주": ("_HSCE", "10:30", "12:30")}


def read60(name):
    df = pd.read_csv(RAW / f"{name}_60m_adj.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def morning_returns():
    out = {}
    for mkt, (f, t_open, t_close) in MORNING.items():
        df = read60(f)
        k = df.index.tz_convert(KST)
        df["date"], df["hm"] = k.normalize().tz_localize(None), k.strftime("%H:%M")
        o = df[df["hm"] == t_open].set_index("date")["Open"]
        c = df[df["hm"] == t_close].set_index("date")["Close"]
        out[mkt] = np.log(c / o)
    return pd.DataFrame(out).dropna()


def features(r):
    z = r / r.rolling(60).std().shift(1)
    sup, hk = z[["대만", "닛케이"]].mean(axis=1), z[["항셍", "H주"]].mean(axis=1)
    f = pd.DataFrame({"갈림": sup - hk, "공통": (sup + hk) / 2}).dropna()
    f["|갈림|"], f["|공통|"] = f["갈림"].abs(), f["공통"].abs()
    return f


m = read60("MNQ=F")
m = m[m.index.minute == 0]["Open"]
daily = pd.read_csv(RAW / "MNQ=F_1d_adj.csv", index_col=0, parse_dates=True).index


def at(d, hour, tz):
    return m.get(pd.Timestamp(d.date()).tz_localize(tz).replace(hour=hour).tz_convert("UTC"), np.nan)


def roll_days():
    out = set()
    for y in range(2024, 2027):
        for mth in (3, 6, 9, 12):
            d = pd.Timestamp(y, mth, 15)
            tf = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
            pos = daily.searchsorted(tf)
            out |= {daily[k].normalize() for k in range(pos - 1, pos + 3) if 0 <= k < len(daily)}
    return out


def build():
    feat = features(morning_returns())
    roll = roll_days()
    rows = []
    for d in feat.index:
        if d < pd.Timestamp("2024-06-27") or d >= HOLDOUT_FROM or d in roll:
            continue
        p14, p16, p18 = at(d, 14, KST), at(d, 16, KST), at(d, 18, KST)
        prev = np.nan
        for back in range(1, 5):
            prev = at(d - pd.Timedelta(days=back), 16, NY)
            if not np.isnan(prev):
                break
        rows.append({"date": d, "A 14→16": np.log(p16 / p14), "B 14→18": np.log(p18 / p14), "통제": np.log(p14 / prev)})
    return pd.DataFrame(rows).set_index("date").join(feat, how="inner").dropna()


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def boot_p(y, X_full, X_restr, t_obs, block, rng):
    """γ=0 제한모형의 잔차에 블록 단위 ±1 부호를 곱해 HAC t 통계량의 양측 귀무분포를 만든다."""
    r = sm.OLS(y, sm.add_constant(X_restr)).fit()
    fitted, resid = r.fittedvalues.values, r.resid.values
    n, hits = len(y), 0
    for _ in range(N_BOOT):
        signs = np.repeat(rng.choice([-1.0, 1.0], n // block + 1), block)[:n]
        yb = pd.Series(fitted + resid * signs, index=y.index)
        t_b = hac(yb, X_full).tvalues.iloc[1]
        hits += abs(t_b) >= abs(t_obs)
    return (1 + hits) / (1 + N_BOOT)


def walk_forward(df, ycol, xcols, problem):
    init, step = 252, 63
    ew, eo, blocks = [], [], []
    for start in range(init, len(df), step):
        tr, te = df.iloc[:start], df.iloc[start:start + step]
        e = {}
        for key, cols in [("with", xcols), ("without", xcols[1:])]:
            mm = sm.OLS(tr[ycol], sm.add_constant(tr[cols])).fit()
            e[key] = te[ycol].values - mm.predict(sm.add_constant(te[cols], has_constant="add")).values
        ew.append(e["with"]), eo.append(e["without"])
        if problem == "수익률":
            blocks.append(1 - (e["with"] ** 2).sum() / (e["without"] ** 2).sum() > 0)
        else:
            blocks.append(np.abs(e["with"]).mean() < np.abs(e["without"]).mean())
    ew, eo = np.concatenate(ew), np.concatenate(eo)
    if problem == "수익률":
        return 1 - (ew ** 2).sum() / (eo ** 2).sum(), blocks, len(ew)
    return np.abs(eo).mean() - np.abs(ew).mean(), blocks, len(ew)


def main():
    df = build()
    print(f"사용 관측일: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")
    if len(df) < 300:
        raise SystemExit("공통 완전 관측일 300일 미만 → 중단")
    rng = np.random.default_rng(SEED)
    rows = []
    for target in ["A 14→16", "B 14→18"]:
        for problem in ["수익률", "변동성"]:
            d = df.copy()
            if problem == "수익률":
                ycol, xcols = target, ["갈림", "통제", "공통"]
            else:
                d["|y|"], d["|통제|"] = d[target].abs(), d["통제"].abs()
                ycol, xcols = "|y|", ["|갈림|", "|통제|", "|공통|"]
            fit = hac(d[ycol], d[xcols])
            t = fit.tvalues[xcols[0]]
            ps = {b: boot_p(d[ycol], d[xcols], d[xcols[1:]], t, b, rng) for b in (10, 5, 20)}
            wf, blocks, n_te = walk_forward(d, ycol, xcols, problem)
            row = {"목표": target, "문제": problem, "b1": fit.params[xcols[0]], "HAC p": fit.pvalues[xcols[0]],
                   "부트 p(10일)": ps[10], "부트 p(5일)": ps[5], "부트 p(20일)": ps[20],
                   "워크포워드": wf, "개선 블록": f"{sum(blocks)}/{len(blocks)}", "시험 n": n_te}
            if problem == "수익률":
                pred_sign = []
                for start in range(252, len(d), 63):
                    tr, te = d.iloc[:start], d.iloc[start:start + 63]
                    mm = sm.OLS(tr[ycol], sm.add_constant(tr[xcols])).fit()
                    p = mm.predict(sm.add_constant(te[xcols], has_constant="add"))
                    pred_sign.append((np.sign(p.values) == np.sign(te[ycol].values)).mean() * len(te))
                row["방향 적중률(보조)"] = sum(pred_sign) / n_te
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab["BH 유의(10%)"] = multipletests(tab["HAC p"], alpha=0.10, method="fdr_bh")[0]
    ok_wf = tab["워크포워드"] > 0
    tab["후보 인정"] = ok_wf & (tab["부트 p(10일)"] < 0.10)
    tab.to_csv(OUT / "4_morning.csv", index=False)
    pd.set_option("display.width", 220)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
