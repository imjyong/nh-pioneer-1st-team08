"""11/19에 동결한 모델로 아시아 마감 뒤 신호를 계산해 results/signal_log.csv에 기록한다.
학습에는 11/11까지의 고정 자료만 쓰고 당일 자료는 예측에만 쓰며, 주문은 live.py가 낸다.
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent / "research"))  # second_tier_delay, vol_ml

import second_tier_delay  # noqa: E402
import vol_ml  # noqa: E402

TH = 0.30
KST = ZoneInfo("Asia/Seoul")
LIVE = Path(__file__).resolve().parents[2] / "data" / "market" / "week05_live"
LOG = second_tier_delay.RES / "signal_log.csv"
TRAIN_END = "2026-11-11"


def fetch_live(t):
    """최근 3개월 일봉을 받는다. 마지막 종가가 없으면 history, 시세 정보 순으로 종가와 거래량을 보완한다."""
    df = yf.download(t, period="3mo", interval="1d", progress=False, auto_adjust=True)  # 분석과 같은 수정주가
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    note = ""
    if len(df) and pd.isna(df["Close"].iloc[-1]):
        d = df.index[-1]
        h = yf.Ticker(t).history(period="10d", auto_adjust=True)
        h.index = h.index.tz_localize(None).normalize()
        if d in h.index and pd.notna(h.loc[d, "Close"]):
            df.loc[d, ["Open", "High", "Low", "Close", "Volume"]] = h.loc[d, ["Open", "High", "Low", "Close", "Volume"]].values
            note = "history"
        else:
            info = yf.Ticker(t).info
            ts, tz = info.get("regularMarketTime"), info.get("exchangeTimezoneName", "UTC")
            if ts and pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize() == d:
                df.loc[d, ["Open", "Close", "Volume"]] = [info.get("regularMarketOpen"), info.get("regularMarketPrice"),
                                                          info.get("regularMarketVolume") or 1]
                note = "시세정보"
    df.to_csv(LIVE / f"{t.replace('^', '_')}_live.csv")
    return df, note


def frozen_model():
    """분석용 고정 자료로 학습한다. 실행할 때마다 같은 계수가 나와야 한다."""
    vol_ml.END = pd.Timestamp(TRAIN_END)
    D = vol_ml.build("A_QQQ장중")
    cols = vol_ml.BASE + vol_ml.M1
    m = vol_ml.logit().fit(D[cols].values, D["y"].values)
    b = vol_ml.logit().fit(D[vol_ml.BASE].values, D["y"].values)
    coef = m[-1].coef_[0]
    return m, b, cols, len(D), str(D.index[-1].date()), ";".join(f"{c}={v:+.3f}" for c, v in zip(cols, coef))


def main():
    now = datetime.now(KST)
    check = len(sys.argv) > 1  # 날짜를 주면 지난 날짜로 점검한다 (예: python3 daily_signal.py 2026-11-18)
    today = pd.Timestamp(sys.argv[1] if check else now.date())
    LIVE.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    rets, used, excluded, notes = {}, [], [], []
    for c, g in second_tier_delay.COUNTRY.items():
        for t in g["sec"] + g["large"]:
            df, note = fetch_live(t)
            if note and today in df.index:
                notes.append(f"{t}({note})")
            if today not in df.index or pd.isna(df.loc[today, "Close"]) or not df.loc[today, "Volume"] > 0:
                excluded.append(f"{t}(오늘 값 없음)")
                continue
            prev = df.loc[df.index < today, "Close"].dropna()
            if prev.empty:
                excluded.append(f"{t}(직전 값 없음)")
                continue
            rets[t] = (c, np.log(df.loc[today, "Close"] / prev.iloc[-1]))
            used.append(t)

    countries_open = {c for c, _ in rets.values()}
    if len(used) < 10 or countries_open != set(second_tier_delay.COUNTRY):
        decision = "거래 안 함 / 사유: 오늘 아시아 자료 부족(10종목 미만 또는 세 나라 중 휴장)"
        x = None
    else:
        r = pd.Series({t: v for t, (_, v) in rets.items()})
        by_c = pd.Series({c: np.mean([v for cc, v in rets.values() if cc == c]) for c in countries_open})
        q, _ = fetch_live("QQQ")
        s = np.log(q["Close"] / q["Open"]).abs()
        s = s[s.index < today].dropna()
        x = pd.Series({
            "disp": r.std(), "breadth_dev": abs((r > 0).mean() - 0.5), "ctry_gap": by_c.max() - by_c.min(),
            "ai_abs": abs(r.mean()), "log_m20": np.log(s.iloc[-20:].mean()), "log_last": np.log(s.iloc[-1] + 1e-4),
        })
        qqq_last = str(s.index[-1].date())

    m, b, cols, n_train, train_end, coef = frozen_model()
    if x is not None:
        p = m.predict_proba(x[cols].values.reshape(1, -1))[0, 1]
        pb = b.predict_proba(x[vol_ml.BASE].values.reshape(1, -1))[0, 1]
        decision = "매수(미국장 개장~마감)" if p < TH else "거래 안 함"
        if not check and now.hour * 60 + now.minute < 16 * 60 + 10:
            decision += " / 주의: 16:10 전 계산, 주문 금지"
    else:
        p = pb = np.nan
        qqq_last = ""

    row = dict(구분="점검(사후 계산)" if check else "실시간", 계산시각_KST=now.strftime("%Y-%m-%d %H:%M"),
               아시아날짜=str(today.date()), 모델=f"고정 학습 {n_train}행(~{train_end})", 계수=coef,
               기본확률=round(pb, 4), 방법1확률=round(p, 4), 임계값=TH, 결정=decision,
               사용종목수=len(used), 제외종목=";".join(excluded) or "없음", 종가대체=";".join(notes) or "없음",
               QQQ직전날짜=qqq_last,
               **({k: round(float(x[k]), 6) for k in cols} if x is not None else {}))
    out = pd.DataFrame([row])
    header = not LOG.exists()
    out.to_csv(LOG, mode="a", header=header, index=False, encoding="utf-8-sig")
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
