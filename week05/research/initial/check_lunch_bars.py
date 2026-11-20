"""60분봉 각 시각 봉이 실제로 어떤 30분 구간을 담는지 최근 60일 30분봉으로 대조한다."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[4] / "data" / "market" / "week05_yahoo"


def rd(t, iv):
    df = pd.read_csv(RAW / f"{t.replace('^', '_')}_{iv}_adj.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Seoul")
    return df


for t in ["^N225", "^TWII", "^HSI", "^HSCE"]:
    h, m = rd(t, "60m"), rd(t, "30m")
    h = h[h.index >= m.index[0]]
    print(f"== {t}")
    for hh, g in h.groupby(h.index.strftime("%H:%M")):
        res = {"[같은시각 30분]": 0, "[같은시각+30분 두 개]": 0, "[30분 뒤 30분만]": 0, "기타": 0}
        for ts, r in g.iterrows():
            a = m.get("Close").get(ts, np.nan)
            b = m.get("Close").get(ts + pd.Timedelta(minutes=30), np.nan)
            o = m.get("Open").get(ts, np.nan)
            if not np.isnan(b) and np.isclose(r["Close"], b) and np.isclose(r["Open"], o):
                res["[같은시각+30분 두 개]"] += 1
            elif np.isclose(r["Close"], a) and np.isclose(r["Open"], o):
                res["[같은시각 30분]"] += 1
            elif not np.isnan(b) and np.isclose(r["Close"], b) and np.isclose(r["Open"], m["Open"].get(ts + pd.Timedelta(minutes=30), np.nan)):
                res["[30분 뒤 30분만]"] += 1
            else:
                res["기타"] += 1
        print(f"  {hh} 봉({len(g)}일): " + ", ".join(f"{k} {v}" for k, v in res.items() if v))
