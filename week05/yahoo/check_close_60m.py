"""야후 60분봉 마지막 봉 종가와 일봉 종가를 AI 13종목 최근 40일로 비교한다.
운영 신호에서 최신 종가가 비었을 때 60분봉으로 대신할 수 있는지 보려고 만들었다. 결과: results/check_close_60m.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))  # research의 종목 목록과 결과 경로를 쓴다

import numpy as np
import pandas as pd
import yfinance as yf

from second_tier_delay import COUNTRY, RES

rows = []
for g in COUNTRY.values():
    for t in g["sec"] + g["large"]:
        d = yf.download(t, period="40d", interval="1d", progress=False, auto_adjust=False)
        h = yf.download(t, period="40d", interval="60m", progress=False, auto_adjust=False)
        d.columns, h.columns = d.columns.get_level_values(0), h.columns.get_level_values(0)
        h.index = h.index.tz_convert(None).normalize()
        x = pd.concat([d["Close"], h.groupby(level=0)["Close"].last()], axis=1, keys=["일봉", "60분봉"]).dropna()
        diff = np.log(x["60분봉"] / x["일봉"]).abs() * 100
        rows.append(dict(종목=t, 비교일=len(x), 기간=f"{x.index[0].date()}~{x.index[-1].date()}",
                         차이중앙값_퍼센트=round(diff.median(), 3), 차이최대_퍼센트=round(diff.max(), 3)))
out = pd.DataFrame(rows)
out.to_csv(RES / "check_close_60m.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
