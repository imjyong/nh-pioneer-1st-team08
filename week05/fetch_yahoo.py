"""5주차 연구용 야후 데이터를 다시 받는다. 원본은 data/market/week05_yahoo/에만 둔다 (저장소에 올리지 않음).

- 일봉: auto_adjust=True(수정주가), 최근 5년
- 60분봉: MNQ=F, QQQ, 대만·닛케이·항셍·H주, auto_adjust=True, 최근 730일 (야후 60분봉 한도)
- 30분봉: 대만·닛케이·항셍·H주 최근 60일 (점심 봉 구조 확인용)
- 시각: 야후 일봉은 거래소 현지 날짜, 60분봉은 UTC
- 받은 뒤 fix_yahoo.py를 실행해 분할 오류를 고친다
- 60분봉은 받는 날 기준 최근 730일이라, 다시 받으면 분석(11/17~18 수신) 때와 시작일이 달라진다
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parents[2] / "data" / "market" / "week05_yahoo"
OUT.mkdir(parents=True, exist_ok=True)

HERE = Path(__file__).resolve().parent
NODES = [t for t in pd.read_csv(HERE / "map" / "nodes.csv", dtype=str)["ticker"] if t != "NVDA"]
AI13 = ["083450.KQ", "007660.KS", "000660.KS", "005930.KS", "6787.T", "5805.T", "8035.T", "6857.T",
        "2421.TW", "6412.TW", "2368.TW", "6213.TW", "2330.TW"]

# 1~2번 가설(지역 갈림) + 4~6번 가설(AI 13종목, 시장 지수)
DAILY = ["^TWII", "^N225", "3033.HK", "^HSI", "^HSCE", "NVDA", "QQQ", "MNQ=F",
         "4062.T", "9988.HK", "0700.HK", "0981.HK", "^KS11", "^KQ11", "1306.T"] + AI13
# 1~2번 가설 + 3번 가설(공급망 지도 회사와 시장 조정용 지수)
HOURLY = ["MNQ=F", "QQQ", "^TWII", "^N225", "^HSI", "^HSCE", "^KS11", "1306.T"] + NODES
HALF = ["^HSI", "^HSCE", "^N225", "^TWII"]  # 점심 봉 구조 확인용 30분봉 (최근 60일)


def save(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise SystemExit(f"{ticker} {interval} 조회 결과 없음")
    df.columns = df.columns.get_level_values(0)
    name = ticker.replace("^", "_")
    path = OUT / (f"{name}_1d_adj.csv" if interval == "1d" else f"{name}_{interval}_adj.csv")
    df.to_csv(path)
    print(f"{ticker} {interval}: {len(df)}행 {df.index[0]} ~ {df.index[-1]}")


if __name__ == "__main__":
    for t in DAILY:
        save(t, "1d", "5y")
    for t in HOURLY:
        save(t, "60m", "730d")
    for t in HALF:
        save(t, "30m", "60d")
