"""야후 원본에서 분할 처리가 꼬인 곳을 바로잡는다. fetch_yahoo.py 다음에 실행.
파일마다 분할 전후 불연속이 아직 남아 있을 때만 고치므로 여러 번 실행해도 된다. 고치기 직전 파일은 *_orig.csv로 남긴다.

- 1306.T 일봉: 2026-03-30, 03-31 두 날만 1/10로 기록됨 (1:10 분할 이중 반영) → ×10
  공식 분할: Nomura NEXT FUNDS 공지 https://nextfunds.jp/en/data/2026/td_en_260217d.pdf (3/30부터 분할 후 가격)
- 1306.T 60분봉: 2026-03-30 분할 전 봉이 수정되지 않음 → 그 전 가격 ÷10, 거래량 ×10
  주의: 일봉은 분할 후 기준을 과거로, 60분봉은 분할 후 기준으로 맞췄다. 수익률 분석에는 문제없지만 두 파일의 가격 수준을 직접 비교하면 안 된다
- 083450.KQ 일봉: 2024-06-17에 전날 대비 약 1/2로 불연속 → 그 전 가격 ÷2, 거래량 ×2
  공식 기업행사 자료를 확인하지 못해 가격 연속성만 복원한 보정이다
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "data" / "market" / "week05_yahoo"
PX = ["Open", "High", "Low", "Close"]


def patch(name, needs_fix, fn):
    path = DATA / name
    df = pd.read_csv(path, index_col=0)
    if not needs_fix(df):
        print(f"{name}: 불연속 없음, 그대로 둠")
        return
    df.to_csv(DATA / name.replace(".csv", "_orig.csv"))
    fn(df)
    df.to_csv(path)
    print(f"{name}: 고침")


def close_at(df, day):
    m = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize() == pd.Timestamp(day)
    return df.loc[m, "Close"]


# 1306.T 일봉
def topix_daily_bad(df):
    return close_at(df, "2026-03-30").iloc[0] < 0.2 * close_at(df, "2026-03-27").iloc[0]


def topix_daily(df):
    m = df.index.isin(["2026-03-30", "2026-03-31"])
    df.loc[m, PX] *= 10


# 1306.T 60분봉
def topix_hourly_bad(df):
    t = pd.to_datetime(df.index, utc=True)
    cut = pd.Timestamp("2026-03-30", tz="Asia/Tokyo")
    return df.loc[t < cut, "Close"].iloc[-1] > 5 * df.loc[t >= cut, "Open"].iloc[0]


def topix_hourly(df):
    m = pd.to_datetime(df.index, utc=True) < pd.Timestamp("2026-03-30", tz="Asia/Tokyo")
    df.loc[m, PX] /= 10
    df.loc[m, "Volume"] *= 10


# 083450.KQ 일봉
def gst_daily_bad(df):
    return close_at(df, "2024-06-14").iloc[0] > 1.6 * close_at(df, "2024-06-17").iloc[0]


def gst_daily(df):
    m = pd.to_datetime(df.index) < pd.Timestamp("2024-06-17")
    df.loc[m, PX] /= 2
    df.loc[m, "Volume"] *= 2


if __name__ == "__main__":
    patch("1306.T_1d_adj.csv", topix_daily_bad, topix_daily)
    patch("1306.T_60m_adj.csv", topix_hourly_bad, topix_hourly)
    patch("083450.KQ_1d_adj.csv", gst_daily_bad, gst_daily)
