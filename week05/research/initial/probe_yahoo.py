"""야후 파이낸스에서 아시아 AI 공급망·홍콩 기술주·미국 종목의 일봉과 60분봉이 받아지는지 확인한다."""
from pathlib import Path
import yfinance as yf

OUT = Path(__file__).resolve().parents[4] / "data" / "market" / "week05_yahoo"
TICKERS = {
    "2330.TW": "TSMC", "6857.T": "어드밴테스트", "8035.T": "도쿄일렉트론", "4062.T": "이비덴",
    "9988.HK": "알리바바", "0700.HK": "텐센트", "0981.HK": "SMIC", "^HSTECH": "항셍테크 지수", "3033.HK": "항셍테크 ETF",
    "NVDA": "엔비디아", "QQQ": "나스닥100 ETF", "^TWII": "대만 가권", "^N225": "닛케이",
}
for t, name in TICKERS.items():
    row = [f"{t:9s} {name:10s}"]
    for interval, period in [("1d", "5y"), ("60m", "730d"), ("30m", "60d")]:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception as e:
            row.append(f"{interval}: 오류 {type(e).__name__}"); continue
        if df is None or df.empty:
            row.append(f"{interval}: 없음"); continue
        df.to_csv(OUT / f"{t.replace('^','_')}_{interval}.csv")
        idx = df.index
        row.append(f"{interval}: {len(df)}행 {idx[0]:%Y-%m-%d}~{idx[-1]:%Y-%m-%d %H:%M} tz={idx.tz}")
    print(" | ".join(row))
