"""봉 데이터를 정의하고 CSV로 저장하거나 불러온다."""

import csv
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Bar:
    date: str  # YYYYMMDD
    time: str  # HHMMSS, 일봉은 ""
    open: float
    high: float
    low: float
    close: float
    cum_volume: int  # 그날 누적 거래량
    open_interest: int


def save_bars(bars, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Bar)])
        writer.writeheader()
        writer.writerows(asdict(bar) for bar in bars)


def load_bars(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Bar(
                date=row["date"],
                time=row["time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                cum_volume=int(row["cum_volume"]),
                open_interest=int(row["open_interest"]),
            )
            for row in csv.DictReader(f)
        ]


def add_minutes(hhmmss, minutes):
    total = int(hhmmss[:2]) * 60 + int(hhmmss[2:4]) + minutes
    return f"{total // 60:02d}{total % 60:02d}00"
