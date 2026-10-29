"""과거 분봉과 일봉을 조회해 CSV로 저장한다. 주문은 내지 않는다."""

import argparse
import logging
from pathlib import Path

from core.bars import save_bars
from core.client import create_client
from core.config import DATA_DIR, load_json
from core.market_data import fetch_daily_bars, fetch_minute_bars

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_json(get_args().config)
    symbol = config["symbol"]
    interval_minutes = config["interval_minutes"]
    client = create_client("domestic")

    minute_bars = fetch_minute_bars(client, symbol, config["minute_bars"], interval_minutes)
    daily_bars = fetch_daily_bars(client, symbol, config["daily_bars"])
    if not minute_bars or not daily_bars:
        raise SystemExit(f"조회 결과가 없습니다. 종목코드({symbol})를 확인하세요.")

    save_bars(minute_bars, DATA_DIR / f"{symbol}_{interval_minutes}min.csv")
    logger.info(f"{interval_minutes}분봉 {len(minute_bars)}개 ({minute_bars[0].date} ~ {minute_bars[-1].date})")

    save_bars(daily_bars, DATA_DIR / f"{symbol}_daily.csv")
    logger.info(f"일봉 {len(daily_bars)}개 ({daily_bars[0].date} ~ {daily_bars[-1].date})")


if __name__ == "__main__":
    main()
