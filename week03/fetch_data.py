"""과거 분봉을 조회해 CSV로 저장한다"""

import argparse
import logging
from pathlib import Path

from core.bars import save_bars
from core.client import create_client
from core.config import DATA_DIR, load_json
from core.market_data import fetch_chart

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
    payload = {"exch_cd": config["exchange"], "sym": symbol, "dcnt": str(interval_minutes)}

    client = create_client("overseas")
    bars = fetch_chart(client, "/trade/v1/overseas/chart-minute", payload, config["minute_bars"])
    if not bars:
        raise SystemExit(f"조회 결과가 없습니다. 종목코드({symbol})를 확인하세요.")

    save_bars(bars, DATA_DIR / f"{symbol}_{interval_minutes}min.csv")
    logger.info(f"{interval_minutes}분봉 {len(bars)}개 ({bars[0].date} ~ {bars[-1].date})")


if __name__ == "__main__":
    main()
