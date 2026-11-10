"""두 레그의 과거 1분봉을 조회해 CSV로 저장한다."""

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
    parser.add_argument("--count", type=int, default=40000, help="레그별로 받을 1분봉 개수")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = get_args()
    config = load_json(args.config)
    client = create_client("overseas")
    for leg in config["legs"]:
        payload = {"exch_cd": config["exchange"], "sym": leg["symbol"], "dcnt": "1"}
        bars = fetch_chart(client, "/trade/v1/overseas/chart-minute", payload, args.count)
        if not bars:
            raise SystemExit(f"조회 결과가 없습니다. 종목코드({leg['symbol']})를 확인하세요.")
        save_bars(bars, DATA_DIR / f"{leg['symbol']}_1min.csv")
        logger.info(f"{leg['symbol']} 1분봉 {len(bars)}개 ({bars[0].date} ~ {bars[-1].date})")


if __name__ == "__main__":
    main()
