"""나스닥, 금 레그의 과거 1분봉을 조회해 CSV로 저장한다."""

import argparse
import logging

from core.bars import save_bars
from core.client import create_client
from core.config import DATA_DIR
from core.market_data import fetch_chart

logger = logging.getLogger(__name__)

# 4주차 데이터 파일을 덮어쓰지 않도록 주차 폴더에 따로 저장한다
OUT_DIR = DATA_DIR / "week06"


def get_args():
    parser = argparse.ArgumentParser()
    # 금 2월물(MGCG27)은 15~18시 거래가 드물어(11/23 확인) 실행과 백테스트 모두 12월물을 쓴다
    parser.add_argument("--symbols", nargs="+", default=["MNQZ26", "MGCZ26"])
    parser.add_argument("--count", type=int, default=40000, help="종목별로 받을 1분봉 개수")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = get_args()
    client = create_client("overseas")
    for symbol in args.symbols:
        payload = {"exch_cd": "FCME", "sym": symbol, "dcnt": "1"}
        bars = fetch_chart(client, "/trade/v1/overseas/chart-minute", payload, args.count)
        if not bars:
            raise SystemExit(f"조회 결과가 없습니다. 종목코드({symbol})를 확인하세요.")
        save_bars(bars, OUT_DIR / f"{symbol}_1min.csv")
        logger.info(f"{symbol} 1분봉 {len(bars)}개 ({bars[0].date} ~ {bars[-1].date})")


if __name__ == "__main__":
    main()
