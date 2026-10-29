"""저장된 과거 5분봉으로 전략을 백테스트하고 설정 버전별 결과를 비교한다.

API는 호출하지 않는다.
"""

import argparse
import logging
import math
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from core.bars import load_bars
from core.config import DATA_DIR, load_json
from core.indicators import atr
from week02.strategy import BUY, crossover_signals, daily_direction

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    version: str
    date: str
    side: int  # BUY(1) 또는 SELL(-1)
    entry_time: str
    entry_price: float
    stop_price: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""  # 청산 사유: stop, cross, close
    ticks: int = 0  # 진입과 청산 슬리피지를 반영한 손익(틱)


def get_args():
    """명령행 인자를 읽어 반환한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--period", choices=["compare", "confirm"], default="compare")
    parser.add_argument("--version", default="all")
    parser.add_argument("--slippage", type=int, help="config의 slippage_ticks 대신 사용할 값")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = get_args()
    config = load_json(args.config)
    dates = config["backtest"]

    bars = [
        bar
        for bar in load_bars(DATA_DIR / f"{config['symbol']}_{config['interval_minutes']}min.csv")
        if dates["start_date"] <= bar.date <= dates["end_date"]
    ]
    daily_bars = load_bars(DATA_DIR / f"{config['symbol']}_daily.csv")

    # 지표는 전체 기간으로 계산하고, 결과는 선택한 기간의 거래만 집계한다.
    # 확인 기간은 이전 날짜의 봉으로 이동평균이 준비된 상태에서 시작한다.
    if args.period == "compare":
        in_period = lambda date: date <= dates["split_date"]
    else:
        in_period = lambda date: date > dates["split_date"]
    days = sorted({bar.date for bar in bars if in_period(bar.date)})
    logger.info(f"{args.period} 기간: {days[0]} ~ {days[-1]}, {len(days)}거래일")

    versions = config["versions"]
    if args.version != "all":
        versions = {args.version: versions[args.version]}

    summaries = []
    for name, overrides in versions.items():
        params = {**config["strategy"], **overrides}
        if args.slippage is not None:
            params["slippage_ticks"] = args.slippage
        version_trades = [
            trade
            for trade in run_backtest(bars, daily_bars, params, name, config["tick_size"], config["interval_minutes"])
            if in_period(trade.date)
        ]
        summaries.append(summarize(name, version_trades, len(days)))

    for row in summaries:
        logger.info(
            f"{row['version']:<16} 거래 {row['trades']:>3} (하루 {row['trades_per_day']}) | 승률 {row['win_rate']} | "
            f"합계 {row['total_ticks']:>4}틱 | 평균 {row['avg_ticks']}틱 | 최대낙폭 {row['max_drawdown_ticks']}틱 | "
            f"손절 {row['stop']} 교차 {row['cross']} 마감 {row['close']}"
        )


def run_backtest(bars, daily_bars, params, version, tick_size, interval_minutes):
    """하루씩 규칙을 적용해 거래 목록을 반환한다.

    신호는 봉 마감에 판정하고 다음 봉 시가에 체결한다.
    손절은 봉의 고가와 저가가 손절선에 닿았는지로 판정한다.
    slippage_ticks는 체결 한 번당 적용하며 진입과 청산 가격을 각각 불리하게 조정한다.
    """
    signals = crossover_signals([bar.close for bar in bars], params["short_window"], params["long_window"])
    atrs = atr(bars, params["atr_window"])
    slippage = params["slippage_ticks"] * tick_size
    trades = []

    start = 0
    for date, day_bars in groupby(bars, key=lambda bar: bar.date):
        day_bars = list(day_bars)
        allowed_side = daily_direction(daily_bars, date, params["daily_filter_window"]) if params["daily_filter"] else 0
        position = None
        order = None  # 다음 봉 시가에 체결할 주문: ("entry", 방향, 손절 폭) 또는 ("exit",)
        entries = 0
        day_ticks = 0

        for offset, bar in enumerate(day_bars):
            i = start + offset

            # 1. 직전 봉 마감에 예약한 주문을 이번 봉 시가에 체결한다.
            if order and order[0] == "entry":
                side, stop_distance = order[1], order[2]
                price = round(bar.open + side * slippage, 2)
                position = Trade(version, date, side, bar.time, price, round(price - side * stop_distance, 2))
                entries += 1
            elif order and order[0] == "exit":
                day_ticks += close_position(position, bar.time, bar.open, "cross", slippage, tick_size)
                trades.append(position)
                position = None
            order = None

            # 2. 마감 청산 시각이면 해당 봉 시가에 정리하고 그날 거래를 끝낸다.
            #    실제 운영은 15:28 청산 주문이지만 5분봉으로는 15:25 봉 시가 청산으로 근사한다.
            if bar.time >= params["close_time"]:
                if position:
                    day_ticks += close_position(position, bar.time, bar.open, "close", slippage, tick_size)
                    trades.append(position)
                    position = None
                break

            # 3. 손절선에 닿으면 손절한다.
            #    시가가 이미 손절선보다 불리하면 시가에 체결된 것으로 본다.
            stopped = False
            if position:
                if position.side == BUY and bar.low <= position.stop_price:
                    fill = min(bar.open, position.stop_price)
                elif position.side != BUY and bar.high >= position.stop_price:
                    fill = max(bar.open, position.stop_price)
                else:
                    fill = None
                if fill is not None:
                    day_ticks += close_position(position, bar.time, fill, "stop", slippage, tick_size)
                    trades.append(position)
                    position = None
                    stopped = True

            # 4. 봉 마감에 교차 신호를 판정한다. 손절한 봉의 신호는 쓰지 않는다.
            signal = signals[i]
            if signal == 0 or stopped:
                continue
            if position:
                # 반대 교차는 청산에만 쓰고, 같은 신호로 반대 방향 진입은 하지 않는다.
                if signal == -position.side:
                    order = ("exit",)
                continue

            # ATR 기반 값을 정수 틱으로 올림한 뒤 설정한 최소 손절 폭과 비교한다.
            # 올림 전에 소수점 아래 6자리로 반올림해 부동소수점 오차를 제거한다.
            stop_ticks = 0
            if atrs[i]:
                atr_ticks = math.ceil(round(params["atr_multiplier"] * atrs[i] / tick_size, 6))
                stop_ticks = max(params["min_stop_ticks"], atr_ticks)
            bar_end = add_minutes(bar.time, interval_minutes)
            # 현재 손실에 손절 폭과 왕복 슬리피지를 더해도 일일 한도 안일 때만 진입한다.
            can_enter = (
                params["entry_start"] <= bar_end <= params["entry_end"]
                and entries < params["max_entries_per_day"]
                and stop_ticks > 0
                and -day_ticks + stop_ticks + 2 * params["slippage_ticks"] <= params["daily_loss_limit_ticks"]
                and allowed_side in (0, signal)
            )
            if can_enter:
                order = ("entry", signal, stop_ticks * tick_size)

        # 마감 청산 시각의 봉이 없는 날은 마지막 봉 종가에 정리한다.
        if position:
            last = day_bars[-1]
            close_position(position, last.time, last.close, "close", slippage, tick_size)
            trades.append(position)

        start += len(day_bars)

    return trades


def close_position(trade, time, price, reason, slippage, tick_size):
    """포지션을 정리하고 손익(틱)을 반환한다.

    청산 가격에 불리한 방향으로 한 번의 슬리피지를 적용한다.
    """
    trade.exit_time = time
    trade.exit_price = round(price - trade.side * slippage, 2)
    trade.exit_reason = reason
    trade.ticks = round(trade.side * (trade.exit_price - trade.entry_price) / tick_size)
    return trade.ticks


def summarize(version, trades, days):
    """거래 목록을 버전별 요약 지표로 집계한다."""
    ticks = [trade.ticks for trade in trades]
    equity = peak = max_drawdown = 0
    for tick in ticks:
        equity += tick
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    reasons = Counter(trade.exit_reason for trade in trades)

    return {
        "version": version,
        "trades": len(ticks),
        "trades_per_day": round(len(ticks) / days, 2),
        "win_rate": round(sum(tick > 0 for tick in ticks) / len(ticks), 2) if ticks else 0,
        "total_ticks": sum(ticks),
        "avg_ticks": round(sum(ticks) / len(ticks), 2) if ticks else 0,
        "max_drawdown_ticks": max_drawdown,
        "stop": reasons["stop"],
        "cross": reasons["cross"],
        "close": reasons["close"],
    }


def add_minutes(hhmmss, minutes):
    """HHMMSS 형식의 시각에 분을 더해 같은 형식으로 반환한다."""
    total = int(hhmmss[:2]) * 60 + int(hhmmss[2:4]) + minutes
    return f"{total // 60:02d}{total % 60:02d}00"


if __name__ == "__main__":
    main()
