"""저장된 과거 5분봉으로 전략을 백테스트하고 설정 버전별 결과를 비교한다."""

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path

from core.bars import add_minutes, load_bars
from core.config import DATA_DIR, load_json
from core.indicators import vwap
from week03.strategy import (
    BUY,
    SELL,
    breakout_signal,
    calculate_stop_price,
    calculate_target_price,
    can_enter,
    opening_range,
    reached_breakeven,
    sma_direction,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    date: str
    side: int  # BUY(1) 또는 SELL(-1)
    entry_time: str
    entry_price: float
    first_stop: float
    stop_price: float
    target_price: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""  # 청산 사유: stop, breakeven, target, time
    ticks: int = 0  # 진입과 청산 슬리피지를 반영한 손익(틱)


def get_args():
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
    bars = align_to_winter_time(bars, dates["summer_time_end"], config["strategy"]["vwap_start"])

    if args.period == "compare":
        in_period = lambda date: date <= dates["split_date"]
    else:
        in_period = lambda date: date > dates["split_date"]
    days = sorted({bar.date for bar in bars if in_period(bar.date)})
    if not days:
        raise SystemExit(f"{args.period} 기간에 데이터가 없습니다.")
    logger.info(f"{args.period} 기간: {days[0]} ~ {days[-1]}, {len(days)}거래일")

    versions = config["versions"]
    if args.version != "all":
        versions = {args.version: versions[args.version]}

    for name, overrides in versions.items():
        params = {**config["strategy"], **overrides}
        if args.slippage is not None:
            params["slippage_ticks"] = args.slippage
        trades, skipped = run_backtest(bars, params, config, in_period)
        report(name, trades, skipped, config)


def align_to_winter_time(bars, summer_time_end, vwap_start):
    """서머타임 기간 봉을 1시간 늦춰 미국 정규장 개장을 23:30에 맞추고, 거래일 시작 전과 자정 이후 봉은 뺀다."""
    aligned = []
    for bar in bars:
        time = add_minutes(bar.time, 60) if bar.date < summer_time_end else bar.time
        if vwap_start <= time < "240000":
            aligned.append(replace(bar, time=time))
    return aligned


def run_backtest(bars, params, config, in_period):
    """하루씩 규칙을 적용해 거래 목록과 한도로 제외된 신호 수를 반환한다.

    신호는 봉 마감에 판정하고 다음 봉 시가에 체결한다.
    한 봉에서 손절과 익절에 모두 닿으면 손절로 본다. 본전 이동은 다음 봉부터 적용한다.
    """
    tick_size = config["tick_size"]
    interval_minutes = config["interval_minutes"]
    slippage = params["slippage_ticks"] * tick_size
    trades = []
    skipped = 0

    for date, day_bars in groupby(bars, key=lambda bar: bar.date):
        if not in_period(date):
            continue
        day_bars = list(day_bars)
        price_range = opening_range(day_bars, params, interval_minutes)
        if not price_range:
            continue
        vwaps = vwap(day_bars, [bar.cum_volume for bar in day_bars])

        for i, bar in enumerate(day_bars[:-1]):
            signal = breakout_signal(bar.close, *price_range, vwaps[i])
            stop_price = calculate_stop_price(signal, *price_range, tick_size)
            stop_ticks = round(abs(bar.close - stop_price) / tick_size)
            bar_end = add_minutes(bar.time, interval_minutes)
            allowed_side = sma_direction([item.close for item in day_bars[: i + 1]], params["sma_window"]) if params["sma_filter"] else 0
            if not can_enter(params, signal, bar_end, 0, min(stop_ticks, params["max_stop_ticks"]), allowed_side):
                continue
            if stop_ticks > params["max_stop_ticks"]:  # 실행 코드와 같이 한도를 넘으면 그날 거래를 끝낸다
                skipped += 1
                break
            trades.append(simulate_trade(day_bars[i + 1 :], date, signal, stop_price, params, tick_size, slippage))
            break

    return trades, skipped


def simulate_trade(bars, date, side, stop_price, params, tick_size, slippage):
    entry_price = round(bars[0].open + side * slippage, 2)
    target_price = calculate_target_price(side, entry_price, stop_price, params)
    trade = Trade(date, side, bars[0].time, entry_price, stop_price, stop_price, target_price)

    for bar in bars:
        if bar.time >= params["close_time"]:
            return close_trade(trade, bar.time, bar.open, "time", slippage, tick_size)

        favorable, adverse = (bar.high, bar.low) if side == BUY else (bar.low, bar.high)
        if side * (adverse - trade.stop_price) <= 0:
            # 시가가 이미 손절선보다 불리하면 시가에 체결된 것으로 본다.
            fill = min(bar.open, trade.stop_price) if side == BUY else max(bar.open, trade.stop_price)
            reason = "breakeven" if trade.stop_price == entry_price else "stop"
            return close_trade(trade, bar.time, fill, reason, slippage, tick_size)
        if side * (favorable - target_price) >= 0:
            return close_trade(trade, bar.time, target_price, "target", slippage, tick_size)
        if reached_breakeven(side, entry_price, trade.first_stop, favorable):
            trade.stop_price = entry_price

    # 청산 시각 봉이 없는 날은 마지막 봉 종가에 정리한다.
    return close_trade(trade, bars[-1].time, bars[-1].close, "time", slippage, tick_size)


def close_trade(trade, time, price, reason, slippage, tick_size):
    trade.exit_time = time
    trade.exit_price = round(price - trade.side * slippage, 2)
    trade.exit_reason = reason
    trade.ticks = round(trade.side * (trade.exit_price - trade.entry_price) / tick_size)
    return trade


def report(version, trades, skipped, config):
    ticks = [trade.ticks for trade in trades]
    equity = peak = max_drawdown = 0
    for tick in ticks:
        equity += tick
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    reasons = Counter(trade.exit_reason for trade in trades)
    losses = [trade.ticks for trade in trades if trade.exit_reason == "stop"]
    count = len(ticks)
    tick_value = config["tick_value"]
    net = sum(ticks) * tick_value - count * config["backtest"]["round_trip_fee"]
    sides = Counter("매수" if trade.side == BUY else "매도" for trade in trades if trade.side in (BUY, SELL))

    logger.info(
        f"{version:<14} 거래 {count:>2} (매수 {sides['매수']} 매도 {sides['매도']}) | 한도 제외 {skipped} | "
        f"승률 {sum(tick > 0 for tick in ticks) / count if count else 0:.2f} | 합계 {sum(ticks):>5}틱 | "
        f"평균 {sum(ticks) / count if count else 0:.1f}틱 | 평균 손절 {sum(losses) / len(losses) if losses else 0:.0f}틱 | "
        f"최대낙폭 {max_drawdown}틱 | 손절 {reasons['stop']} 본전 {reasons['breakeven']} 익절 {reasons['target']} "
        f"시간 {reasons['time']} | 수수료 포함 ${net:+,.2f}"
    )


if __name__ == "__main__":
    main()
