"""저장된 과거 1분봉으로 스프레드 전략을 백테스트하고 설정 버전별 결과를 비교한다."""

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.bars import add_minutes, load_bars
from core.config import DATA_DIR, load_json
from week04.strategy import check_entry, contract_ratio, entry_side, exit_reason, pair_pnl, round_trip_cost, spread, zscore

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    date: str
    side: int
    quantities: tuple
    entry_time: str
    exit_time: str
    reason: str
    gross: float  # 슬리피지 반영, 수수료 전
    net: float


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--version", default="all")
    parser.add_argument("--session", help="config의 selected_session 대신 사용할 시간 설정")
    parser.add_argument("--slippage", type=int, help="config의 slippage_ticks 대신 사용할 값")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    config = load_json(args.config)
    dates = config["backtest"]
    session = config["sessions"][args.session or config["selected_session"]]
    slippage = dates["slippage_ticks"] if args.slippage is None else args.slippage

    legs = [load_leg(leg["symbol"], dates) for leg in config["legs"]]
    days = sorted(set(legs[0]) & set(legs[1]))
    logger.info(f"기간 {days[0]} ~ {days[-1]}, {len(days)}일, 비교 기간은 {dates['split_date']}까지, 슬리피지 {slippage}틱")

    versions = config["versions"]
    if args.version != "all":
        versions = {args.version: versions[args.version]}
    for name, overrides in versions.items():
        params = {**config["strategy"], **session, **overrides}
        trades, blocked = run_backtest(legs, days, params, config, slippage)
        report(name, trades, blocked, dates["split_date"])


def load_leg(symbol, dates):
    """날짜별로 {봉 시작 시각: 봉}을 만든다."""
    by_date = defaultdict(dict)
    for bar in load_bars(DATA_DIR / f"{symbol}_1min.csv"):
        if dates["start_date"] <= bar.date <= dates["end_date"]:
            by_date[bar.date][bar.time] = bar
    return by_date


def minute_prices(bars, times):
    """각 분의 (시가, 종가)를 만든다. 봉이 없는 분은 직전 종가로 채운다."""
    earlier = [time for time in bars if time < times[0]]
    last = bars[max(earlier)].close if earlier else None
    prices = []
    for time in times:
        bar = bars.get(time)
        if bar:
            prices.append((bar.open, bar.close))
            last = bar.close
        else:
            prices.append((last, last))
    return prices


def run_backtest(legs, days, params, config, slippage):
    """하루씩 규칙을 적용해 거래 목록과 진입을 막은 사유별 횟수를 반환한다.

    신호는 1분봉 종가로 판단하고 다음 분 시가에 체결한다.
    """
    point_values = [leg["point_value"] for leg in config["legs"]]
    tick_sizes = [leg["tick_size"] for leg in config["legs"]]
    tick_values = [size * point for size, point in zip(tick_sizes, point_values)]
    trades, blocked = [], Counter()

    for date in days:
        first = add_minutes(params["entry_start"], -(params["window"] + 5))
        times = [add_minutes(first, i) for i in range(minutes_between(first, params["close_time"]) + 2)]
        prices = [minute_prices(leg[date], times) for leg in legs]
        start = times.index(params["entry_start"])
        closes = [prices[0][start][1], prices[1][start][1]]
        if None in closes:
            blocked["데이터 부족"] += 1
            continue
        quantities = contract_ratio(closes, point_values, params["size_tolerance"])
        notional = sum(q * c * p for q, c, p in zip(quantities, closes, point_values)) / 2
        cost = round_trip_cost(quantities, config["fee_per_contract"], slippage, tick_values)

        spreads, position, entries, day_net, cooldown_until = [], None, 0, 0.0, None
        for i, time in enumerate(times[:-1]):
            (_, close1), (_, close2) = prices[0][i], prices[1][i]
            if close1 is None or close2 is None:
                continue
            spreads.append(spread(close1, close2))
            if time < params["entry_start"]:
                continue
            next_open = [prices[0][i + 1][0] or close1, prices[1][i + 1][0] or close2]
            result = zscore(spreads, params["window"])

            if position:
                gross = pair_pnl(position["side"], quantities, position["prices"], (close1, close2), point_values)
                position["peak"] = max(position["peak"], gross)
                held = minutes_between(position["time"], time)
                reason = "마감" if time >= params["close_time"] else None
                if not reason and result:
                    reason = exit_reason(position["side"], result[0], gross, position["peak"], cost, held, params)
                if reason:
                    side = position["side"]
                    exit_prices = [next_open[0] - side * slippage * tick_sizes[0], next_open[1] + side * slippage * tick_sizes[1]]
                    gross = pair_pnl(side, quantities, position["prices"], exit_prices, point_values)
                    net = gross - 2 * config["fee_per_contract"] * sum(quantities)
                    trades.append(Trade(date, side, quantities, position["time"], time, reason, gross, net))
                    day_net += net
                    position = None
                    if reason in ("z 손절", "금액 손절"):
                        cooldown_until = add_minutes(time, params["cooldown_minutes"])
                continue

            if not result or time >= params["entry_end"] or entries >= params["max_entries_per_day"]:
                continue
            if day_net <= -params["daily_loss"] or (cooldown_until and time < cooldown_until):
                continue
            side = entry_side(result[0], params["entry_z"])
            if not side:
                continue
            block = check_entry(result[0], result[1], notional, cost, params)
            if block:
                blocked[block] += 1
                continue
            entry_prices = [next_open[0] + side * slippage * tick_sizes[0], next_open[1] - side * slippage * tick_sizes[1]]
            position = {"side": side, "prices": entry_prices, "time": time, "peak": 0.0}
            entries += 1

    return trades, blocked


def minutes_between(start, end):
    return (int(end[:2]) * 60 + int(end[2:4])) - (int(start[:2]) * 60 + int(start[2:4]))


def report(version, trades, blocked, split_date):
    nets = [trade.net for trade in trades]
    equity = peak = max_drawdown = 0
    for net in nets:
        equity += net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    compare = sum(trade.net for trade in trades if trade.date <= split_date)
    confirm = sum(trade.net for trade in trades if trade.date > split_date)
    reasons = Counter(trade.reason for trade in trades)
    count = len(trades)
    logger.info(
        f"{version:<13} 거래 {count:>3} | 승률 {sum(net > 0 for net in nets) / count if count else 0:.2f} | "
        f"수수료 전 ${sum(trade.gross for trade in trades):+,.0f} | 순손익 ${sum(nets):+,.0f} "
        f"(비교 ${compare:+,.0f} / 확인 ${confirm:+,.0f}) | 최대낙폭 ${max_drawdown:,.0f} | "
        f"계약 {trades[0].quantities if trades else '-'} | 청산 {dict(reasons)} | 진입 막힘 {dict(blocked)}"
    )


if __name__ == "__main__":
    main()
