"""4주차 스프레드 전략에 위험관리 규칙을 하나씩 더해 같은 기간, 같은 비용으로 비교한다.

신호는 1분봉 종가로 판단하고 다음 분 시가에 체결한다(4주차와 같음).
실시간 손절은 체결가를 볼 수 없어서, 그 분의 고가와 저가 중 포지션에 가장 불리한 값으로 근사한다.
"""

import argparse
import csv
import logging
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from core.bars import Bar, add_minutes, load_bars
from core.config import DATA_DIR, load_json
from week04.backtest import minutes_between
from week04.strategy import check_entry, contract_ratio, entry_side, exit_reason, pair_pnl, round_trip_cost, spread, zscore
from week06.risk import expected_move, half_life, leg_volatility, worst_prices

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).with_name("results")
STOP_REASONS = ("z 손절", "금액 손절", "실시간 손절")


@dataclass
class Trade:
    date: str
    side: int
    entry_time: str
    exit_time: str
    reason: str
    gross: float
    net: float
    half_life: float
    expected_move: float


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--slippage", type=int, help="config의 slippage_ticks 대신 사용할 값")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    config = load_json(args.config)
    settings = config["backtest"]
    slippage = settings["slippage_ticks"] if args.slippage is None else args.slippage

    legs = [load_leg(leg["symbol"], settings) for leg in config["legs"]]
    days, dropped = trading_days(legs, config["session"])
    logger.info(f"기간 {days[0]} ~ {days[-1]}, {len(days)}일, 슬리피지 {slippage}틱, 15~18시 봉이 부족해 뺀 날 {dropped or '없음'}")
    volatility = past_volatility(legs, days, config)

    results = {}
    for name, overrides in config["versions"].items():
        rules = {**config["rules"], **overrides}
        results[name] = run_backtest(legs, days, volatility, rules, config, slippage)

    rows = summarize(results, days, config)
    print_summary(rows, settings["periods"])
    judge(results, rows, days, config)
    filter_value(results["base"], config)
    stop_by_trend(results, legs, days)
    save(results, rows)


def load_leg(symbol, settings):
    """4주차 파일과 6주차 파일을 합쳐 날짜별 {봉 시작 시각: 봉}을 만든다. 겹치면 6주차 파일을 쓴다."""
    merged = {}
    for path in (DATA_DIR / f"{symbol}_1min.csv", DATA_DIR / "week06" / f"{symbol}_1min.csv"):
        if path.exists():
            for bar in load_bars(path):
                merged[(bar.date, bar.time)] = bar
    by_date = defaultdict(dict)
    for (date, time), bar in merged.items():
        if settings["start_date"] <= date <= settings["end_date"]:
            by_date[date][time] = bar
    return by_date


def trading_days(legs, session):
    """두 레그 모두 진입 시간대 봉이 150개 이상인 날만 쓴다."""
    days, dropped = [], []
    for date in sorted(set(legs[0]) & set(legs[1])):
        counts = [sum(session["entry_start"] <= time < "180000" for time in leg[date]) for leg in legs]
        (days if min(counts) >= 150 else dropped).append(date)
    return days, dropped


def past_volatility(legs, days, config):
    """날짜별로 직전 5거래일 15:00~17:55 1분 수익률의 (σ1, σ2, ρ)를 구한다."""
    session, count = config["session"], config["risk"]["vol_days"]
    result = {}
    for k, date in enumerate(days):
        if k < count:
            continue
        returns = ([], [])
        for past in days[k - count:k]:
            times = sorted(t for t in legs[0][past] if session["entry_start"] <= t <= session["close_time"] and t in legs[1][past])
            for before, after in zip(times, times[1:]):
                for i in (0, 1):
                    returns[i].append(math.log(legs[i][past][after].close / legs[i][past][before].close))
        result[date] = leg_volatility(*returns)
    return result


def minute_bars(bars, times):
    """각 분의 봉을 만든다. 봉이 없는 분은 직전 종가로 채운 봉이다."""
    earlier = [time for time in bars if time < times[0]]
    last = bars[max(earlier)] if earlier else None
    result = []
    for time in times:
        bar = bars.get(time)
        if bar:
            last = bar
        elif last:
            bar = Bar(last.date, time, last.close, last.close, last.close, last.close, 0, 0)
        result.append(bar)
    return result


def run_backtest(legs, days, volatility, rules, config, slippage):
    params, risk, session = {**config["strategy"], **config["session"]}, config["risk"], config["session"]
    point_values = [leg["point_value"] for leg in config["legs"]]
    tick_sizes = [leg["tick_size"] for leg in config["legs"]]
    tick_values = [size * point for size, point in zip(tick_sizes, point_values)]
    fee = config["fee_per_contract"]
    lookback = max(params["window"], risk["half_life_window"]) + 5

    trades, blocked, daily = [], Counter(), {}
    for k, date in enumerate(days):
        if rules["multi_day"] and k >= risk["multi_day_count"]:
            if sum(daily[d] for d in days[k - risk["multi_day_count"]:k]) <= -risk["multi_day_loss"]:
                blocked["3일 누적 한도"] += 1
                daily[date] = 0.0
                continue

        first = add_minutes(session["entry_start"], -lookback)
        times = [add_minutes(first, i) for i in range(minutes_between(first, session["close_time"]) + 2)]
        bars = [minute_bars(leg[date], times) for leg in legs]
        start = times.index(session["entry_start"])
        closes = [bars[0][start].close, bars[1][start].close]
        quantities = contract_ratio(closes, point_values, params["size_tolerance"])
        notional = sum(q * c * p for q, c, p in zip(quantities, closes, point_values)) / 2
        cost = round_trip_cost(quantities, fee, slippage, tick_values)
        fees = 2 * fee * sum(quantities)
        slip_cost = slippage * sum(q * v for q, v in zip(quantities, tick_values))

        spreads, position, entries, day_net, losses, cooldown_until, done = [], None, 0, 0.0, 0, None, False
        for i, time in enumerate(times[:-1]):
            bar1, bar2 = bars[0][i], bars[1][i]
            if bar1 is None or bar2 is None:
                continue
            spreads.append(spread(bar1.close, bar2.close))
            if time < session["entry_start"] or done:
                continue
            nxt1, nxt2 = bars[0][i + 1] or bar1, bars[1][i + 1] or bar2
            result = zscore(spreads, params["window"])

            if position:
                side = position["side"]
                exit_net = None
                if rules["live_stop"]:
                    worst = pair_pnl(side, quantities, position["prices"], worst_prices(side, bar1, bar2), point_values)
                    if worst - fees <= -risk["live_stop"]:
                        reason, exit_net = "실시간 손절", -risk["live_stop"] - slip_cost
                if exit_net is None:
                    gross = pair_pnl(side, quantities, position["prices"], (bar1.close, bar2.close), point_values)
                    held = minutes_between(position["time"], time)
                    reason = "마감" if time >= session["close_time"] else None
                    if not reason and rules["daily_flatten"] and day_net + gross - fees <= -params["daily_loss"]:
                        reason = "하루 한도"
                    if not reason and result:
                        reason = exit_reason(side, result[0], gross, 0.0, cost, held, params)
                    if reason:
                        exit_prices = [nxt1.open - side * slippage * tick_sizes[0], nxt2.open + side * slippage * tick_sizes[1]]
                        exit_net = pair_pnl(side, quantities, position["prices"], exit_prices, point_values) - fees
                if exit_net is not None:
                    trades.append(Trade(date, side, position["time"], time, reason, exit_net + fees, exit_net,
                                        position["half_life"], position["move"]))
                    day_net += exit_net
                    losses = losses + 1 if exit_net < 0 else 0
                    position = None
                    if reason in STOP_REASONS:
                        cooldown_until = add_minutes(time, params["cooldown_minutes"])
                    if reason == "하루 한도" or (rules["daily_flatten"] and day_net <= -params["daily_loss"]):
                        done = True
                    if rules["loss_streak"] and losses >= risk["max_loss_streak"]:
                        blocked["연속 손실 중단"] += 1
                        done = True
                continue

            if not result or time >= session["entry_end"] or entries >= params["max_entries_per_day"]:
                continue
            if day_net <= -params["daily_loss"] or (cooldown_until and time < cooldown_until):
                continue
            side = entry_side(result[0], params["entry_z"])
            if not side:
                continue
            block = check_entry(result[0], result[1], notional, cost, params)
            life = half_life(spreads[-risk["half_life_window"]:])
            move = expected_move((bar1.close, bar2.close), quantities, point_values, volatility[date], risk["vol_hold_minutes"]) if date in volatility else math.nan
            if not block and rules["half_life"] and life > risk["max_half_life"]:
                block = "반감기"
            if not block and rules["vol_cap"] and move > risk["max_expected_move"]:
                block = "변동성 상한"
            if block:
                blocked[block] += 1
                continue
            entry_prices = [nxt1.open + side * slippage * tick_sizes[0], nxt2.open - side * slippage * tick_sizes[1]]
            position = {"side": side, "prices": entry_prices, "time": time, "half_life": life, "move": move}
            entries += 1
        daily[date] = day_net

    return {"trades": trades, "blocked": blocked, "daily": daily}


def in_period(date, period):
    return period[0] <= date <= period[1]


def metrics(trades, daily, days, max_loss):
    nets = [trade.net for trade in trades]
    equity = peak = drawdown = 0.0
    for date in days:
        equity += daily[date]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    day_values = sorted(daily[date] for date in days)
    worst_count = max(1, math.ceil(len(day_values) * 0.05))
    streak = longest = 0
    for net in nets:
        streak = streak + 1 if net < 0 else 0
        longest = max(longest, streak)
    return {
        "trades": len(trades),
        "win_rate": sum(net > 0 for net in nets) / len(nets) if nets else 0.0,
        "net": sum(nets),
        "max_drawdown": drawdown,
        "worst_day": day_values[0],
        "worst5_avg": statistics.fmean(day_values[:worst_count]),
        "loss_streak": longest,
        "worst_trade": min(nets, default=0.0),
        "over_limit": sum(net < -max_loss for net in nets),
    }


def summarize(results, days, config):
    rows = []
    periods = {"전체": (days[0], days[-1]), **config["backtest"]["periods"]}
    for name, result in results.items():
        for label, period in periods.items():
            period_days = [d for d in days if in_period(d, period)]
            trades = [t for t in result["trades"] if in_period(t.date, period)]
            row = {"version": name, "period": label, **metrics(trades, result["daily"], period_days, config["strategy"]["max_loss"])}
            if label == "전체":
                row["reasons"] = dict(Counter(t.reason for t in result["trades"]))
                row["blocked"] = dict(result["blocked"])
            rows.append(row)
    return rows


def print_summary(rows, periods):
    for label in ["전체", *periods]:
        logger.info(f"\n[{label}]")
        logger.info(f"{'버전':<17}{'거래':>4}{'승률':>6}{'순손익':>9}{'최대낙폭':>9}{'최악일':>8}{'하위5%일':>9}{'연속손실':>6}{'최악거래':>8}{'한도초과':>6}")
        for row in rows:
            if row["period"] != label:
                continue
            logger.info(
                f"{row['version']:<17}{row['trades']:>4}{row['win_rate']:>6.2f}{row['net']:>+9.0f}{row['max_drawdown']:>9.0f}"
                f"{row['worst_day']:>+8.0f}{row['worst5_avg']:>+9.0f}{row['loss_streak']:>6}{row['worst_trade']:>+8.0f}{row['over_limit']:>6}"
            )
    logger.info("\n[청산 사유와 진입 막은 사유, 전체 기간]")
    for row in rows:
        if row["period"] == "전체":
            logger.info(f"{row['version']:<17} 청산 {row['reasons']} | 막음 {row['blocked']}")


def judge(results, rows, days, config):
    """미리 정한 판정 기준. 낙폭과 순손익 기준이 본 판정이고 p값은 참고용이다."""
    periods = config["backtest"]["periods"]
    falling, fitted = list(periods)[2], list(periods)[1]

    def pick(version, period):
        return next(row for row in rows if row["version"] == version and row["period"] == period)

    logger.info("\n[판정: 11/10~11/20 최대 낙폭 30% 이상 감소, 9/30~11/9 순손익 감소가 base의 30% 이내]")
    base_dd, base_net = pick("base", falling)["max_drawdown"], pick("base", fitted)["net"]
    for name in results:
        if name == "base":
            continue
        dd, net = pick(name, falling)["max_drawdown"], pick(name, fitted)["net"]
        dd_ok = dd <= base_dd * 0.7
        net_ok = base_net - net <= abs(base_net) * 0.3
        gain, p = bootstrap_p(results["base"]["daily"], results[name]["daily"], days, config["backtest"])
        logger.info(
            f"{name:<17} 낙폭 {base_dd:.0f} → {dd:.0f} ({'통과' if dd_ok else '미달'}) | "
            f"순손익 {base_net:+.0f} → {net:+.0f} ({'통과' if net_ok else '미달'}) | "
            f"하위 25% 날 평균 개선 ${gain:+.0f}, p {p:.2f} (참고)"
        )


def lower_quarter_mean(values):
    values = sorted(values)
    return statistics.fmean(values[: max(1, len(values) // 4)])


def bootstrap_p(base_daily, daily, days, settings):
    """각 버전 자신의 하위 25% 날 평균 손익이 base보다 나아졌는지, 같은 날짜를 5일 블록으로 다시 뽑아 본다.

    base가 나빴던 날만 고르면 어떤 규칙이든 그날은 좋아 보이므로, 날짜 선택을 버전마다 따로 한다.
    p는 다시 뽑은 표본에서 개선이 0 이하인 비율(단측)이다.
    """
    observed = lower_quarter_mean([daily[d] for d in days]) - lower_quarter_mean([base_daily[d] for d in days])
    block, rng, n, fails = settings["bootstrap_block"], random.Random(0), len(days), 0
    for _ in range(settings["bootstrap_samples"]):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample += [days[(start + j) % n] for j in range(block)]
        sample = sample[:n]
        gain = lower_quarter_mean([daily[d] for d in sample]) - lower_quarter_mean([base_daily[d] for d in sample])
        fails += gain <= 0
    return observed, fails / settings["bootstrap_samples"]


def filter_value(base, config):
    """진입 필터가 base 거래 중 막았을 거래와 그 손익, 손실 거래 구분력(AUC)을 구한다."""
    risk = config["risk"]
    logger.info("\n[진입 필터가 base 거래에 적용됐다면]")
    for label, blocks, score in (
        ("반감기", lambda t: t.half_life > risk["max_half_life"], lambda t: min(t.half_life, 1e6)),
        ("변동성 상한", lambda t: t.expected_move > risk["max_expected_move"], lambda t: t.expected_move),
    ):
        trades = [t for t in base["trades"] if not math.isnan(score(t))]
        hit = [t for t in trades if blocks(t)]
        logger.info(
            f"{label:<8} 막을 거래 {len(hit)}/{len(trades)} 손익 합 ${sum(t.net for t in hit):+.0f} | "
            f"손실 거래 구분력 AUC {auc([score(t) for t in trades], [t.net < 0 for t in trades]):.2f}"
        )


def auc(scores, labels):
    positives = [s for s, label in zip(scores, labels) if label]
    negatives = [s for s, label in zip(scores, labels) if not label]
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def stop_by_trend(results, legs, days):
    """실시간 손절이 아낀 돈이 추세가 강한 날에 몰리는지 본다(Kaminski & Lo 2014 비교용).

    추세 강도 = 15~18시 스프레드 시작과 끝 차이 ÷ 그 시간 1분 변화의 표준편차.
    """
    strength = {}
    for date in days:
        times = sorted(t for t in legs[0][date] if "150000" <= t < "180000" and t in legs[1][date])
        values = [spread(legs[0][date][t].close, legs[1][date][t].close) for t in times]
        changes = [after - before for before, after in zip(values, values[1:])]
        strength[date] = abs(values[-1] - values[0]) / statistics.pstdev(changes)
    cut = sorted(strength.values())[int(len(days) * 0.75)]
    base, stop = results["base"]["daily"], results["R2_live_stop"]["daily"]
    trend = [d for d in days if strength[d] >= cut]
    rest = [d for d in days if strength[d] < cut]
    logger.info("\n[실시간 손절 효과: 추세 강한 날(상위 25%) vs 나머지]")
    for label, group in (("추세 강한 날", trend), ("나머지", rest)):
        diff = sum(stop[d] - base[d] for d in group)
        logger.info(f"{label:<7} {len(group)}일, R2 − base 합 ${diff:+.0f}, 하루 평균 ${diff / len(group):+.1f}")


def save(results, rows):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "backtest_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = [key for key in rows[0] if key not in ("reasons", "blocked")]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS_DIR / "backtest_trades.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["version", *Trade.__dataclass_fields__])
        writer.writeheader()
        for name, result in results.items():
            for trade in result["trades"]:
                writer.writerow({"version": name, **asdict(trade)})


if __name__ == "__main__":
    main()
