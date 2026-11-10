"""MNQ-MGC 스프레드 되돌림 전략에 필요한 값을 계산한다.

백테스트와 실시간 실행에서 같은 함수를 사용한다.
스프레드 매수(side=1)는 1번 레그 매수, 2번 레그 매도다.
"""

import math
import statistics

BUY_SPREAD = 1
SELL_SPREAD = -1


def spread(price1, price2):
    return math.log(price1) - math.log(price2)


def zscore(spreads, window):
    """최근 window개 스프레드로 (z, 표준편차)를 구한다. 개수가 부족하면 None을 반환한다."""
    if len(spreads) < window:
        return None
    recent = spreads[-window:]
    std = statistics.stdev(recent)
    if std == 0:
        return None
    return (recent[-1] - statistics.fmean(recent)) / std, std


def contract_ratio(prices, point_values, tolerance):
    """두 레그의 명목금액 차이가 tolerance 이내인 가장 작은 계약 조합을 구한다."""
    value1, value2 = (price * point for price, point in zip(prices, point_values))
    for total in range(2, 30):
        for qty1 in range(1, total):
            qty2 = total - qty1
            amount1, amount2 = qty1 * value1, qty2 * value2
            if abs(amount1 - amount2) / max(amount1, amount2) <= tolerance:
                return qty1, qty2
    return None


def entry_side(z, entry_z):
    """스프레드가 평균보다 낮으면 매수, 높으면 매도를 반환한다. 그 외에는 0이다."""
    if z <= -entry_z:
        return BUY_SPREAD
    if z >= entry_z:
        return SELL_SPREAD
    return 0


def check_entry(z, std, notional, cost, params):
    """진입을 막는 사유를 반환한다. 막지 않으면 None이다.

    목표는 평균을 지나 반대쪽 target_z, 손절은 괴리가 stop_z까지 확대되는 것으로 본다.
    """
    if abs(z) >= params["stop_z"]:
        return "손절선 초과"
    reward = (abs(z) + params["target_z"]) * std * notional
    risk = (params["stop_z"] - abs(z)) * std * notional
    if params["cost_filter"] and reward < 2 * cost:
        return "비용"
    if risk > params["max_loss"]:
        return "손실 한도"
    return None


def pair_pnl(side, quantities, entry_prices, prices, point_values):
    """수수료를 빼기 전 손익(달러)을 구한다."""
    leg1 = quantities[0] * (prices[0] - entry_prices[0]) * point_values[0]
    leg2 = quantities[1] * (prices[1] - entry_prices[1]) * point_values[1]
    return side * (leg1 - leg2)


def round_trip_cost(quantities, fee_per_contract, slippage_ticks, tick_values):
    """진입과 청산을 합친 수수료와 슬리피지(달러)를 구한다."""
    fee = 2 * fee_per_contract * sum(quantities)
    slippage = 2 * slippage_ticks * sum(qty * value for qty, value in zip(quantities, tick_values))
    return fee + slippage


def exit_reason(side, z, gross, peak_gross, cost, held_minutes, params):
    """청산 사유를 반환한다. 계속 보유하면 None이다.

    peak_gross는 진입 후 가장 컸던 수수료 전 손익이다.
    """
    if side * z <= -params["stop_z"]:
        return "z 손절"
    if gross - cost <= -params["max_loss"]:
        return "금액 손절"
    if params["breakeven"] and peak_gross >= cost and gross <= 0:
        return "본전"
    if side * z >= params["target_z"]:
        return "익절"
    if held_minutes >= params["hold_minutes"]:
        return "시간 손절"
    return None
