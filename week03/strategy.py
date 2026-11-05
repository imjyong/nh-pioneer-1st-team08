"""MNQ 개장 전 범위 돌파 전략에 필요한 신호와 가격을 계산한다.

백테스트와 실시간 실행에서 같은 함수를 사용한다.
"""

import math

from core.indicators import sma

BUY = 1
SELL = -1


def minutes_since(start_time, current_time):
    """두 시각의 차이를 분 단위로 계산한다. 자정을 넘으면 다음 날로 본다."""
    start_minutes = int(start_time[:2]) * 60 + int(start_time[2:4])
    current_minutes = int(current_time[:2]) * 60 + int(current_time[2:4])
    return (current_minutes - start_minutes) % (24 * 60)


def opening_range(bars, params, interval_minutes):
    """한 세션의 봉에서 지정한 구간의 고가와 저가를 구한다.

    봉이 부족하면 None을 반환한다. bar.time은 봉 시작 시각으로 본다.
    """
    range_bars = [bar for bar in bars if minutes_since(params["range_start"], bar.time) < params["range_minutes"]]
    if len(range_bars) < params["range_minutes"] // interval_minutes:
        return None
    return max(bar.high for bar in range_bars), min(bar.low for bar in range_bars)


def breakout_signal(close_price, range_high, range_low, vwap_value):
    """종가가 범위와 VWAP을 모두 넘으면 BUY, 모두 밑돌면 SELL을 반환한다. 그 외에는 0이다."""
    if vwap_value is None:
        return 0
    if close_price > range_high and close_price > vwap_value:
        return BUY
    if close_price < range_low and close_price < vwap_value:
        return SELL
    return 0


def sma_direction(closes, window):
    """SMA가 직전 봉보다 오르면 BUY, 내리면 SELL을 반환한다. 그 외에는 0이다."""
    values = sma(closes, window)
    if len(values) < 2 or values[-2] is None:
        return 0
    if values[-1] > values[-2]:
        return BUY
    if values[-1] < values[-2]:
        return SELL
    return 0


def calculate_stop_price(side, range_high, range_low, tick_size):
    # 중간값이 호가 단위와 맞지 않으면 손절 폭이 넓어지도록 조정한다.
    midpoint_ticks = (range_high + range_low) / 2 / tick_size
    stop_ticks = math.floor(midpoint_ticks) if side == BUY else math.ceil(midpoint_ticks)
    return stop_ticks * tick_size


def calculate_target_price(side, entry_price, stop_price, params):
    return entry_price + side * params["target_multiple"] * abs(entry_price - stop_price)


def reached_breakeven(side, entry_price, initial_stop_price, current_price):
    """현재 이익 폭이 처음 정한 손절 폭 이상인지 확인한다."""
    return side * (current_price - entry_price) >= abs(entry_price - initial_stop_price)


def can_enter(params, signal, bar_end, entries, stop_ticks, allowed_side):
    # 개장 범위가 확정된 다음 봉부터 진입한다.
    elapsed = minutes_since(params["range_start"], bar_end)
    entry_end = minutes_since(params["range_start"], params["entry_end"])
    return (
        signal != 0
        and params["range_minutes"] < elapsed <= entry_end
        and entries < params["max_entries_per_day"]
        and 0 < stop_ticks <= params["max_stop_ticks"]
        and allowed_side in (0, signal)
    )
