"""3년 국채선물 이동평균 교차 전략의 신호를 만든다.

백테스트와 실시간 실행에서 동일한 함수를 사용해 전략 규칙을 일치시킨다.
"""

import math

from core.indicators import sma

BUY = 1
SELL = -1


def crossover_signals(closes, short_window, long_window):
    """각 봉의 교차 신호를 반환한다.

    BUY는 상향 교차, SELL은 하향 교차, 0은 신호 없음을 뜻한다.
    직전 봉과 현재 봉 사이에서 두 이동평균의 상대적 위치가 바뀔 때만 신호가 발생한다.
    """
    short_sma = sma(closes, short_window)
    long_sma = sma(closes, long_window)
    signals = [0] * len(closes)

    for i in range(1, len(closes)):
        if long_sma[i - 1] is None:
            continue
        previous_gap = short_sma[i - 1] - long_sma[i - 1]
        current_gap = short_sma[i] - long_sma[i]
        if previous_gap <= 0 < current_gap:
            signals[i] = BUY
        elif previous_gap >= 0 > current_gap:
            signals[i] = SELL
    return signals


def daily_direction(daily_bars, date, window):
    """일봉으로 해당 거래일에 허용할 방향을 정한다.

    date 이전의 최근 종가가 최근 window개 일봉 종가 평균보다 높으면 BUY, 낮으면 SELL을 반환한다.
    평균과 같거나 일봉이 부족하면 0을 반환해 방향을 제한하지 않는다.
    """
    closes = [bar.close for bar in daily_bars if bar.date < date][-window:]
    if len(closes) < window:
        return 0

    average = sum(closes) / window
    if closes[-1] > average:
        return BUY
    if closes[-1] < average:
        return SELL
    return 0


def calculate_stop_ticks(atr_value, params, tick_size):
    if not atr_value:
        return 0
    # ATR × 배수를 틱 단위로 환산해 올림한 뒤 최소 손절 폭과 비교한다.
    # 올림 전에 소수점 아래 6자리로 반올림해 부동소수점 오차를 제거한다.
    atr_ticks = math.ceil(round(params["atr_multiplier"] * atr_value / tick_size, 6))
    return max(params["min_stop_ticks"], atr_ticks)


def can_enter(params, signal, bar_end, entries, day_ticks, stop_ticks, allowed_side):
    # 누적 손익에 손절 폭과 왕복 슬리피지를 보수적으로 반영해도 일일 한도 안일 때만 진입한다.
    return (
        params["entry_start"] <= bar_end <= params["entry_end"]
        and entries < params["max_entries_per_day"]
        and stop_ticks > 0
        and -day_ticks + stop_ticks + 2 * params["slippage_ticks"] <= params["daily_loss_limit_ticks"]
        and allowed_side in (0, signal)
    )
