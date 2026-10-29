"""이동평균과 ATR을 계산한다."""


def sma(values, window):
    """단순 이동평균을 계산한다. 처음 window - 1개 위치는 None으로 둔다."""
    result = []
    for i in range(len(values)):
        if i + 1 < window:
            result.append(None)
        else:
            result.append(sum(values[i + 1 - window : i + 1]) / window)
    return result


def true_ranges(bars):
    """각 봉의 TR을 계산한다.

    TR은 고가-저가, |고가-직전 종가|, |저가-직전 종가| 중 가장 큰 값이다.
    첫 봉은 직전 종가가 없으므로 고가와 저가의 차이를 사용한다.
    """
    result = [bars[0].high - bars[0].low]
    for prev, bar in zip(bars, bars[1:]):
        result.append(max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close)))
    return result


def atr(bars, window):
    """최근 window개 TR의 산술평균으로 ATR을 계산한다."""
    return sma(true_ranges(bars), window)
