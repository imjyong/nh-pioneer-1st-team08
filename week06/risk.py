"""6주차에 추가한 위험관리 계산. 백테스트와 실시간 실행에서 같은 함수를 쓴다."""

import math
import statistics


def half_life(spreads):
    """Δs(t) = a + b·s(t−1) 회귀로 반감기(분)를 구한다. b ≥ 0이면 돌아오는 힘이 없어 inf다."""
    x = spreads[:-1]
    y = [after - before for before, after in zip(spreads[:-1], spreads[1:])]
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((v - mean_x) ** 2 for v in x)
    if sxx == 0:
        return math.inf
    b = sum((u - mean_x) * (w - mean_y) for u, w in zip(x, y)) / sxx
    if b >= 0:
        return math.inf
    return -math.log(2) / math.log(1 + b)


def leg_volatility(returns1, returns2):
    return statistics.pstdev(returns1), statistics.pstdev(returns2), statistics.correlation(returns1, returns2)


def expected_move(prices, quantities, point_values, volatility, minutes):
    """포지션을 45분 들고 있을 때 예상 변동액(달러)을 구한다.

    반대 방향 포지션이라 두 레그가 같이 움직이는 만큼 상쇄된다(−2ρAB).
    """
    sigma1, sigma2, rho = volatility
    scale = math.sqrt(minutes)
    a = prices[0] * point_values[0] * quantities[0] * sigma1 * scale
    b = prices[1] * point_values[1] * quantities[1] * sigma2 * scale
    return math.sqrt(max(a * a + b * b - 2 * rho * a * b, 0.0))


def psi(values, reference, bins=10):
    """기준 기간(9/30 ~ 10/27)의 10분위 구간으로 나눠, 새 기간 분포가 얼마나 달라졌는지 PSI를 구한다."""
    ordered = sorted(reference)
    edges = [ordered[int(len(ordered) * i / bins)] for i in range(1, bins)]

    def shares(data):
        counts = [0] * bins
        for value in data:
            counts[sum(value > edge for edge in edges)] += 1
        return [max(count / len(data), 1e-4) for count in counts]

    return sum((a - b) * math.log(a / b) for a, b in zip(shares(values), shares(reference)))


def worst_prices(side, bar1, bar2):
    """1분 안에서 포지션에 가장 불리한 두 레그 가격을 고른다. 스프레드 매수면 1번 레그 저가, 2번 레그 고가다."""
    if side == 1:
        return bar1.low, bar2.high
    return bar1.high, bar2.low
