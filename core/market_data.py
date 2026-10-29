"""국내 차트 API에서 분봉과 일봉을 조회한다."""

import time

from core.bars import Bar

DAY_SESSION = "1"  # 주야간구분 1: 주간, 2: 야간


def fetch_minute_bars(client, symbol, count, interval_minutes=5):
    payload = {"nght_tp": DAY_SESSION, "sym": symbol, "dcnt": str(interval_minutes)}
    return fetch_chart(client, "/trade/v1/domestic/chart-minute", payload, count)


def fetch_daily_bars(client, symbol, count):
    payload = {"nght_tp": DAY_SESSION, "sym": symbol}
    return fetch_chart(client, "/trade/v1/domestic/chart-daily", payload, count)


def fetch_chart(client, path, payload, count):
    """최신 봉부터 과거 방향으로 count개를 조회하고, 오래된 순으로 돌려준다."""
    bars = {}
    next_key = "99999999"

    while len(bars) < count:
        # 2000개씩 요청하면 모의 서버에서 시간 초과가 나서 300개씩 요청한다
        body = client.post(path, {**payload, "req_qty": "300", "nxt_key": next_key})
        rows = body.get("Occurs1") or []
        if isinstance(rows, dict):  # 한 건이면 리스트가 아니라 객체로 온다
            rows = [rows]

        for row in rows:
            bar = to_bar(row)
            bars[(bar.date, bar.time)] = bar

        # 응답의 nxt_key를 다음 요청에 넣으면 더 과거 봉을 준다. 값이 바뀌지 않으면 조회를 종료한다.
        new_key = (body.get("nxt_key") or "").strip()
        if not rows or new_key in ("", next_key, "99999999"):
            break
        next_key = new_key
        time.sleep(0.2)  # 시세 API 호출 제한(초당 10건)에 여유를 둔다

    return [bars[key] for key in sorted(bars)[-count:]]


def to_bar(row):
    return Bar(
        date=row["occurs1_ref_dt"].strip(),
        time=row.get("occurs1_exch_tm", "").strip(),
        open=float(row["occurs1_open_pric"]),
        high=float(row["occurs1_high_pric"]),
        low=float(row["occurs1_low_pric"]),
        close=float(row["occurs1_close_pric"]),
        cum_volume=int(row.get("occurs1_cum_trd_qty") or 0),
        open_interest=int(row.get("occurs1_opint") or 0),
    )


def fetch_best_quote(client, symbol):
    """최우선 (매도호가, 매수호가)를 반환한다. 장 시작 전처럼 호가가 0이면 None."""
    body = client.post("/trade/v1/domestic/price-bid", {"exch_cd": "KSP", "sym": symbol})  # KSP: 주간
    levels = body.get("Occurs1") or []
    best = levels[0] if isinstance(levels, list) and levels else levels
    ask = float(best.get("occurs1_ask_best_pric") or 0) if best else 0
    bid = float(best.get("occurs1_bid_best_pric") or 0) if best else 0
    return (ask, bid) if ask > 0 and bid > 0 else None
