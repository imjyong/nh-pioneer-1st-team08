"""MNQ 개장 전 범위 돌파 전략을 모의계좌에서 실행한다."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from core.bars import Bar, add_minutes
from core.client import create_client
from core.config import load_env, load_json
from core.indicators import vwap
from core.market_data import fetch_chart
from week03.broker import FILLED, MARKET, REJECTED, STOP, OverseasBroker
from week03.stream import OverseasStream
from week03.strategy import (
    breakout_signal,
    calculate_stop_price,
    calculate_target_price,
    can_enter,
    minutes_since,
    opening_range,
    reached_breakeven,
    sma_direction,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).with_name("results")
SIDE_NAME = {1: "매수", -1: "매도"}


class EventLog:
    """실행 로그와 JSONL 이벤트 기록을 함께 저장한다."""

    def __init__(self, run_id):
        self.run_id = run_id
        RESULTS_DIR.mkdir(exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / f"live_{run_id}.log", encoding="utf-8"),
        ]
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self.jsonl = open(RESULTS_DIR / f"events_{run_id}.jsonl", "a", encoding="utf-8")

    def write(self, level, event, **fields):
        message = " | ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(getattr(logging, level), f"[{event}] {message}")
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "level": level,
            "event": event,
            "data": fields,
        }
        self.jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.jsonl.flush()

    def close(self):
        self.jsonl.close()
        for handler in logger.handlers:
            handler.close()


class LiveTrader:
    def __init__(self, config, client, broker, stream, log):
        self.config = config
        self.live = config["live"]
        self.params = {**config["strategy"], **config["versions"][config["selected_version"]]}
        self.tick = config["tick_size"]
        self.interval = config["interval_minutes"]
        self.client = client
        self.broker = broker
        self.stream = stream
        self.log = log

        self.today = datetime.now().strftime("%Y%m%d")
        self.bars = []
        self.building = None  # 체결로 만들고 있는 봉
        self.closed_bars = []  # 마감했지만 아직 판단하지 않은 봉
        self.needs_reload = True  # 접속 직후 봉은 일부 체결만 담겨 있어 REST로 다시 받는다
        self.last_price = None
        self.last_tick_time = "-"
        self.last_tick_at = time.time()
        self.fills = {}  # 주문번호: (상태, 체결가, 오류코드)
        self.position = None  # 방향, 진입가, 처음 손절가, 현재 손절가, 익절가, 손절 주문번호
        self.entries = 0
        self.day_ticks = 0
        self.day_closed = False
        self.range_logged = False
        self.last_status = None
        self.last_exit_try = 0

    def run(self):
        self.prepare()
        while datetime.now().strftime("%H%M%S") < self.live["final_check_time"]:
            try:
                self.read_stream()
                self.process_closed_bars()
                if self.position:
                    self.check_exit()
                self.write_status()
            except Exception as error:
                self.log.write("ERROR", "오류", 단계="실행", 원인=type(error).__name__, 내용=str(error)[:300], 조치="계속")
                time.sleep(1)
        self.finish()

    def prepare(self):
        self.log.write(
            "INFO", "시작",
            종목=f"{self.config['name']}({self.config['symbol']})",
            브로커="mock",
            전략=self.config["selected_version"],
            범위=f"{self.params['range_start']}부터 {self.params['range_minutes']}분",
            진입종료=self.params["entry_end"],
            청산=self.params["close_time"],
        )
        deposit = self.broker.get_deposit()
        position = self.broker.get_position()
        open_orders = self.broker.count_open_orders()
        if position or open_orders:
            self.log.write("ERROR", "오류", 단계="계좌 동기화", 포지션=position or "없음", 미체결=f"{open_orders}건", 조치="전략 시작 안 함")
            raise SystemExit(1)
        self.log.write("INFO", "계좌 동기화", 예탁자산=f"${deposit:,.2f}", 포지션="없음", 미체결="0건")
        self.stream.connect()
        self.reload_bars()
        self.needs_reload = True
        self.log.write("INFO", "준비", 분봉=f"{len(self.bars)}개", 수신="WebSocket")

    def reload_bars(self):
        """오늘 VWAP 시작 이후의 마감된 봉을 REST로 다시 받는다."""
        payload = {"exch_cd": self.config["exchange"], "sym": self.config["symbol"], "dcnt": str(self.interval)}
        bars = fetch_chart(self.client, "/trade/v1/overseas/chart-minute", payload, self.live["warmup_bars"])
        current_bar = self.bar_start(datetime.now().strftime("%H%M%S"))
        self.bars = [
            bar for bar in bars
            if bar.date == self.today and self.params["vwap_start"] <= bar.time < current_bar
        ]
        self.needs_reload = False

    def bar_start(self, hhmmss):
        minute = int(hhmmss[2:4]) // self.interval * self.interval
        return f"{hhmmss[:2]}{minute:02d}00"

    def read_stream(self, timeout=1):
        try:
            message = self.stream.read(timeout)
        except Exception as error:
            self.reconnect(type(error).__name__)
            return
        if message and message[0] == "FA":
            self.on_tick(message[1])
        elif message and message[0] == "O1":
            self.on_order(message[1])
        if time.time() - self.last_tick_at > self.live["max_silence_seconds"]:
            self.reconnect("시세 수신 없음")

    def reconnect(self, reason):
        # 끊긴 동안 서버의 손절 주문은 유지된다. 봉은 다음 마감 때 REST로 다시 받는다.
        self.log.write("WARNING", "재접속", 사유=reason)
        self.stream.close()
        self.building = None
        self.needs_reload = True
        self.last_tick_at = time.time()
        try:
            self.stream.connect()
        except Exception as error:
            self.log.write("ERROR", "재접속 실패", 원인=type(error).__name__)
            time.sleep(3)

    def on_tick(self, body):
        self.last_price = float(body["last_pric"])
        self.last_tick_time = body["ko_exec_tm"]
        self.last_tick_at = time.time()
        start = self.bar_start(body["ko_exec_tm"])
        volume = int(body["exec_qty"] or 0)
        if self.building and self.building.time != start:
            self.closed_bars.append(self.building)
            self.building = None
        if not self.building:
            self.building = Bar(body["ko_trd_dt"], start, self.last_price, self.last_price, self.last_price, self.last_price, volume, 0)
            return
        bar = self.building
        self.building = Bar(
            bar.date, bar.time, bar.open, max(bar.high, self.last_price), min(bar.low, self.last_price),
            self.last_price, bar.cum_volume + volume, 0,
        )

    def on_order(self, body):
        order_no = body["ord_no"].strip()
        status = body["ord_sts"].strip()
        self.fills[order_no] = (status, float(body["exec_pric"] or 0), body["err_cd"].strip())
        if status == REJECTED:
            self.log.write("ERROR", "주문 응답", 주문번호=order_no, 상태="거부", 오류코드=body["err_cd"].strip())
        elif self.position and order_no == self.position["stop_order"] and status == FILLED:
            self.record_exit("손절" if not self.position["breakeven"] else "본전", self.fills[order_no][1])

    def process_closed_bars(self):
        while self.closed_bars:
            bar = self.closed_bars.pop(0)
            if self.needs_reload:
                try:
                    self.reload_bars()
                except Exception as error:
                    self.log.write("ERROR", "오류", 단계="분봉 조회", 내용=str(error)[:300])
                    continue
            elif bar.date == self.today:
                self.bars.append(bar)
            if self.bars and self.bars[-1].time == bar.time:
                self.on_bar(self.bars[-1])

    def on_bar(self, bar):
        price_range = opening_range(self.bars, self.params, self.interval)
        if price_range and not self.range_logged:
            self.range_logged = True
            self.log.write("INFO", "범위 확정", 고점=f"{price_range[0]:.2f}", 저점=f"{price_range[1]:.2f}")
        bar_end = add_minutes(bar.time, self.interval)
        elapsed = minutes_since(self.params["range_start"], bar_end)
        entry_end = minutes_since(self.params["range_start"], self.params["entry_end"])
        if not price_range or self.position or self.day_closed or not self.params["range_minutes"] < elapsed <= entry_end:
            return

        vwap_value = vwap(self.bars, [item.cum_volume for item in self.bars])[-1]
        signal = breakout_signal(bar.close, *price_range, vwap_value)
        self.log.write(
            "INFO", "판단",
            봉=f"{bar.time[:4]}~{bar_end[:4]}",
            종가=f"{bar.close:.2f}",
            범위=f"{price_range[1]:.2f}~{price_range[0]:.2f}",
            VWAP=f"{vwap_value:.2f}" if vwap_value else "없음",
            신호=SIDE_NAME.get(signal, "없음"),
        )
        if signal == 0:
            return
        stop_price = calculate_stop_price(signal, *price_range, self.tick)
        stop_ticks = round(abs(bar.close - stop_price) / self.tick)
        if stop_ticks > self.params["max_stop_ticks"]:
            self.day_closed = True
            self.log.write("WARNING", "거래 안 함", 사유="손절 폭 한도 초과", 손절폭=f"{stop_ticks}틱", 한도=f"{self.params['max_stop_ticks']}틱")
            return
        allowed_side = sma_direction([item.close for item in self.bars], self.params["sma_window"]) if self.params["sma_filter"] else 0
        if can_enter(self.params, signal, bar_end, self.entries, stop_ticks, allowed_side):
            self.enter(signal, stop_price)

    def enter(self, side, stop_price):
        self.log.write("INFO", "진입 주문", 방향=SIDE_NAME[side], 수량=1, 유형="시장가")
        order_no = self.submit(side, MARKET, None, "진입")
        fill_price = self.wait_fill(order_no) if order_no else None
        if fill_price is None:
            self.day_closed = True
            self.log.write("ERROR", "오류", 단계="진입 체결", 주문번호=order_no or "없음", 조치="그날 거래 중단")
            return
        self.entries += 1
        target_price = calculate_target_price(side, fill_price, stop_price, self.params)
        self.position = {
            "side": side, "entry": fill_price, "first_stop": stop_price, "stop": stop_price,
            "target": target_price, "stop_order": None, "breakeven": False,
        }
        self.log.write("INFO", "진입 체결", 방향=SIDE_NAME[side], 가격=f"{fill_price:.2f}", 손절=f"{stop_price:.2f}", 익절=f"{target_price:.2f}")
        self.position["stop_order"] = self.submit(-side, STOP, stop_price, "손절")
        if not self.position["stop_order"]:
            self.exit_position("손절 주문 실패")

    def check_exit(self):
        now = datetime.now().strftime("%H%M%S")
        position = self.position
        if now >= self.params["close_time"]:
            self.exit_position("시간 청산")
        elif self.last_price is not None and position["side"] * (self.last_price - position["target"]) >= 0:
            self.exit_position("익절")
        elif (
            not position["breakeven"] and self.last_price is not None
            and reached_breakeven(position["side"], position["entry"], position["first_stop"], self.last_price)
        ):
            self.move_stop_to_entry()

    def move_stop_to_entry(self):
        position = self.position
        position["breakeven"] = True
        self.cancel_stop()
        if not self.position or position["stop_order"]:  # 그사이 손절됐거나 취소에 실패했으면 원래 손절을 유지한다
            return
        if not self.broker.get_position():
            self.record_stop_filled()
            return
        position["stop"] = position["entry"]
        position["stop_order"] = self.submit(-position["side"], STOP, position["entry"], "본전 손절")
        self.log.write("INFO", "손절 이동", 손절=f"{position['entry']:.2f}", 현재가=f"{self.last_price:.2f}")
        if not position["stop_order"]:
            self.exit_position("손절 이동 실패")

    def exit_position(self, reason):
        # 실패하면 다음 반복에서 3초 간격으로 다시 시도한다.
        if time.time() - self.last_exit_try < 3:
            return
        self.last_exit_try = time.time()
        self.cancel_stop()
        if not self.position or self.position["stop_order"]:  # 손절이 살아 있을 때 청산하면 반대 포지션이 생길 수 있다
            return
        if not self.broker.get_position():
            self.record_stop_filled()
            return
        self.log.write("INFO", "청산 주문", 사유=reason, 방향=SIDE_NAME[-self.position["side"]], 수량=1, 유형="시장가")
        order_no = self.submit(-self.position["side"], MARKET, None, "청산")
        fill_price = self.wait_fill(order_no) if order_no else None
        if fill_price is None:
            self.log.write("ERROR", "오류", 단계="청산 체결", 주문번호=order_no or "없음", 조치="재시도")
            return
        self.record_exit(reason, fill_price)

    def cancel_stop(self):
        position = self.position
        if not position["stop_order"]:
            return
        stop_order, position["stop_order"] = position["stop_order"], None
        try:
            self.broker.cancel_order(stop_order, -position["side"], STOP)
            self.log.write("INFO", "손절 주문 취소", 주문번호=stop_order)
        except Exception as error:
            self.log.write("WARNING", "손절 주문 취소 실패", 주문번호=stop_order, 내용=str(error)[:300])
            position["stop_order"] = stop_order
        deadline = time.time() + 1
        while self.position and time.time() < deadline:
            self.read_stream(0.2)

    def record_stop_filled(self):
        # 취소 직전에 손절이 체결된 경우다. 체결가는 잔고가 비어 있어 주문내역에서 찾는다.
        order = self.broker.get_order(self.position["stop_order"]) if self.position["stop_order"] else None
        price = order["price"] if order and order["price"] else self.position["stop"]
        self.record_exit("손절" if not self.position["breakeven"] else "본전", price)

    def record_exit(self, reason, price):
        position, self.position = self.position, None
        ticks = round(position["side"] * (price - position["entry"]) / self.tick)
        self.day_ticks += ticks
        self.log.write(
            "INFO", "청산 체결",
            사유=reason,
            진입가=f"{position['entry']:.2f}",
            청산가=f"{price:.2f}",
            손익=f"{ticks:+d}틱(${ticks * self.config['tick_value']:+,.2f})",
            수수료="별도",
        )

    def submit(self, side, order_type, price, purpose):
        try:
            order_no = self.broker.place_order(side, order_type, price)
        except Exception as error:
            self.log.write("ERROR", "주문 응답", 용도=purpose, 상태="실패", 원인=type(error).__name__, 내용=str(error)[:300])
            return None
        self.log.write("INFO", "주문 응답", 용도=purpose, 상태="접수", 주문번호=order_no)
        return order_no

    def wait_fill(self, order_no):
        """체결가를 반환한다. 시간 안에 체결이 없으면 주문내역을 한 번 더 확인한다."""
        deadline = time.time() + self.live["fill_timeout_seconds"]
        while time.time() < deadline:
            self.read_stream(0.2)
            status, price, _ = self.fills.get(order_no, (None, 0, ""))
            if status == FILLED:
                return price
            if status == REJECTED:
                return None
        order = self.broker.get_order(order_no)
        return order["price"] if order and order["status"] == FILLED else None

    def write_status(self):
        now = datetime.now().strftime("%H%M%S")
        minute = self.bar_start(now)
        if minute == self.last_status or now[2:4] != minute[2:4] or now[4:] < "05":
            return
        self.last_status = minute
        fields = {
            "현재가": f"{self.last_price:.2f}" if self.last_price else "없음",
            "시세시각": self.last_tick_time,
            "포지션": f"{SIDE_NAME[self.position['side']]} 1계약" if self.position else "없음",
        }
        if self.position:
            ticks = round(self.position["side"] * (self.last_price - self.position["entry"]) / self.tick)
            fields["손익"] = f"{ticks:+d}틱"
            fields["손절"] = f"{self.position['stop']:.2f}"
        self.log.write("INFO", "상태", **fields, 호출=f"{self.client.request_count}회")
        if self.position and not self.broker.get_position():  # 수신이 끊긴 사이 손절된 경우
            self.record_stop_filled()

    def shutdown(self):
        # Ctrl+C로 멈출 때 남은 포지션을 정리하고 끝낸다.
        deadline = time.time() + 15
        while self.position and time.time() < deadline:
            self.exit_position("사용자 중단")
            time.sleep(0.2)
        self.finish(사유="사용자 중단")

    def finish(self, **extra):
        deadline = time.time() + 15
        while self.position and time.time() < deadline:
            self.exit_position("최종 청산")
            time.sleep(0.2)
        try:
            position = self.broker.get_position()
            open_orders = self.broker.count_open_orders()
            deposit = self.broker.get_deposit()
            if position or open_orders:
                self.log.write("ERROR", "오류", 단계="최종 확인", 잔여포지션=position or "없음", 미체결=f"{open_orders}건", 조치="수동 확인 필요")
            else:
                self.log.write("INFO", "최종 확인", 예탁자산=f"${deposit:,.2f}", 포지션="없음", 미체결="0건")
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="최종 확인", 내용=str(error)[:300], 조치="수동 확인 필요")
        self.stream.close()
        self.log.write(
            "INFO", "종료 요약",
            **extra,
            진입=f"{self.entries}회",
            손익=f"{self.day_ticks:+d}틱(${self.day_ticks * self.config['tick_value']:+,.2f})",
            호출=f"{self.client.request_count}회",
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    return parser.parse_args()


def main():
    # 이 프로그램은 실제 주문을 내므로 모의 환경에서만 실행한다.
    if load_env()["NH_ENV"].upper() != "DEMO":
        raise SystemExit("NH_ENV가 demo가 아니라서 실행하지 않습니다.")
    config = load_json(get_args().config)
    client = create_client("overseas")
    broker = OverseasBroker(client, config["symbol"])
    stream = OverseasStream(client, config["symbol"])
    log = EventLog(datetime.now().strftime("%Y%m%d_%H%M%S"))

    trader = LiveTrader(config, client, broker, stream, log)
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.shutdown()
    finally:
        log.close()


if __name__ == "__main__":
    main()
