"""3년 국채선물 전략을 장중에 실행한다."""

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from core.bars import add_minutes
from core.broker import NHBroker, OrderRejected, PaperBroker
from core.client import create_client
from core.config import load_json
from core.indicators import atr, sma
from core.market_data import fetch_best_quote, fetch_daily_bars, fetch_minute_bars
from week02.strategy import BUY, calculate_stop_ticks, can_enter, crossover_signals, daily_direction

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
    def __init__(self, config, broker_name, client, broker, log):
        self.config = config
        self.live = config["live"]
        self.params = {**config["strategy"], **config["versions"][config["selected_version"]]}
        self.symbol = config["symbol"]
        self.tick = config["tick_size"]
        self.broker_name = broker_name
        self.client = client
        self.broker = broker
        self.log = log

        self.today = datetime.now().strftime("%Y%m%d")
        self.daily_bars = []
        self.position = None  # 방향, 진입가, 손절가
        self.exit_order = None  # 주문번호, 청산 사유
        self.open_entry = None  # 취소를 요청했지만 아직 끝나지 않은 진입 주문
        self.entries = 0
        self.day_ticks = 0
        self.exit_counts = Counter()
        self.last_bar_end = None
        self.query_failures = Counter()  # 대상(호가, 분봉, 주문내역)별 연속 실패 횟수
        self.entry_stopped = False
        self.close_logged = False

    def run(self):
        self.prepare()
        while True:
            now = datetime.now().strftime("%H%M%S")
            if now >= self.live["final_check_time"]:
                self.finish()
                return

            quote = self.fetch_quote()
            if self.open_entry:
                self.resolve_open_entry()
            if now >= self.live["stop_entry_time"] and not self.entry_stopped:
                self.entry_stopped = True
                self.log.write("INFO", "진입 중단", 사유="장 마감", 대기주문=f"{1 if self.open_entry else 0}건")

            if self.position and now >= self.live["close_order_time"]:
                if not self.close_logged:
                    self.close_logged = True
                    self.log.write("INFO", "마감 청산", 포지션=f"{SIDE_NAME[self.position['side']]} 1계약", 유형="시장가")
                self.exit("마감", quote)
            elif self.position and quote:
                self.check_daily_loss(quote)
                if self.position:
                    self.check_stop(quote)

            if now < self.live["close_order_time"]:
                self.check_new_bar(quote)
            time.sleep(self.live["poll_seconds"])

    def prepare(self):
        self.log.write(
            "INFO", "시작",
            종목=f"{self.config['name']}({self.symbol})",
            브로커=self.broker_name,
            전략=self.config["selected_version"],
            조회주기=f"{self.live['poll_seconds']}초",
        )
        try:
            deposit = self.broker.get_deposit()
            position = self.broker.get_position()
            open_orders = self.broker.count_open_orders()
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="계좌 동기화", 내용=str(error)[:120], 조치="전략 시작 안 함")
            raise SystemExit(1)
        if position or open_orders:
            self.log.write("ERROR", "오류", 단계="계좌 동기화", 포지션=position or "없음", 미체결=f"{open_orders}건", 조치="전략 시작 안 함")
            raise SystemExit(1)
        self.log.write(
            "INFO", "계좌 동기화",
            예탁자산=f"{deposit:,.0f}원" if deposit is not None else "paper",
            포지션="없음",
            미체결="0건",
        )

        for attempt in range(1, 4):
            try:
                self.daily_bars = fetch_daily_bars(self.client, self.symbol, 30)
                bars = self.completed_bars()
                break
            except Exception as error:
                self.record_failure("준비 봉", error)
                if attempt == 3:
                    self.log.write("ERROR", "오류", 단계="준비", 내용="준비 봉 조회 3회 실패", 조치="전략 시작 안 함")
                    raise SystemExit(1)
                time.sleep(2)
        if len(bars) <= self.params["long_window"]:
            self.log.write("ERROR", "오류", 단계="준비", 분봉=f"{len(bars)}개", 조치="전략 시작 안 함")
            raise SystemExit(1)
        closes = [bar.close for bar in bars]
        self.log.write(
            "INFO", "준비 완료",
            분봉=f"{len(bars)}개",
            SMA=f"{sma(closes, self.params['short_window'])[-1]:.3f}/{sma(closes, self.params['long_window'])[-1]:.3f}",
            ATR14=f"{atr(bars, self.params['atr_window'])[-1] / self.tick:.1f}틱",
            필터="ON" if self.params["daily_filter"] else "OFF",
        )

    def completed_bars(self):
        """아직 끝나지 않은 봉은 빼고 반환한다."""
        interval = self.config["interval_minutes"]
        bars = fetch_minute_bars(self.client, self.symbol, self.live["warmup_bars"], interval)
        now = datetime.now()
        return [
            bar for bar in bars
            if datetime.strptime(bar.date + bar.time, "%Y%m%d%H%M%S") + timedelta(minutes=interval) <= now
        ]

    def fetch_quote(self):
        # 호가는 1초마다 보므로 오래 기다리지 않는다. 실패하면 바로 한 번 더 본다.
        try:
            quote = fetch_best_quote(self.client, self.symbol, timeout=3)
        except Exception:
            try:
                quote = fetch_best_quote(self.client, self.symbol, timeout=3)
            except Exception as error:
                self.record_failure("호가", error)
                return None
        self.query_failures["호가"] = 0
        if quote and isinstance(self.broker, PaperBroker):
            self.broker.quote = quote
        return quote

    def record_failure(self, target, error=None):
        self.query_failures[target] += 1
        count = self.query_failures[target]
        if count >= self.live["max_query_failures"]:
            self.log.write("ERROR", "조회 실패", 대상=target, 원인=type(error).__name__, 연속실패=f"{count}회", 신규진입="중단")
        else:
            self.log.write("WARNING", "조회 실패", 대상=target, 원인=type(error).__name__, 연속실패=f"{count}회", 조치="재시도")

    def check_daily_loss(self, quote):
        # 실현 손익과 지금 청산했을 때의 미실현 손익을 더해 일일 손실 한도를 넘으면 청산하고 그날 진입을 멈춘다.
        ask, bid = quote
        side = self.position["side"]
        exit_price = bid if side == BUY else ask
        open_ticks = round(side * (exit_price - self.position["entry_price"]) / self.tick)
        limit = self.params["daily_loss_limit_ticks"]
        if self.day_ticks + open_ticks > -limit:
            return
        if not self.entry_stopped:
            self.entry_stopped = True
            self.log.write("WARNING", "진입 중단", 사유="일일 손실 한도", 손익=f"{self.day_ticks + open_ticks:+d}틱", 한도=f"-{limit}틱")
        self.exit("일일 한도", quote)

    def check_stop(self, quote):
        ask, bid = quote
        side = self.position["side"]
        # 실제 청산할 수 있는 호가를 기준으로 손절 여부를 확인한다.
        price = bid if side == BUY else ask
        stop_price = self.position["stop_price"]
        hit = price <= stop_price if side == BUY else price >= stop_price
        if hit:
            if not self.exit_order:
                self.log.write(
                    "INFO", "손절 감지",
                    기준="최우선 매수호가" if side == BUY else "최우선 매도호가",
                    가격=f"{price:.2f}",
                    손절선=f"{stop_price:.2f}",
                )
            self.exit("손절", quote)

    def check_new_bar(self, quote):
        interval = self.config["interval_minutes"]
        now = datetime.now()
        boundary = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % interval)
        # 막 끝난 봉이 서버에 반영될 때까지 2초 기다린다.
        if boundary == self.last_bar_end or now - boundary < timedelta(seconds=2):
            return

        try:
            bars = self.completed_bars()
        except Exception as error:
            self.record_failure("분봉", error)
            return
        self.query_failures["분봉"] = 0
        last = bars[-1] if bars else None
        bar_end = add_minutes(last.time, interval) if last else ""
        if not last or last.date != self.today or bar_end != boundary.strftime("%H%M%S"):
            # 서버 반영이 늦을 수 있어 30초까지는 다시 조회하고, 그래도 없으면 거래가 없던 봉으로 본다.
            if now - boundary >= timedelta(seconds=30):
                self.last_bar_end = boundary
            return

        closes = [bar.close for bar in bars]
        signal = crossover_signals(closes, self.params["short_window"], self.params["long_window"])[-1]
        if signal and not self.position and quote is None and now - boundary < timedelta(seconds=30):
            return  # 진입에 쓸 호가를 받을 때까지 이 봉의 신호를 소비하지 않는다.
        self.last_bar_end = boundary
        if not signal:
            return

        filter_side = daily_direction(self.daily_bars, self.today, self.params["daily_filter_window"])
        allowed_side = filter_side if self.params["daily_filter"] else 0
        atr_value = atr(bars, self.params["atr_window"])[-1]
        stop_ticks = calculate_stop_ticks(atr_value, self.params, self.tick)
        enterable = (
            not self.position
            and not self.open_entry
            and not self.entry_stopped
            and max(self.query_failures.values(), default=0) < self.live["max_query_failures"]
            and quote is not None
            and can_enter(self.params, signal, bar_end, self.entries, self.day_ticks, stop_ticks, allowed_side)
        )
        self.log.write(
            "INFO", "신호",
            봉=f"{last.time[:2]}:{last.time[2:4]}~{bar_end[:2]}:{bar_end[2:4]}",
            방향=SIDE_NAME[signal],
            교차="상향" if signal == BUY else "하향",
            필터="ON" if self.params["daily_filter"] else "OFF",
            필터판정="통과" if filter_side in (0, signal) else "차단",
            진입="가능" if enterable else "불가",
        )

        if self.position:
            # 반대 교차는 기존 포지션 청산에만 사용한다.
            if signal == -self.position["side"]:
                self.exit("반대 교차", quote)
        elif enterable:
            self.enter(signal, stop_ticks, atr_value, quote)

    def enter(self, side, stop_ticks, atr_value, quote):
        ask, bid = quote
        price = ask if side == BUY else bid
        self.log.write("INFO", "진입 주문", 방향=SIDE_NAME[side], 수량=1, 유형="지정가", 가격=f"{price:.2f}")
        order_no = self.submit(side, price, "진입")
        if not order_no:
            return

        fill = self.wait_fill(order_no, self.live["entry_timeout_seconds"])
        if fill:
            self.open_position(side, fill, stop_ticks, atr_value)
            return

        try:
            self.broker.cancel_order(order_no, side)
            self.log.write("WARNING", "주문 취소", 주문번호=order_no, 상태="요청", 사유=f"{self.live['entry_timeout_seconds']}초 미체결")
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="주문 취소", 주문번호=order_no, 내용=str(error)[:120])
        # 취소가 확인되거나 그사이 체결된 것을 확인할 때까지 신규 진입을 막는다.
        self.open_entry = (order_no, side, stop_ticks, atr_value)
        self.resolve_open_entry()

    def resolve_open_entry(self):
        order_no, side, stop_ticks, atr_value = self.open_entry
        try:
            order = self.broker.get_order(order_no)
        except Exception as error:
            self.record_failure("주문내역", error)
            return
        self.query_failures["주문내역"] = 0
        if order and order["filled"] >= 1 and order["price"] > 0:
            self.open_entry = None
            self.open_position(side, order, stop_ticks, atr_value)
        elif order and order["remaining"] == 0:
            self.open_entry = None
            self.log.write("INFO", "주문 취소", 주문번호=order_no, 상태="완료")

    def open_position(self, side, fill, stop_ticks, atr_value):
        stop_price = round(fill["price"] - side * stop_ticks * self.tick, 2)
        self.position = {"side": side, "entry_price": fill["price"], "stop_price": stop_price}
        self.entries += 1
        self.log.write(
            "INFO", "진입 체결",
            방향=SIDE_NAME[side],
            가격=f"{fill['price']:.2f}",
            ATR14=f"{atr_value / self.tick:.1f}틱",
            손절=f"{stop_price:.2f}({stop_ticks}틱)",
        )

    def exit(self, reason, quote):
        side = -self.position["side"]
        if self.exit_order:
            order_no, reason = self.exit_order
            fill = self.wait_fill(order_no, 1)
            if fill:
                self.close_position(fill, reason)
                return
            try:
                order = self.broker.get_order(order_no)
            except Exception as error:
                self.record_failure("주문내역", error)
                return
            self.query_failures["주문내역"] = 0
            if order and order["remaining"] > 0:
                return  # 청산 주문이 남아 있으면 계속 기다린다.
            self.exit_order = None  # 미체결 잔량이 없으면 청산 주문을 다시 낸다.

        if isinstance(self.broker, PaperBroker) and not quote:
            return
        self.log.write("INFO", "청산 주문", 사유=reason, 방향=SIDE_NAME[side], 수량=1, 유형="시장가")
        order_no = self.submit(side, None, "청산")
        if not order_no:
            return
        self.exit_order = (order_no, reason)
        fill = self.wait_fill(order_no, self.live["exit_timeout_seconds"])
        if fill:
            self.close_position(fill, reason)

    def close_position(self, fill, reason):
        ticks = round(self.position["side"] * (fill["price"] - self.position["entry_price"]) / self.tick)
        self.day_ticks += ticks
        self.exit_counts[reason] += 1
        self.position = None
        self.exit_order = None
        self.log.write(
            "INFO", "청산 체결",
            사유=reason,
            가격=f"{fill['price']:.2f}",
            손익=f"{ticks:+d}틱({ticks * self.config['tick_value']:+,}원)",
            누적=f"{self.day_ticks:+d}틱",
        )

    def submit(self, side, price, purpose):
        try:
            order_no = self.broker.place_order(side, price)
        except (OrderRejected, RuntimeError) as error:
            self.log.write("ERROR", "주문 응답", 용도=purpose, 상태="거부", 내용=str(error)[:120])
            return None
        self.log.write("INFO", "주문 응답", 용도=purpose, 상태="접수", 주문번호=order_no)
        return order_no

    def wait_fill(self, order_no, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                order = self.broker.get_order(order_no)
                self.query_failures["주문내역"] = 0
            except Exception as error:
                self.record_failure("주문내역", error)
                order = None
            if order and order["rejected"]:
                self.log.write("ERROR", "주문 응답", 주문번호=order_no, 상태="거부", 내용=order["reason"])
                return None
            if order and order["filled"] >= 1 and order["price"] > 0:
                return order
            time.sleep(1)
        return None

    def shutdown(self):
        # Ctrl+C로 멈출 때 취소 중인 진입 주문과 남은 포지션을 정리하고 끝낸다.
        if self.open_entry:
            order_no, side = self.open_entry[:2]
            try:
                self.broker.cancel_order(order_no, side)
            except Exception:
                pass
            self.resolve_open_entry()
        deadline = time.time() + 15
        while self.position and time.time() < deadline:
            self.exit("사용자 중단", self.fetch_quote())
            if self.position:
                time.sleep(1)
        self.finish(사유="사용자 중단")

    def finish(self, **extra):
        remaining, open_orders = None, None
        try:
            remaining = self.broker.get_position()
            open_orders = self.broker.count_open_orders()
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="최종 확인", 내용=str(error)[:120])
        if remaining or open_orders or self.position or self.open_entry or self.exit_order:
            self.log.write(
                "ERROR", "오류",
                단계="최종 확인",
                잔여포지션=remaining or self.position or "없음",
                미체결=f"{open_orders}건",
                조치="수동 확인 필요",
            )
        self.summary(**extra)

    def summary(self, **extra):
        self.log.write(
            "INFO", "종료 요약",
            **extra,
            진입=f"{self.entries}회",
            손절=f"{self.exit_counts['손절']}건",
            교차청산=f"{self.exit_counts['반대 교차']}건",
            마감청산=f"{self.exit_counts['마감']}건",
            한도청산=f"{self.exit_counts['일일 한도']}건",
            손익=f"{self.day_ticks:+d}틱({self.day_ticks * self.config['tick_value']:+,}원)",
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--broker", choices=["paper", "mock"], default="paper")
    return parser.parse_args()


def main():
    args = get_args()
    config = load_json(args.config)
    client = create_client("domestic")
    broker = NHBroker(client, config["symbol"]) if args.broker == "mock" else PaperBroker()
    log = EventLog(datetime.now().strftime("%Y%m%d_%H%M%S"))

    trader = LiveTrader(config, args.broker, client, broker, log)
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.shutdown()
    finally:
        log.close()


if __name__ == "__main__":
    main()
