"""MNQ-MGC 스프레드 되돌림 전략을 모의계좌에서 실행한다."""

import argparse
import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

from core.bars import add_minutes
from core.client import create_client
from core.config import load_env, load_json
from core.market_data import fetch_chart
from week03.broker import FILLED, REJECTED
from week04.broker import LegBroker
from week04.stream import PairStream
from week04.strategy import check_entry, contract_ratio, entry_side, exit_reason, pair_pnl, round_trip_cost, spread, zscore

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).with_name("results")
SIDE_NAME = {1: "매수", -1: "매도"}
SPREAD_NAME = {1: "스프레드 매수", -1: "스프레드 매도"}


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


class PairTrader:
    def __init__(self, config, client, brokers, stream, log, session):
        self.config = config
        self.live = config["live"]
        self.params = {**config["strategy"], **config["sessions"][session], **config["versions"][config["selected_version"]]}
        self.session = session
        self.symbols = [leg["symbol"] for leg in config["legs"]]
        self.point_values = [leg["point_value"] for leg in config["legs"]]
        self.client = client
        self.brokers = brokers
        self.stream = stream
        self.log = log

        self.today = datetime.now().strftime("%Y%m%d")
        self.spreads = []
        self.minute = None  # 현재 처리 중인 분의 시작 시각
        self.needs_reload = False
        self.last = {symbol: None for symbol in self.symbols}
        self.pending = {}  # 직전 분을 확정하는 동안 들어온 체결가
        self.rolling = False
        self.last_tick_at = time.time()
        self.tick_at = {symbol: None for symbol in self.symbols}  # 실시간 체결을 한 번도 못 받았으면 None
        self.blank_symbol_logged = False
        self.fills = {}  # 주문번호: (상태, 체결평균가, 오류코드)
        self.last_orders = [None, None]  # 마지막 order_legs가 낸 주문번호 (레그 순서)
        self.quantities = None
        self.notional = 0.0
        self.cost = 0.0
        self.position = None
        self.entries = 0
        self.day_net = 0.0
        self.cooldown_until = None
        self.rejects = 0
        self.stopped = False
        self.needs_flatten = False
        self.last_status = None
        self.last_exit_try = 0
        self.state_path = RESULTS_DIR / f"state_{self.today}_{session}.json"
        self.start_deposit = None

    def run(self):
        self.prepare()
        while datetime.now().strftime("%H%M%S") < self.params["final_check_time"]:
            try:
                self.read_stream()
                if self.needs_flatten and self.flatten("정리 재시도"):
                    self.needs_flatten = False
                self.roll_minute()
                if self.position and self.position["exit_reason"]:
                    self.exit_pair(self.position["exit_reason"])
                self.write_status()
            except Exception as error:
                self.log.write("ERROR", "오류", 단계="실행", 원인=type(error).__name__, 내용=str(error)[:300], 조치="계속")
                time.sleep(1)
        self.finish()

    def prepare(self):
        self.log.write(
            "INFO", "시작",
            종목=" / ".join(f"{leg['name']}({leg['symbol']})" for leg in self.config["legs"]),
            브로커="mock",
            전략=self.config["selected_version"],
            시간=f"{self.session} {self.params['entry_start']}~{self.params['entry_end']} 진입, {self.params['close_time']} 청산",
        )
        positions = [broker.get_position() for broker in self.brokers]
        open_orders = self.brokers[0].count_open_orders()
        if open_orders:
            self.log.write("ERROR", "오류", 단계="계좌 동기화", 미체결=f"{open_orders}건", 조치="전략 시작 안 함")
            raise SystemExit(1)
        self.stream.connect()  # 남은 포지션을 정리할 때 체결 알림을 받아야 해서 먼저 연결한다
        if any(positions):
            # 이전 실행이 비정상 종료돼 남은 포지션이다. 실제 잔고 기준으로 모두 정리하고 시작한다.
            self.log.write("WARNING", "남은 포지션", **self.position_fields(positions), 조치="정리 후 시작")
            if not self.flatten("시작 전 정리"):
                raise SystemExit(1)
        deposit = self.brokers[0].get_deposit()
        self.log.write("INFO", "계좌 동기화", 예탁자산=f"${deposit:,.2f}", 포지션="없음", 미체결="0건")
        self.load_state(deposit)
        self.reload_history()
        self.log.write("INFO", "준비", 스프레드=f"{len(self.spreads)}분", 수신="WebSocket")

    def load_state(self, deposit):
        """같은 날 다시 시작하면 하루 손익과 진입 횟수, 계약 수를 이어받는다. 하루 손익은 시작 예탁자산과의 차이로 본다."""
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.start_deposit = state["start_deposit"]
            self.entries = state["entries"]
            self.stopped = state.get("stopped", False)
            self.cooldown_until = state.get("cooldown_until")
            self.day_net = deposit - self.start_deposit
            if state["quantities"]:
                self.quantities = tuple(state["quantities"])
                self.notional, self.cost = state["notional"], state["cost"]
            self.log.write("WARNING", "재시작", 진입=f"{self.entries}회", 하루손익=f"${self.day_net:+,.2f}", 계약=self.quantities or "미정")
        else:
            self.start_deposit = deposit
            self.save_state()

    def save_state(self):
        state = {
            "start_deposit": self.start_deposit, "entries": self.entries, "stopped": self.stopped,
            "cooldown_until": self.cooldown_until,
            "quantities": self.quantities, "notional": self.notional, "cost": self.cost,
        }
        try:
            # 기존 파일이 깨지지 않도록 임시 파일에 쓴 뒤 교체한다
            temp = self.state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state), encoding="utf-8")
            temp.replace(self.state_path)
        except Exception as error:
            self.stopped = True  # 재시작 때 한도를 잘못 이어받지 않게 신규 진입을 멈춘다
            self.log.write("ERROR", "상태 저장 실패", 내용=str(error)[:300], 조치="신규 진입 중단")

    def reload_history(self):
        """오늘 마감된 1분봉으로 스프레드를 다시 만든다. 한쪽 봉이 없으면 직전 종가를 사용한다."""
        now_minute = datetime.now().strftime("%H%M") + "00"
        closes = []
        for symbol in self.symbols:
            payload = {"exch_cd": self.config["exchange"], "sym": symbol, "dcnt": "1"}
            bars = fetch_chart(self.client, "/trade/v1/overseas/chart-minute", payload, self.live["warmup_bars"])
            closes.append({bar.time: bar.close for bar in bars if bar.date == self.today and bar.time < now_minute})
        times = sorted(set(closes[0]) | set(closes[1]))
        last = [None, None]
        self.spreads = []
        for minute in times:
            last = [closes[i].get(minute, last[i]) for i in range(2)]
            if None not in last:
                self.spreads.append(spread(*last))
        for symbol, price in zip(self.symbols, last):
            self.last[symbol] = price or self.last[symbol]
        self.minute = now_minute
        self.needs_reload = False

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
        # 끊긴 동안의 분봉은 다음 분이 바뀔 때 REST로 다시 받는다.
        self.log.write("WARNING", "재접속", 사유=reason)
        self.stream.close()
        self.needs_reload = True
        self.last_tick_at = time.time()
        self.tick_at = {symbol: None for symbol in self.symbols}  # 두 종목 구독이 모두 살아났는지 다시 확인한다
        try:
            self.stream.connect()
        except Exception as error:
            self.log.write("ERROR", "재접속 실패", 원인=type(error).__name__)
            time.sleep(3)

    def on_tick(self, body):
        symbol = body.get("sym", "").strip()
        if symbol not in self.last:
            if not self.blank_symbol_logged:
                self.blank_symbol_logged = True
                self.log.write("WARNING", "알 수 없는 종목", 종목코드=symbol or "빈 값")
            return
        price = float(body["last_pric"])
        tick_minute = body.get("ko_exec_tm", "")[:4] + "00"
        if self.minute and tick_minute < self.minute:  # 늦게 도착한 이전 분 체결
            return
        self.last_tick_at = self.tick_at[symbol] = time.time()
        # 분 확정 중에 들어온 체결은 판단이 끝난 뒤 반영한다
        if self.rolling or (self.minute and tick_minute > self.minute):
            self.pending[symbol] = price
        else:
            self.last[symbol] = price

    def on_order(self, body):
        order_no = body["ord_no"].strip()
        status = body["ord_sts"].strip()
        price = float(body.get("exec_avg_pric") or body.get("exec_pric") or 0)
        self.fills[order_no] = (status, price, body["err_cd"].strip())
        if status == REJECTED:
            self.log.write("ERROR", "주문 응답", 주문번호=order_no, 상태="거부", 오류코드=body["err_cd"].strip())

    def roll_minute(self):
        """분이 바뀌면 직전 분의 스프레드를 확정하고 판단한다. 백테스트의 봉 종가 판단과 같다."""
        now_minute = datetime.now().strftime("%H%M") + "00"
        if now_minute == self.minute:
            return
        closed = self.minute
        self.rolling = True
        try:
            reloaded = False
            if self.needs_reload:
                try:
                    self.reload_history()
                    reloaded = True
                except Exception as error:
                    # 분봉 복구에 실패해도 보유 포지션의 청산 판단은 계속한다
                    self.log.write("ERROR", "오류", 단계="분봉 복구", 내용=str(error)[:300], 조치="다음 분에 재시도")
            if reloaded:
                try:
                    self.reconcile()
                except Exception as error:
                    self.needs_reload = True  # 잔고를 맞추기 전에는 신규 진입하지 않는다
                    self.log.write("ERROR", "오류", 단계="잔고 동기화", 내용=str(error)[:300], 조치="다음 분에 재시도")
            else:
                self.minute = now_minute
                if None not in self.last.values():
                    self.spreads.append(spread(*self.prices()))
            if None not in self.last.values():
                self.decide(closed)
        finally:
            self.last.update(self.pending)
            self.pending.clear()
            self.rolling = False

    def decide(self, minute):
        params = self.params
        result = zscore(self.spreads, params["window"])
        if self.position:
            position = self.position
            gross = pair_pnl(position["side"], self.quantities, position["prices"], self.prices(), self.point_values)
            position["peak"] = max(position["peak"], gross)
            held = minutes_between(position["time"], minute)
            reason = "마감" if minute >= params["close_time"] else None
            if not reason and result:
                reason = exit_reason(position["side"], result[0], gross, position["peak"], self.cost, held, params)
            if reason:
                self.exit_pair(reason)
            return

        if not result or self.stopped or self.needs_reload or not params["entry_start"] <= minute < params["entry_end"]:
            return
        if self.entries >= params["max_entries_per_day"] or self.day_net <= -params["daily_loss"]:
            return
        if self.cooldown_until and minute < self.cooldown_until:
            return
        # 오래된 가격으로 계약 수를 정하지 않도록 시세 상태를 먼저 확인한다
        silence = self.live["max_leg_silence_seconds"]
        fresh = all(tick is not None and time.time() - tick <= silence for tick in self.tick_at.values())
        if fresh and not self.quantities:
            self.set_quantities()
        z, std = result
        side = entry_side(z, params["entry_z"])
        if not side:
            return
        block = check_entry(z, std, self.notional, self.cost, params) if fresh else "실시간 시세 없음"
        self.log.write("INFO", "신호", 분=minute[:4], z=f"{z:+.2f}", 방향=SPREAD_NAME[side], 진입=f"안 함 ({block})" if block else "함")
        if not block:
            self.enter_pair(side, minute)

    def set_quantities(self):
        """진입 시간대 첫 판단 때의 가격으로 계약 수를 정하고 당일 고정한다. 백테스트도 진입 시작 시각 가격을 쓴다."""
        prices = self.prices()
        params = self.params
        self.quantities = contract_ratio(prices, self.point_values, params["size_tolerance"])
        self.notional = sum(q * p * v for q, p, v in zip(self.quantities, prices, self.point_values)) / 2
        tick_values = [leg["tick_size"] * leg["point_value"] for leg in self.config["legs"]]
        self.cost = round_trip_cost(self.quantities, self.config["fee_per_contract"], self.config["backtest"]["slippage_ticks"], tick_values)
        self.save_state()
        self.log.write(
            "INFO", "계약 수",
            **{symbol: f"{qty}계약 (${qty * price * point:,.0f})" for symbol, qty, price, point in zip(self.symbols, self.quantities, prices, self.point_values)},
            예상비용=f"${self.cost:,.2f}",
        )

    def enter_pair(self, side, minute):
        sides = (side, -side)
        fills = self.order_legs([(i, sides[i], self.quantities[i]) for i in range(2)], "진입")
        missing = [i for i in range(2) if fills[i] is None]
        if len(missing) == 1:
            i = missing[0]
            try:
                order_no = self.last_orders[i]
                held = self.brokers[i].get_position()
                if not held and order_no:  # 체결이 늦게 확인될 수 있어 2초 동안 체결 알림을 더 기다린다
                    deadline = time.time() + 2
                    while time.time() < deadline and self.fills.get(order_no, (None,))[0] != FILLED:
                        self.read_stream(0.2)
                    held = self.brokers[i].get_position()
                if self.fills.get(order_no, (None,))[0] == FILLED:
                    fills[i] = self.fills[order_no][1]
                elif held and held[:2] == (sides[i], self.quantities[i]):  # 체결 확인만 늦은 경우라 다시 주문하지 않는다
                    fills[i] = held[2]
                elif not held:
                    fills[i] = self.order_legs([(i, sides[i], self.quantities[i])], "진입 재주문")[0]
            except Exception as error:
                self.log.write("ERROR", "오류", 단계="진입 잔고 확인", 내용=str(error)[:300], 조치="정리")
        if None in fills:
            self.log.write("ERROR", "진입 실패", 체결=self.fill_text(fills), 조치="체결된 레그 정리")
            if not self.flatten("진입 실패 정리"):
                self.stopped = self.needs_flatten = True
                self.save_state()
            return
        self.position = {"side": side, "prices": fills, "time": minute, "peak": 0.0, "exit_reason": None, "exit_prices": [None, None]}
        self.entries += 1
        self.save_state()
        self.log.write("INFO", "진입 체결", 방향=SPREAD_NAME[side], 체결=self.fill_text(fills), 진입=f"{self.entries}회")

    def exit_pair(self, reason):
        # 실패하면 다음 반복에서 3초 간격으로 다시 시도한다.
        self.position["exit_reason"] = reason
        if time.time() - self.last_exit_try < 3:
            return
        self.last_exit_try = time.time()
        prices = self.close_legs(reason)
        if prices is None:
            return
        position, self.position = self.position, None
        gross = pair_pnl(position["side"], self.quantities, position["prices"], prices, self.point_values)
        net = gross - 2 * self.config["fee_per_contract"] * sum(self.quantities)
        self.day_net += net
        if reason in ("z 손절", "금액 손절"):
            self.cooldown_until = add_minutes(datetime.now().strftime("%H%M") + "00", self.params["cooldown_minutes"])
        self.save_state()
        self.log.write(
            "INFO", "청산 체결",
            사유=reason,
            청산=self.fill_text(prices),
            손익=f"${net:+,.2f} (수수료 전 ${gross:+,.2f})",
            하루손익=f"${self.day_net:+,.2f}",
        )

    def close_legs(self, reason):
        """실제 잔고를 보고 남은 레그를 시장가로 정리한다. 모두 정리되면 레그별 청산가를 반환한다."""
        # 재시도할 때도 먼저 정리된 레그의 체결가를 유지한다.
        done = self.position["exit_prices"] if self.position else [None, None]
        prices = [done[i] or self.prices()[i] for i in range(2)]
        held = [broker.get_position() for broker in self.brokers]
        legs = [(i, -position[0], position[1]) for i, position in enumerate(held) if position]
        for (i, _, _), price in zip(legs, self.order_legs(legs, reason)):
            if price is None:
                return None
            prices[i] = done[i] = price
        if any(broker.get_position() for broker in self.brokers):
            return None
        return prices

    def flatten(self, reason):
        """기록과 관계없이 실제 잔고를 모두 정리한다. 15초 안에 못 끝내면 False를 반환한다."""
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                if self.close_legs(reason) is not None:
                    self.position = None
                    return True
            except Exception as error:
                self.log.write("WARNING", "정리 실패", 단계=reason, 내용=str(error)[:300], 조치="3초 뒤 재시도")
            time.sleep(3)
        self.log.write("ERROR", "오류", 단계=reason, 조치="수동 확인 필요")
        return False

    def order_legs(self, legs, purpose):
        """(레그 번호, 방향, 수량) 목록을 시장가로 모두 주문한 뒤 체결을 기다린다. 레그별 체결가를 반환하고 실패는 None이다."""
        orders = []
        for index, side, quantity in legs:
            symbol = self.symbols[index]
            try:
                order_no = self.brokers[index].place_market(side, quantity)
                self.log.write("INFO", "주문 응답", 용도=purpose, 종목=symbol, 방향=SIDE_NAME[side], 수량=quantity, 상태="접수", 주문번호=order_no)
            except Exception as error:
                order_no = None
                self.log.write("ERROR", "주문 응답", 용도=purpose, 종목=symbol, 상태="실패", 원인=type(error).__name__, 내용=str(error)[:300])
            orders.append(order_no)
        self.last_orders = orders
        prices = [self.wait_fill(index, order_no) if order_no else None for (index, _, _), order_no in zip(legs, orders)]
        if None in prices:
            self.on_reject()
        else:
            self.rejects = 0
        return prices

    def on_reject(self):
        self.rejects += 1
        if self.rejects >= self.live["max_rejects"] and not self.stopped:
            self.stopped = True
            self.save_state()
            self.log.write("ERROR", "거래 중단", 사유=f"주문 실패 연속 {self.rejects}회", 조치="그날 신규 진입 안 함")

    def wait_fill(self, index, order_no):
        """체결가를 반환한다. 시간 안에 체결이 없으면 주문내역을 한 번 더 확인한다."""
        deadline = time.time() + self.live["fill_timeout_seconds"]
        while time.time() < deadline:
            self.read_stream(0.2)
            status, price, _ = self.fills.get(order_no, (None, 0, ""))
            if status == FILLED:
                return price
            if status == REJECTED:
                return None
        order = self.brokers[index].get_order(order_no)
        return order["price"] if order and order["status"] == FILLED else None

    def reconcile(self):
        """기록과 실제 잔고가 다르면 실제 잔고 기준으로 모두 정리한다."""
        positions = [broker.get_position() for broker in self.brokers]
        expected = None
        if self.position:
            side = self.position["side"]
            expected = [(side, self.quantities[0]), (-side, self.quantities[1])]
        actual = [(held[0], held[1]) if held else None for held in positions]
        if actual == (expected or [None, None]):
            return
        self.log.write("WARNING", "잔고 불일치", 기록=SPREAD_NAME[self.position["side"]] if self.position else "없음", **self.position_fields(positions), 조치="정리")
        if self.position:
            self.position["exit_reason"] = "잔고 불일치"
        elif not self.flatten("잔고 불일치 정리"):
            self.stopped = self.needs_flatten = True
            self.save_state()

    def write_status(self):
        now = datetime.now().strftime("%H%M%S")
        if int(now[2:4]) % 5 or now[4:] < "05" or now[:4] == self.last_status:
            return
        self.last_status = now[:4]
        result = zscore(self.spreads, self.params["window"])
        fields = {symbol: f"{price:.2f}" if price else "없음" for symbol, price in self.last.items()}
        fields["z"] = f"{result[0]:+.2f}" if result else "계산 전"
        fields["포지션"] = SPREAD_NAME[self.position["side"]] if self.position else "없음"
        if self.position:
            gross = pair_pnl(self.position["side"], self.quantities, self.position["prices"], self.prices(), self.point_values)
            fields["평가손익"] = f"${gross:+,.2f}"
        self.log.write(
            "INFO", "상태", **fields, 하루손익=f"${self.day_net:+,.2f}",
            호출=f"{self.client.request_count}회", 토큰재발급=f"{self.client.token_refreshes}회",
        )
        if self.position or now[2:4] == "00":  # 포지션이 있으면 5분마다, 없으면 1시간마다 잔고를 확인한다
            self.reconcile()

    def prices(self):
        return tuple(self.last[symbol] for symbol in self.symbols)

    def fill_text(self, prices):
        return " / ".join(f"{symbol} {price:.2f}" if price else f"{symbol} 실패" for symbol, price in zip(self.symbols, prices))

    def position_fields(self, positions):
        return {
            symbol: f"{SIDE_NAME[held[0]]} {held[1]}계약" if held else "없음"
            for symbol, held in zip(self.symbols, positions)
        }

    def shutdown(self):
        # Ctrl+C로 멈출 때 남은 포지션을 정리하고 끝낸다.
        if self.position:
            self.exit_pair("사용자 중단")
        self.finish(사유="사용자 중단")

    def finish(self, **extra):
        if self.position:
            self.last_exit_try = 0
            self.exit_pair("최종 청산")
        try:
            if any(broker.get_position() for broker in self.brokers):
                self.flatten("최종 정리")
            positions = [broker.get_position() for broker in self.brokers]
            open_orders = self.brokers[0].count_open_orders()
            deposit = self.brokers[0].get_deposit()
            if any(positions) or open_orders:
                self.log.write("ERROR", "오류", 단계="최종 확인", **self.position_fields(positions), 미체결=f"{open_orders}건", 조치="수동 확인 필요")
            else:
                self.log.write("INFO", "최종 확인", 예탁자산=f"${deposit:,.2f}", 포지션="없음", 미체결="0건")
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="최종 확인", 내용=str(error)[:300], 조치="수동 확인 필요")
        self.stream.close()
        self.log.write("INFO", "종료 요약", **extra, 진입=f"{self.entries}회", 손익=f"${self.day_net:+,.2f}", 호출=f"{self.client.request_count}회")


def minutes_between(start, end):
    return (int(end[:2]) * 60 + int(end[2:4])) - (int(start[:2]) * 60 + int(start[2:4]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--session", help="config의 selected_session 대신 사용할 시간 설정 (contest, long)")
    return parser.parse_args()


def main():
    # 이 프로그램은 실제 주문을 내므로 모의 환경에서만 실행한다.
    if load_env()["NH_ENV"].upper() != "DEMO":
        raise SystemExit("NH_ENV가 demo가 아니라서 실행하지 않습니다.")
    args = get_args()
    config = load_json(args.config)
    session = args.session or config["selected_session"]
    client = create_client("overseas")
    symbols = [leg["symbol"] for leg in config["legs"]]
    brokers = [LegBroker(client, symbol) for symbol in symbols]
    stream = PairStream(client, symbols)
    log = EventLog(datetime.now().strftime("%Y%m%d_%H%M%S"))

    trader = PairTrader(config, client, brokers, stream, log, session)
    try:
        trader.run()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # 정리 중에 Ctrl+C를 또 누르면 청산이 끊긴다
        trader.shutdown()
    finally:
        log.close()


if __name__ == "__main__":
    main()
