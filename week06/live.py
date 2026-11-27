"""4주차 MNQ-MGC 스프레드 전략에 6주차 위험관리를 더해 모의계좌에서 실행한다.

진입 신호와 기존 청산 규칙은 4주차 코드를 그대로 쓰고, 아래만 더한다.
- 진입 전: 반감기 필터, 변동성 상한, 3거래일 누적 한도
- 보유 중: 체결가마다 합산 손익으로 실시간 손절과 하루 한도 확인, 레그별 비상 스톱(STOP-M)
- 이상 감지: 진입 실패, 손절 초과, 연속 손실, 비상 스톱 체결 시 신규 진입 중단과 알림
"""

import argparse
import json
import math
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

from core.bars import add_minutes
from core.client import create_client
from core.config import DATA_DIR, load_env, load_json
from core.market_data import fetch_chart
from week03.broker import FILLED, REJECTED
import week04.live as week04_live
from week04.live import SIDE_NAME, SPREAD_NAME, EventLog, PairTrader
from week04.strategy import check_entry, entry_side, pair_pnl, spread, zscore
from week04.stream import PairStream
from week06.broker import StopLegBroker
from week06.risk import expected_move, half_life, leg_volatility, psi

RESULTS_DIR = Path(__file__).with_name("results")
STOP_REASONS = ("z 손절", "금액 손절", "실시간 손절", "비상 스톱")
ALERT_EVENTS = ("거래 중단", "비상 스톱 체결", "손절 초과", "진입 실패", "잔고 불일치", "시장 변화")


class AlertLog(EventLog):
    """4주차 로그에 디스코드 알림을 더한다. 웹훅 주소가 .env에 없으면 알림 없이 기록만 한다."""

    def __init__(self, run_id, webhook):
        super().__init__(run_id)
        self.webhook = webhook

    def write(self, level, event, **fields):
        super().write(level, event, **fields)
        if self.webhook and (level == "ERROR" or event in ALERT_EVENTS):
            text = " | ".join(f"{key}={value}" for key, value in fields.items())
            try:
                requests.post(self.webhook, json={"content": f"[6주차 {level}] {event} {text}"[:1900]}, timeout=3)
            except Exception:
                pass  # 알림 실패로 거래 판단이 멈추면 안 된다


class RiskTrader(PairTrader):
    def __init__(self, config, client, brokers, stream, log):
        legacy = {**config, "sessions": {"contest": config["session"]}, "selected_session": "contest"}
        super().__init__(legacy, client, brokers, stream, log, "contest")
        self.rules = {**config["rules"], **config["versions"][config["selected_version"]]}
        self.risk = config["risk"]
        self.state_path = RESULTS_DIR / f"state_{self.today}.json"
        self.volatility = None
        self.losses = 0
        self.stop_orders = [None, None]
        self.daily_path = RESULTS_DIR / "daily_pnl.json"

    def prepare(self):
        super().prepare()
        self.load_history()
        if self.rules["multi_day"]:
            recent = self.recent_days()
            if len(recent) == self.risk["multi_day_count"] and sum(recent.values()) <= -self.risk["multi_day_loss"]:
                self.stop_trading(f"최근 {len(recent)}거래일 손익 ${sum(recent.values()):+,.0f}", "3일 누적 한도")
        self.log.write("INFO", "위험관리", **{key: "켬" if value else "끔" for key, value in self.rules.items()})

    def load_history(self):
        """직전 거래일들의 1분봉으로 변동성(σ1, σ2, ρ)과 전일 PSI를 계산한다."""
        start, end = self.risk["history_window"]
        closes = []
        for symbol in self.symbols:
            payload = {"exch_cd": self.config["exchange"], "sym": symbol, "dcnt": "1"}
            bars = fetch_chart(self.client, "/trade/v1/overseas/chart-minute", payload, self.live["history_bars"])
            by_date = defaultdict(dict)
            for bar in bars:
                if bar.date < self.today and start <= bar.time <= end:
                    by_date[bar.date][bar.time] = bar.close
            closes.append(by_date)
        days = sorted(d for d in set(closes[0]) & set(closes[1]) if min(len(closes[0][d]), len(closes[1][d])) >= 150)
        days = days[-self.risk["vol_days"]:]
        returns = ([], [])
        for date in days:
            times = sorted(set(closes[0][date]) & set(closes[1][date]))
            for before, after in zip(times, times[1:]):
                for i in (0, 1):
                    returns[i].append(math.log(closes[i][date][after] / closes[i][date][before]))
        self.volatility = leg_volatility(*returns)
        sigma1, sigma2, rho = self.volatility
        self.log.write("INFO", "변동성", 기간=f"{days[0]}~{days[-1]} ({len(days)}일)", σ=f"{sigma1 * 1e4:.2f}bp / {sigma2 * 1e4:.2f}bp", 상관=f"{rho:.2f}")

        previous = days[-1]
        times = sorted(set(closes[0][previous]) & set(closes[1][previous]))
        changes = [spread(closes[0][previous][b], closes[1][previous][b]) - spread(closes[0][previous][a], closes[1][previous][a]) for a, b in zip(times, times[1:])]
        try:
            value = psi(changes, self.reference_changes())
        except Exception as error:
            self.log.write("WARNING", "PSI 계산 실패", 내용=str(error)[:300], 조치="거래는 계속")
            return
        level = "WARNING" if value > self.risk["psi_alert"] else "INFO"
        self.log.write(level, "시장 변화" if level == "WARNING" else "PSI", 날짜=previous, PSI=f"{value:.2f}", 기준=f"{self.risk['psi_alert']} 초과면 백테스트 때와 다른 시장")

    def reference_changes(self):
        """PSI 기준 기간(9/30 ~ 10/27, 4주차 설정을 고른 기간)의 15~18시 1분 스프레드 변화. 저장소 밖 4주차 데이터 파일을 쓴다."""
        from week06.backtest import load_leg

        start, end = self.risk["psi_reference"]
        legs = [load_leg(symbol, {"start_date": start, "end_date": end}) for symbol in self.symbols]
        changes = []
        for date in sorted(set(legs[0]) & set(legs[1])):
            times = sorted(t for t in legs[0][date] if "150000" <= t < "180000" and t in legs[1][date])
            values = [spread(legs[0][date][t].close, legs[1][date][t].close) for t in times]
            changes += [b - a for a, b in zip(values, values[1:])]
        return changes

    def recent_days(self):
        if not self.daily_path.exists():
            return {}
        daily = json.loads(self.daily_path.read_text(encoding="utf-8"))
        past = sorted(d for d in daily if d < self.today)[-self.risk["multi_day_count"]:]
        return {d: daily[d] for d in past}

    @property
    def fees(self):
        return 2 * self.config["fee_per_contract"] * sum(self.quantities)

    def decide(self, minute):
        if self.position:
            if not self.position["exit_reason"]:  # 이미 청산 중이면 사유를 덮어쓰지 않는다
                super().decide(minute)
            return
        params = self.params
        result = zscore(self.spreads, params["window"])
        if not result or self.stopped or self.needs_reload or not params["entry_start"] <= minute < params["entry_end"]:
            return
        if self.entries >= params["max_entries_per_day"] or self.day_net <= -params["daily_loss"]:
            return
        if self.cooldown_until and minute < self.cooldown_until:
            return
        silence = self.live["max_leg_silence_seconds"]
        fresh = all(tick is not None and time.time() - tick <= silence for tick in self.tick_at.values())
        if fresh and not self.quantities:
            self.set_quantities()
        z, std = result
        side = entry_side(z, params["entry_z"])
        if not side:
            return
        block = check_entry(z, std, self.notional, self.cost, params) if fresh else "실시간 시세 없음"
        life = half_life(self.spreads[-self.risk["half_life_window"]:]) if len(self.spreads) >= self.risk["half_life_window"] else math.inf
        move = expected_move(self.prices(), self.quantities, self.point_values, self.volatility, self.risk["vol_hold_minutes"]) if self.quantities else math.nan
        if not block and self.rules["half_life"] and life > self.risk["max_half_life"]:
            block = "반감기"
        if not block and self.rules["vol_cap"] and move > self.risk["max_expected_move"]:
            block = "변동성 상한"
        self.log.write(
            "INFO", "신호", 분=minute[:4], z=f"{z:+.2f}", 방향=SPREAD_NAME[side],
            반감기="계산 전" if math.isinf(life) and len(self.spreads) < self.risk["half_life_window"] else f"{life:.1f}분",
            예상변동액=f"${move:,.0f}", 진입=f"안 함 ({block})" if block else "함",
        )
        if not block:
            self.enter_pair(side, minute)

    def enter_pair(self, side, minute):
        entries = self.entries
        super().enter_pair(side, minute)
        if self.entries == entries:
            # 한쪽만 체결돼 정리한 경우라, 같은 일이 반복되지 않게 그날은 신규 진입을 멈춘다
            self.stop_trading("진입 실패 후 정리", "진입 실패")
            return
        self.position["worst"] = 0.0
        self.place_stops()

    def place_stops(self):
        """레그마다 그 레그 혼자 비상 손실 한도($600)를 잃는 가격에 STOP-M을 건다. 프로그램이 멈췄을 때를 대비한다."""
        sides = (self.position["side"], -self.position["side"])
        for i, broker in enumerate(self.brokers):
            leg = self.config["legs"][i]
            distance = self.risk["emergency_leg_loss"] / (leg["point_value"] * self.quantities[i])
            ticks = math.ceil(distance / leg["tick_size"])
            price = self.position["prices"][i] - sides[i] * ticks * leg["tick_size"]
            try:
                self.stop_orders[i] = broker.place_stop(-sides[i], self.quantities[i], price)
                self.log.write("INFO", "비상 스톱", 종목=self.symbols[i], 방향=SIDE_NAME[-sides[i]], 수량=self.quantities[i], 가격=f"{price:.2f}", 주문번호=self.stop_orders[i])
            except Exception as error:
                self.stop_orders[i] = None
                self.log.write("WARNING", "비상 스톱 실패", 종목=self.symbols[i], 내용=str(error)[:300], 조치="프로그램 감시만")

    def on_tick(self, body):
        super().on_tick(body)
        self.check_live_risk()

    def check_live_risk(self):
        """체결가가 들어올 때마다 두 레그 합산 손익으로 실시간 손절(−$280)과 하루 한도(−$600)를 확인한다."""
        position = self.position
        if not position or position["exit_reason"] or None in self.last.values():
            return
        latest = {**self.last, **self.pending}
        prices = tuple(latest[symbol] for symbol in self.symbols)
        net = pair_pnl(position["side"], self.quantities, position["prices"], prices, self.point_values) - self.fees
        position["worst"] = min(position.get("worst", 0.0), net)
        if self.rules["live_stop"] and net <= -self.risk["live_stop"]:
            position["exit_reason"] = "실시간 손절"
        elif self.rules["daily_flatten"] and self.day_net + net <= -self.params["daily_loss"]:
            position["exit_reason"] = "하루 한도"
        if position["exit_reason"]:
            self.log.write("INFO", "손절 판단", 사유=position["exit_reason"], 가격=self.fill_text(prices), 평가손익=f"${net:+,.2f}", 하루손익=f"${self.day_net:+,.2f}")

    def on_order(self, body):
        super().on_order(body)
        order_no = body["ord_no"].strip()
        if order_no not in self.stop_orders:
            return
        i = self.stop_orders.index(order_no)
        status, price, _ = self.fills[order_no]
        if status == FILLED:
            self.stop_orders[i] = None
            if self.position:
                self.position["exit_prices"][i] = price
                self.position["exit_reason"] = self.position["exit_reason"] or "비상 스톱"
            self.log.write("ERROR", "비상 스톱 체결", 종목=self.symbols[i], 가격=f"{price:.2f}", 조치="남은 레그 정리")
        elif status == REJECTED:
            self.stop_orders[i] = None

    def close_legs(self, reason):
        # 비상 스톱이 살아 있을 때 시장가로 청산하면 나중에 스톱이 체결돼 반대 포지션이 생길 수 있다
        if not self.cancel_stops():
            return None
        return super().close_legs(reason)

    def cancel_stops(self):
        """비상 스톱을 취소하고 미체결이 없어질 때까지 확인한다. 끝내 남아 있으면 False."""
        if not any(self.stop_orders):
            return True
        sides = (self.position["side"], -self.position["side"]) if self.position else (None, None)
        for i, order_no in enumerate(self.stop_orders):
            if order_no and sides[i]:
                try:
                    self.brokers[i].cancel_stop(order_no, -sides[i], self.quantities[i])
                    self.log.write("INFO", "비상 스톱 취소", 종목=self.symbols[i], 주문번호=order_no)
                except Exception as error:
                    self.log.write("WARNING", "비상 스톱 취소 실패", 종목=self.symbols[i], 내용=str(error)[:300])
        for _ in range(3):
            deadline = time.time() + 1
            while time.time() < deadline:
                self.read_stream(0.2)
            if not self.brokers[0].count_open_orders():
                self.stop_orders = [None, None]
                return True
        self.log.write("WARNING", "비상 스톱 취소 미확인", 조치="3초 뒤 재시도")
        return False

    def exit_pair(self, reason):
        before, position = self.day_net, self.position
        super().exit_pair(reason)
        if self.position:  # 아직 청산이 끝나지 않아 다음 반복에서 다시 시도한다
            return
        net = self.day_net - before
        if reason in ("실시간 손절", "비상 스톱"):
            # 4주차 코드는 z 손절과 금액 손절에만 재진입 금지를 건다
            self.cooldown_until = add_minutes(datetime.now().strftime("%H%M") + "00", self.params["cooldown_minutes"])
        self.log.write("INFO", "청산 기록", 사유=reason, 손익=f"${net:+,.2f}", 보유중최저=f"${position.get('worst', 0.0):+,.2f}")
        overshoot = -net - self.params["max_loss"]
        if reason in STOP_REASONS and overshoot > self.risk["stop_overshoot_alert"]:
            self.log.write("WARNING", "손절 초과", 손익=f"${net:+,.2f}", 초과=f"${overshoot:,.2f}")
            self.stop_trading(f"손절이 한도보다 ${overshoot:,.0f} 더 잃음", "손절 초과")
        self.losses = self.losses + 1 if net < 0 else 0
        if self.rules["daily_flatten"] and (reason == "하루 한도" or self.day_net <= -self.params["daily_loss"]):
            self.stop_trading(f"하루 손익 ${self.day_net:+,.2f}", "하루 한도")
        if self.rules["loss_streak"] and self.losses >= self.risk["max_loss_streak"]:
            self.stop_trading(f"연속 손실 {self.losses}회", "연속 손실")
        self.save_state()

    def stop_trading(self, cause, rule):
        if self.stopped:
            return
        self.stopped = True
        self.save_state()
        self.log.write("ERROR", "거래 중단", 규칙=rule, 사유=cause, 조치="그날 신규 진입 안 함")

    def write_status(self):
        now = datetime.now().strftime("%H%M%S")
        due = not (int(now[2:4]) % 5 or now[4:] < "05" or now[:4] == self.last_status)
        super().write_status()
        if not due:
            return
        fields = {}
        if len(self.spreads) >= self.risk["half_life_window"]:
            fields["반감기"] = f"{half_life(self.spreads[-self.risk['half_life_window']:]):.1f}분"
        if self.quantities and None not in self.last.values():
            fields["예상변동액"] = f"${expected_move(self.prices(), self.quantities, self.point_values, self.volatility, self.risk['vol_hold_minutes']):,.0f}"
        if self.position:
            fields["보유중최저"] = f"${self.position.get('worst', 0.0):+,.2f}"
        fields["연속손실"] = f"{self.losses}회"
        self.log.write("INFO", "위험 상태", **fields)

    def save_state(self):
        """4주차 상태에 연속 손실 횟수를 더해 저장한다."""
        state = {
            "start_deposit": self.start_deposit, "entries": self.entries, "stopped": self.stopped,
            "cooldown_until": self.cooldown_until, "losses": getattr(self, "losses", 0),
            "quantities": self.quantities, "notional": self.notional, "cost": self.cost,
        }
        try:
            temp = self.state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state), encoding="utf-8")
            temp.replace(self.state_path)
        except Exception as error:
            self.stopped = True
            self.log.write("ERROR", "상태 저장 실패", 내용=str(error)[:300], 조치="신규 진입 중단")

    def load_state(self, deposit):
        super().load_state(deposit)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.losses = state.get("losses", 0)

    def finish(self, **extra):
        super().finish(**extra)
        try:
            daily = json.loads(self.daily_path.read_text(encoding="utf-8")) if self.daily_path.exists() else {}
            daily[self.today] = round(self.day_net, 2)
            self.daily_path.write_text(json.dumps(daily, indent=1), encoding="utf-8")
        except Exception as error:
            self.log.write("ERROR", "오류", 단계="하루 손익 저장", 내용=str(error)[:300])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    # 손절 테스트의 상태 파일과 하루 손익이 본 실행 기록에 섞이지 않게 따로 저장할 때 쓴다
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    global RESULTS_DIR
    # 실제 주문을 내므로 모의 환경에서만 실행한다
    env = load_env()
    if env["NH_ENV"].upper() != "DEMO":
        raise SystemExit("NH_ENV가 demo가 아니라서 실행하지 않습니다.")
    args = get_args()
    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    client = create_client("overseas")
    symbols = [leg["symbol"] for leg in config["legs"]]
    brokers = [StopLegBroker(client, symbol) for symbol in symbols]
    stream = PairStream(client, symbols)
    week04_live.RESULTS_DIR = RESULTS_DIR  # 4주차 로그 클래스가 이 경로에 로그를 남긴다
    log = AlertLog(datetime.now().strftime("%Y%m%d_%H%M%S"), env.get("DISCORD_WEBHOOK_URL"))

    trader = RiskTrader(config, client, brokers, stream, log)
    try:
        trader.run()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # 정리 중에 Ctrl+C를 또 누르면 청산이 끊긴다
        trader.shutdown()
    finally:
        log.close()


if __name__ == "__main__":
    main()
