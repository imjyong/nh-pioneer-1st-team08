"""동결한 아시아 AI 흩어짐 신호로 MNQ 모의 주문 파이프라인을 실행한다.

검증된 전략의 실거래가 아니라, 신호 계산부터 주문·체결·청산까지 확인하는 운영 실험이다.
- 신호: signal_log.csv의 그날 실시간 행 (확률 < 0.30이면 매수, 아니면 거래 안 함)
- 연구상 보유 구간은 미국 정규장 시가부터 종가까지지만, 해외 모의 서버가 24:00에 닫혀 23:30에 진입해 23:45에 청산한다
- 주문 안전 규칙: 주문번호를 받은 주문은 체결·거부·취소가 확정될 때까지 추적한다. 확정되지 않으면 취소하고,
  실제 포지션과 미체결을 다시 조회한 뒤에만 다음 주문을 낸다. 청산 방향은 항상 실제 포지션의 반대다
"""

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from core.client import create_client
from core.config import load_env
from week03.broker import FILLED, MARKET, REJECTED, OverseasBroker
from week03.stream import OverseasStream

RESULTS = Path(__file__).with_name("results")
SYMBOL, TICK, TICK_VALUE = "MNQZ26", 0.25, 0.5
ENTRY, ENTRY_DEADLINE, EXIT = "233000", "233500", "234500"
LAST_NEW_ORDER, HARD_CUTOFF = "235230", "235300"  # 이후에는 새 주문을 내지 않고, 컷오프 뒤에는 조회·취소도 멈춘다
FRESH_SECONDS = 10  # 이보다 오래된 체결가뿐이면 진입하지 않는다

logger = logging.getLogger("week05")


class Log:
    def __init__(self, run_id):
        RESULTS.mkdir(exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        for h in (logging.StreamHandler(), logging.FileHandler(RESULTS / f"live_{run_id}.log", encoding="utf-8")):
            h.setFormatter(fmt)
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        self.jsonl = open(RESULTS / f"events_{run_id}.jsonl", "a", encoding="utf-8")
        self.run_id = run_id

    def write(self, level, event, **fields):
        logger.log(getattr(logging, level), f"[{event}] " + " | ".join(f"{k}={v}" for k, v in fields.items()))
        self.jsonl.write(json.dumps({"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                                     "run_id": self.run_id, "level": level, "event": event, "data": fields},
                                    ensure_ascii=False) + "\n")
        self.jsonl.flush()


def now_hms():
    return datetime.now().strftime("%H%M%S")


def today_signal(path, today):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["구분"] == "실시간" and r["아시아날짜"] == today]
    return rows[-1] if rows else None


class Runner:
    def __init__(self, client, broker, stream, log):
        self.client, self.broker, self.stream, self.log = client, broker, stream, log
        self.stream_ok = False
        self.last = None
        self.last_at = 0.0
        self.quote_keys_logged = False
        self.fills = {}         # 주문번호: (상태, 체결가)
        self.last_fill = None

    # 스트림
    def connect(self):
        try:
            self.stream.connect()
            self.stream_ok = True
        except Exception as e:
            self.stream_ok = False
            self.log.write("WARNING", "스트림 접속 실패", 내용=str(e)[:200], 조치="REST로만 확인")

    def pump(self, timeout=0.2):
        if not self.stream_ok:
            time.sleep(timeout)
            return
        try:
            msg = self.stream.read(timeout)
        except Exception as e:
            self.stream_ok = False
            self.log.write("WARNING", "스트림 끊김", 내용=str(e)[:200], 조치="REST로만 확인")
            return
        if not msg:
            return
        code, body = msg
        if code == "FA" and "last_pric" in body:
            self.last, self.last_at = float(body["last_pric"]), time.time()
            if not self.quote_keys_logged:  # 호가 필드가 있는지 한 번만 기록 (시세 메시지라 계좌 정보 없음)
                self.quote_keys_logged = True
                self.log.write("INFO", "시세 필드", 키=",".join(sorted(body.keys())))
        elif code == "O1" and "ord_no" in body:  # REST와 같게 앞자리 0을 뺀 번호로 맞춘다
            self.fills[body["ord_no"].strip().lstrip("0")] = (body["ord_sts"].strip(), float(body.get("exec_pric") or 0))

    def wait_until(self, hhmmss):
        while now_hms() < hhmmss:
            self.pump(1)

    # 계좌
    def snapshot(self):
        return self.broker.get_position(), self.broker.count_open_orders()

    # 주문
    def resolve(self, order_no, side):
        """주문이 체결·거부·취소로 확정될 때까지 추적한다. (상태, 체결가) 반환. 상태: 체결/거부/취소/불명."""
        key = order_no.lstrip("0")
        deadline = time.time() + 10
        while time.time() < deadline and now_hms() < HARD_CUTOFF:
            self.pump(0.2)
            st, px = self.fills.get(key, (None, 0))
            if st == FILLED:
                return "체결", px
            if st == REJECTED:
                return "거부", 0
        for _ in range(5):  # REST로 확인
            if now_hms() >= HARD_CUTOFF:
                return "불명", 0
            o = self.get_order_safe(order_no)
            if o and o["status"] == FILLED:
                return "체결", o["price"]
            if o and o["status"] == REJECTED:
                return "거부", 0
            time.sleep(1)
        if now_hms() >= HARD_CUTOFF:
            return "불명", 0
        # 확정되지 않으면 취소 요청 후 다시 확인한다
        try:
            self.broker.cancel_order(order_no, side, MARKET)
            self.log.write("WARNING", "주문 취소 요청", 주문번호=order_no, 사유="15초 안에 확정 안 됨")
        except Exception as e:
            self.log.write("WARNING", "주문 취소 실패", 주문번호=order_no, 내용=str(e)[:200])
        inactive = 0  # 취소 판정은 두 번 연속 같은 상태일 때만 (NH 취소 상태 코드를 문서로 확인하지 못해서)
        for _ in range(10):
            if now_hms() >= HARD_CUTOFF:
                return "불명", 0
            time.sleep(1)
            o = self.get_order_safe(order_no)
            if o and o["status"] == FILLED:
                return "체결", o["price"]
            if o and o["status"] == REJECTED:
                return "거부", 0
            inactive = inactive + 1 if o and o["filled"] == 0 and o["remaining"] == 0 else 0
            if inactive >= 2:
                return "취소", 0
        return "불명", 0

    def get_order_safe(self, order_no):
        try:
            return self.broker.get_order(order_no)
        except Exception as e:
            self.log.write("WARNING", "주문 조회 실패", 주문번호=order_no, 내용=str(e)[:200])
            return None

    def order(self, side, purpose):
        """미체결이 없을 때만 1계약 시장가 주문을 내고 확정 결과를 반환한다."""
        pos, open_orders = self.snapshot()
        if open_orders:
            self.log.write("ERROR", "주문 안 함", 용도=purpose, 사유=f"미체결 {open_orders}건 남음")
            return "불명", 0
        ref, ref_age = self.last, time.time() - self.last_at if self.last_at else None
        sent = time.time()
        try:
            order_no = self.broker.place_order(side, MARKET)
        except Exception as e:
            self.log.write("ERROR", "주문 응답", 용도=purpose, 상태="실패", 내용=str(e)[:200])
            return "실패", 0
        self.log.write("INFO", "주문 응답", 용도=purpose, 방향="매수" if side == 1 else "매도", 상태="접수", 주문번호=order_no,
                       직전체결가=f"{ref:.2f}" if ref else "없음",
                       직전체결가경과=f"{ref_age:.1f}초" if ref_age is not None else "없음")
        status, px = self.resolve(order_no, side)
        fields = dict(용도=purpose, 결과=status, 주문번호=order_no, 확인지연=f"{time.time() - sent:.1f}초")
        if status == "체결":
            fields["체결가"] = f"{px:.2f}"
            if ref:  # 스프레드를 반영한 슬리피지가 아니라 직전 체결가 대비 차이
                fields["직전체결가대비"] = f"{round(side * (px - ref) / TICK):+d}틱(스프레드 미반영)"
        self.log.write("INFO" if status == "체결" else "ERROR", "주문 결과", **fields)
        if status == "체결":
            self.last_fill = px
        return status, px

    def flatten(self, purpose):
        """실제 포지션이 없어질 때까지(하드 컷오프 전까지) 포지션 반대 방향으로만 청산한다."""
        while now_hms() < LAST_NEW_ORDER:
            try:
                pos, open_orders = self.snapshot()
            except Exception as e:
                self.log.write("WARNING", "잔고 조회 실패", 내용=str(e)[:200])
                time.sleep(2)
                continue
            if not pos and not open_orders:
                return True
            if open_orders:
                self.log.write("WARNING", "청산 대기", 사유=f"미체결 {open_orders}건, 확정될 때까지 다음 주문 안 함")
                time.sleep(2)
                continue
            status, _ = self.order(-pos[0], purpose)
            if status == "불명":
                time.sleep(2)
        try:
            pos, open_orders = self.snapshot()
        except Exception:
            return False
        return not pos and not open_orders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal", default=RESULTS / "signal_log.csv")
    args = p.parse_args()

    if load_env()["NH_ENV"].upper() != "DEMO":  # 실제 주문을 내므로 모의 환경에서만 실행한다
        raise SystemExit("NH_ENV가 demo가 아니라서 실행하지 않습니다.")
    today = datetime.now().strftime("%Y-%m-%d")
    log = Log(datetime.now().strftime("%Y%m%d_%H%M%S"))

    sig = today_signal(args.signal, today)
    if not sig:
        log.write("WARNING", "거래 안 함", 사유="오늘 실시간 신호 없음", 신호파일=str(args.signal))
        return
    log.write("INFO", "신호", 아시아날짜=sig["아시아날짜"], 계산시각=sig["계산시각_KST"], 방법1확률=sig["방법1확률"],
              문턱=sig["문턱"], 결정=sig["결정"], 사용종목수=sig["사용종목수"], 종가대체=sig["종가대체"])
    try:
        prob = float(sig["방법1확률"])
    except ValueError:
        prob = float("nan")
    if not sig["결정"].startswith("매수") or "주의" in sig["결정"] or not prob < float(sig["문턱"]):
        log.write("INFO", "거래 안 함", 사유=sig["결정"], 확률=sig["방법1확률"])
        return
    if now_hms() > ENTRY_DEADLINE:
        log.write("WARNING", "거래 안 함", 사유=f"진입 마감({ENTRY_DEADLINE}) 지나서 실행됨")
        return

    client = create_client("overseas")
    broker = OverseasBroker(client, SYMBOL)
    run = Runner(client, broker, OverseasStream(client, SYMBOL), log)
    pos, open_orders = run.snapshot()
    if pos or open_orders:
        log.write("ERROR", "계좌 동기화", 포지션=pos or "없음", 미체결=f"{open_orders}건", 조치="실행 안 함")
        return
    log.write("INFO", "계좌 동기화", 예탁자산=f"${broker.get_deposit():,.2f}", 포지션="없음", 미체결="0건")

    try:
        while now_hms() < "232800":  # 연결 유지 시간을 줄이려고 진입 2분 전에 접속한다
            time.sleep(1)
        run.connect()
        run.wait_until(ENTRY)
        t0 = time.time()
        while time.time() - run.last_at > FRESH_SECONDS and time.time() - t0 < 60 and now_hms() < ENTRY_DEADLINE:
            run.pump(1)
        if time.time() - run.last_at > FRESH_SECONDS:
            log.write("WARNING", "거래 안 함", 사유=f"최근 {FRESH_SECONDS}초 안의 체결 시세 없음")
            return
        log.write("INFO", "시세", 체결가=f"{run.last:.2f}")
        status, entry = run.order(1, "진입")
        pos, _ = run.snapshot()  # 주문 결과와 상관없이 실제 잔고를 기준으로 판단한다
        if not pos:
            log.write("WARNING" if status != "체결" else "ERROR", "진입 후 포지션 없음", 주문결과=status)
            return
        entry = pos[1] or entry
        log.write("INFO", "보유", 방향="매수" if pos[0] == 1 else "매도", 평균가=f"{entry:.2f}")
        run.wait_until(EXIT)
        run.last_fill = None
        if run.flatten("시간 청산") and run.last_fill:
            ticks = round(pos[0] * (run.last_fill - entry) / TICK)
            log.write("INFO", "청산 완료", 진입가=f"{entry:.2f}", 청산가=f"{run.last_fill:.2f}",
                      손익=f"{ticks:+d}틱(${ticks * TICK_VALUE:+,.2f})", 수수료="별도")
    except KeyboardInterrupt:
        log.write("WARNING", "사용자 중단")
    finally:
        try:
            pos, open_orders = run.snapshot()
            if pos or open_orders:
                log.write("WARNING", "비상 청산", 포지션=pos or "없음", 미체결=f"{open_orders}건")
                run.flatten("비상 청산")  # 23:52:30 이후면 새 주문 없이 바로 끝난다
            pos, open_orders = run.snapshot()
            log.write("INFO" if not pos and not open_orders else "ERROR", "최종 확인", 포지션=pos or "없음",
                      미체결=f"{open_orders}건", 예탁자산=f"${broker.get_deposit():,.2f}", 호출=f"{client.request_count}회",
                      조치="없음" if not pos and not open_orders else "잔고를 직접 다시 조회해 확인 필요")
        except Exception as e:
            log.write("ERROR", "최종 확인 실패", 내용=str(e)[:200], 조치="잔고를 직접 다시 조회해 확인 필요")
        try:
            run.stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
