"""시뮬레이션 체결과 NH선물 계좌 주문을 같은 방식으로 다룬다."""

from datetime import datetime

SIDE_CODE = {1: "1", -1: "2"}  # 매매구분 1: 매수, 2: 매도


class OrderRejected(Exception):
    pass


class NHBroker:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol

    def place_order(self, side, price=None):
        """1계약을 주문하고 주문번호를 반환한다. 가격이 없으면 시장가로 주문한다."""
        is_market = price is None
        payload = {
            "sym": self.symbol,
            "trd_div": SIDE_CODE[side],
            "ord_tp_cd": "2" if is_market else "1",
            "ord_qty": "1",
            "ord_pric": "0" if is_market else f"{price:.2f}",
            # 시장가는 FOK, FAK 모두 거부된다 (10/30 장중 확인). 시장가, 지정가 모두 FAS로 주문한다.
            "exec_cnd_cd": "1",
        }
        body = self.client.post("/trade/v1/domestic/order", payload)
        order_no = (body.get("ord_no") or "").strip()
        if not order_no:
            raise OrderRejected(f"{body.get('msg_cd')} {body.get('msg_cntn')}")
        return order_no

    def cancel_order(self, order_no, side):
        payload = {"sym": self.symbol, "trd_div": SIDE_CODE[side], "ord_qty": "1", "org_ord_no": order_no}
        self.client.post("/trade/v1/domestic/cancel", payload)

    def get_order(self, order_no):
        for row in self._order_history(exec_div="0"):
            if row.get("occurs1_ord_no", "").strip().lstrip("0") == order_no.lstrip("0"):
                return {
                    "filled": int(float(row.get("occurs1_exec_qty") or 0)),
                    # 평균체결가격은 쓰지 않는 필드라 0으로 온다 (운영진 답변). 체결가격을 쓴다.
                    "price": float(row.get("occurs1_exec_pric") or 0),
                    "remaining": int(float(row.get("occurs1_rem_ord_qty") or 0)),
                    "rejected": bool((row.get("occurs1_rejt_rsn_cd") or "").strip("0 ")),
                    "reason": (row.get("occurs1_ord_rejt_rsn") or "").strip(),
                }
        return None

    def get_position(self):
        """보유 포지션을 (방향, 평균가격)으로 반환한다. 없으면 None."""
        for row in self._rows("/trade/v1/domestic/open-interest", {"nxt_key": ""}):
            if row.get("occurs1_sym", "").strip() == self.symbol and float(row.get("occurs1_opint_qty") or 0) > 0:
                side = 1 if "매수" in row.get("occurs1_trd_div_nm_v2", "") else -1
                return side, float(row.get("occurs1_avg_pric") or 0)
        return None

    def count_open_orders(self):
        return sum(1 for row in self._order_history(exec_div="2") if int(float(row.get("occurs1_rem_ord_qty") or 0)) > 0)

    def get_deposit(self):
        # 휴장일 날짜로 조회하면 오류 없이 0으로 온다 (team-06 제보).
        body = self.client.post("/trade/v1/domestic/deposit", {"biz_dt": datetime.now().strftime("%Y%m%d")})
        return float(body.get("dps_aset_tot_amt") or 0)

    def _order_history(self, exec_div):
        # 체결구분 0: 전체, 2: 미체결
        payload = {"trd_div": "0", "exec_div": exec_div, "trd_tm_div_cd": "1", "ord_div_cd": "0", "nxt_key": ""}
        return self._rows("/trade/v1/domestic/order-history", payload)

    def _rows(self, path, payload):
        rows = self.client.post(path, payload).get("Occurs1") or []
        return [rows] if isinstance(rows, dict) else rows


class PaperBroker:
    """주문하지 않고 현재 호가로 바로 체결된 것으로 계산한다."""

    def __init__(self):
        self.quote = None  # (매도1호가, 매수1호가). 실행 코드가 조회할 때마다 넣는다.
        self.orders = {}
        self.position = None

    def place_order(self, side, price=None):
        ask, bid = self.quote
        fill_price = price if price is not None else (ask if side == 1 else bid)
        order_no = f"P{len(self.orders) + 1:05d}"
        self.orders[order_no] = {"filled": 1, "price": fill_price, "remaining": 0, "rejected": False, "reason": ""}
        if self.position and self.position[0] == -side:
            self.position = None
        else:
            self.position = (side, fill_price)
        return order_no

    def cancel_order(self, order_no, side):
        pass

    def get_order(self, order_no):
        return self.orders.get(order_no)

    def get_position(self):
        return self.position

    def count_open_orders(self):
        return 0

    def get_deposit(self):
        return None
