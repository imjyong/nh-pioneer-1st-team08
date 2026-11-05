"""해외 선물 계좌의 주문, 체결, 잔고를 처리한다."""

from core.broker import OrderRejected

SIDE_CODE = {1: "1", -1: "2"}  # 매매구분 1: 매수, 2: 매도
MARKET, LIMIT, STOP = "1", "2", "3"  # 해외주문유형 3: STOP-M
FILLED, REJECTED = "6", "9"  # 주문상태


class OverseasBroker:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol

    def place_order(self, side, order_type, price=None):
        """1계약을 주문하고 주문번호를 반환한다. price는 지정가 또는 STOP 가격이다."""
        payload = {
            "qt_div_cd": "1",
            "sym": self.symbol,
            "trd_div_cd": SIDE_CODE[side],
            "glb_ord_tp_cd": order_type,
            "ord_qty": "1",
            "ord_pric": f"{price:.2f}" if order_type == LIMIT else "0",
            "stop_pric": f"{price:.2f}" if order_type == STOP else "",
            "org_ord_no": "0",
        }
        body = self.client.post("/trade/v1/overseas/order", payload)
        order_no = (body.get("ord_no") or "").strip()
        if not order_no:
            raise OrderRejected(f"{body.get('err_cd')} {body.get('err_msg')}")
        return order_no

    def cancel_order(self, order_no, side, order_type):
        payload = {
            "qt_div_cd": "3",
            "sym": self.symbol,
            "trd_div_cd": SIDE_CODE[side],
            "glb_ord_tp_cd": order_type,
            "ord_qty": "1",
            "ord_pric": "0",
            "stop_pric": "",
            "org_ord_no": order_no,
        }
        self.client.post("/trade/v1/overseas/order", payload)

    def get_order(self, order_no):
        for row in self._order_history("0"):
            if row.get("occurs1_ord_no", "").strip().lstrip("0") == order_no.lstrip("0"):
                return {
                    "status": row.get("occurs1_ord_sts", "").strip(),
                    "filled": int(float(row.get("occurs1_exec_qty") or 0)),
                    "price": float(row.get("occurs1_exec_pric") or 0),
                    "remaining": int(float(row.get("occurs1_ord_rem_qty") or 0)),
                    "reason": (row.get("occurs1_rejt_rsn") or "").strip(),
                }
        return None

    def get_position(self):
        """보유 포지션을 (방향, 평균가격)으로 반환한다. 없으면 None."""
        for row in self._rows("/trade/v1/overseas/open-interest", {"nxt_key": ""}):
            if row.get("occurs1_sym", "").strip() == self.symbol and float(row.get("occurs1_opint_qty") or 0) > 0:
                side = 1 if row.get("occurs1_trd_tp", "").strip() == "1" else -1
                return side, float(row.get("occurs1_avg_pric") or 0)
        return None

    def count_open_orders(self):
        return len(self._order_history("3"))

    def get_deposit(self):
        body = self.client.post("/trade/v1/overseas/deposit", {"cur_cd": "USD", "cvs_div": "1"})
        return float(body.get("dps_aset_amt") or 0)

    def _order_history(self, inquiry):
        # 조회구분 0: 전체, 3: 미체결
        payload = {"nxt_key": "", "inq_div": inquiry, "prd_cd": "", "trd_div_cd": "0"}
        return self._rows("/trade/v1/overseas/order-history", payload)

    def _rows(self, path, payload):
        rows = self.client.post(path, payload).get("Occurs1") or []
        return [rows] if isinstance(rows, dict) else rows
