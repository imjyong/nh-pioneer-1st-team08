"""스프레드 한 레그의 주문에 비상 스톱(STOP-M)을 더한다."""

from core.broker import OrderRejected
from week03.broker import SIDE_CODE, STOP
from week04.broker import LegBroker


class StopLegBroker(LegBroker):
    def place_stop(self, side, quantity, price):
        payload = {
            "qt_div_cd": "1",
            "sym": self.symbol,
            "trd_div_cd": SIDE_CODE[side],
            "glb_ord_tp_cd": STOP,
            "ord_qty": str(quantity),
            "ord_pric": "0",
            "stop_pric": f"{price:.2f}",
            "org_ord_no": "0",
        }
        body = self.client.post("/trade/v1/overseas/order", payload)
        order_no = (body.get("ord_no") or "").strip()
        if not order_no:
            raise OrderRejected(f"{body.get('err_cd')} {body.get('err_msg')}")
        return order_no

    def cancel_stop(self, order_no, side, quantity):
        payload = {
            "qt_div_cd": "3",
            "sym": self.symbol,
            "trd_div_cd": SIDE_CODE[side],
            "glb_ord_tp_cd": STOP,
            "ord_qty": str(quantity),
            "ord_pric": "0",
            "stop_pric": "",
            "org_ord_no": order_no,
        }
        self.client.post("/trade/v1/overseas/order", payload)
