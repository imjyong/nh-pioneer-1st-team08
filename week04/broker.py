"""스프레드 한 레그의 주문과 잔고를 처리한다."""

from core.broker import OrderRejected
from week03.broker import MARKET, SIDE_CODE, OverseasBroker


class LegBroker(OverseasBroker):
    def place_market(self, side, quantity):
        """지정한 수량을 시장가로 주문하고 주문번호를 반환한다."""
        payload = {
            "qt_div_cd": "1",
            "sym": self.symbol,
            "trd_div_cd": SIDE_CODE[side],
            "glb_ord_tp_cd": MARKET,
            "ord_qty": str(quantity),
            "ord_pric": "0",
            "stop_pric": "",
            "org_ord_no": "0",
        }
        body = self.client.post("/trade/v1/overseas/order", payload)
        order_no = (body.get("ord_no") or "").strip()
        if not order_no:
            raise OrderRejected(f"{body.get('err_cd')} {body.get('err_msg')}")
        return order_no

    def get_position(self):
        """보유 포지션을 (방향, 수량, 평균가격)으로 반환한다. 없으면 None."""
        for row in self._rows("/trade/v1/overseas/open-interest", {"nxt_key": ""}):
            quantity = int(float(row.get("occurs1_opint_qty") or 0))
            if row.get("occurs1_sym", "").strip() == self.symbol and quantity > 0:
                side = 1 if row.get("occurs1_trd_tp", "").strip() == "1" else -1
                return side, quantity, float(row.get("occurs1_avg_pric") or 0)
        return None
