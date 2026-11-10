"""두 레그의 실시간 체결과 주문 상태를 WebSocket 하나로 받는다."""

from week03.stream import OverseasStream


class PairStream(OverseasStream):
    def __init__(self, client, symbols):
        super().__init__(client, symbols[0])
        self.others = symbols[1:]

    def connect(self):
        super().connect()
        for symbol in self.others:
            self._send("A", {"action_code": "FA", "action_sym": symbol})
