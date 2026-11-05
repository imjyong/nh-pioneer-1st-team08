"""해외 선물 실시간 체결과 주문 상태를 WebSocket으로 받는다."""

import json

import websocket


class OverseasStream:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol
        self.ws = None

    def connect(self):
        # 키는 접속 1회용이라 재접속할 때마다 새로 받는다.
        host = self.client.base_url.replace("https://", "wss://")
        self.ws = websocket.create_connection(
            f"{host}/trade/ws-stream?access_key={self.client.get_websocket_key()}", timeout=10
        )
        self._send("0", {"action_code": "session_init", "session_token": self.client.get_token()})
        reply = json.loads(self.ws.recv())
        if reply.get("msg") != "INIT_SUCCESS":
            raise RuntimeError(f"세션 초기화 실패: {reply.get('msg')}")
        self._send("A", {"action_code": "FA", "action_sym": self.symbol})
        self._send("A", {"action_code": "O1"})

    def read(self, timeout=1):
        """실시간 메시지를 (구분 코드, 내용)으로 반환한다. 없으면 None을 반환한다."""
        self.ws.settimeout(timeout)
        try:
            raw = self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        message = json.loads(raw)
        if message.get("code") == "pingpong":  # 그대로 돌려보내지 않으면 세션이 끊긴다
            self.ws.send(raw)
            return None
        if message.get("code") == "RT_FAIL":  # 응답 원문에 계좌번호가 있어 사유만 남긴다
            raise RuntimeError(f"실시간 구독 실패: {message.get('msg')}")
        header = message.get("header")
        return (header["action_code"], message["body"]) if header else None

    def close(self):
        if self.ws:
            self.ws.close()

    def _send(self, action, body):
        self.ws.send(json.dumps({"header": {"api_ver": "1", "action": action}, "body": body}))
