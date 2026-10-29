"""NH선물 API 인증과 요청을 처리한다."""

import time

import requests

from core.config import load_env


class NHClient:
    def __init__(self, base_url, app_key, app_secret):
        self.base_url = base_url
        self.app_key = app_key
        self.app_secret = app_secret
        self.token = None
        self.token_expires = 0

    def get_token(self):
        # 토큰은 24시간 유효하므로 만료 직전에만 요청이 실패하지 않도록 5분 일찍 갱신한다.
        if time.time() > self.token_expires - 300:
            body = self._request("/auth-service/v1/token", data={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            })
            self.token = body["access_token"]
            self.token_expires = time.time() + int(body["expires_in"])
        return self.token

    def post(self, path, payload):
        headers = {"Authorization": f"Bearer {self.get_token()}"}
        return self._request(path, json=payload, headers=headers)

    def _request(self, path, **kwargs):
        res = requests.post(f"{self.base_url}{path}", timeout=30, **kwargs)
        if res.status_code != 200:
            raise RuntimeError(f"{path} 실패: {res.status_code} {res.text}")
        return res.json()


def create_client(market="domestic"):
    env = load_env()
    mode = env["NH_ENV"].upper()  # DEMO 또는 REAL
    prefix = f"NH_{market.upper()}_{mode}"
    return NHClient(env[f"NH_{mode}_URL"], env[f"{prefix}_APP_KEY"], env[f"{prefix}_APP_SECRET"])
