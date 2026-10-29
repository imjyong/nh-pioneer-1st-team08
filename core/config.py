"""공통 경로와 설정 파일을 읽는다."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 원본 시세는 제3자 제공이 금지라서 저장소 밖에 둔다
DATA_DIR = REPO_ROOT.parent / "data" / "market"


def load_env():
    env = {}
    for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
