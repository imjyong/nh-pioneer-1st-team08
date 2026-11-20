"""야후 MNQ=F 60분봉과 NH MNQZ26 1분봉을 같은 시각으로 대조한다 (한국 14·16·18시 수익률)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
nh = pd.read_csv(ROOT / "data/market/MNQZ26_1min.csv", dtype={"date": str, "time": str})
nh["ts"] = pd.to_datetime(nh["date"] + nh["time"].str.zfill(6), format="%Y%m%d%H%M%S").dt.tz_localize("Asia/Seoul")
nh = nh.set_index("ts")["open"]  # 그 분 봉의 시가 = 그 시각 가격

y = pd.read_csv(ROOT / "data/market/week05_yahoo/MNQ=F_60m_adj.csv", index_col=0)
y.index = pd.to_datetime(y.index, utc=True).tz_convert("Asia/Seoul")
y = y[y.index.minute == 0]["Open"]

rows = []
for d in sorted(set(y.index.normalize()) & set(nh.index.normalize())):
    if d < pd.Timestamp("2026-09-22", tz="Asia/Seoul"):  # 9월물 만기(9/18) 뒤부터 둘 다 12월물
        continue
    r = {"date": d.date()}
    for src, s in [("nh", nh), ("yh", y)]:
        p = {h: s.get(d + pd.Timedelta(hours=h), np.nan) for h in (14, 16, 18)}
        r[f"{src} 14→16"] = np.log(p[16] / p[14])
        r[f"{src} 14→18"] = np.log(p[18] / p[14])
        r[f"{src} 14시"] = p[14]
    rows.append(r)
df = pd.DataFrame(rows).dropna()
print(f"비교일 {len(df)}일 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
for k in ["14→16", "14→18"]:
    diff = (df[f"nh {k}"] - df[f"yh {k}"]) * 1e4
    print(f"{k}: 수익률 상관 {df[f'nh {k}'].corr(df[f'yh {k}']):.4f}, 차이 평균 {diff.mean():+.2f}bp, 차이 절댓값 최대 {diff.abs().max():.2f}bp")
lvl = (df["nh 14시"] / df["yh 14시"] - 1) * 1e4
print(f"14시 가격 수준 차이: 평균 {lvl.mean():+.2f}bp, 최대 절댓값 {lvl.abs().max():.2f}bp")
