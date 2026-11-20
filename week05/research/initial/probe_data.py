"""NH 차트 API로 상품별 1분봉의 조회 가능 기간과 시간대별 데이터 밀도를 확인한다. 주문은 내지 않는다."""
import sys, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[4] / "nh-pioneer-1st-team08"
sys.path.insert(0, str(REPO))
from core.client import create_client
from core.market_data import fetch_chart
from core.bars import save_bars
from core.config import DATA_DIR

ITEMS = [("FMTWZ26","FERX"),("NIYZ26","FCME"),("SNKZ26","FSGX"),("HTIX26","FHKE"),
         ("MNQZ26","FCME"),("XNVDAZ26","FCME"),("HSIX26","FHKE"),("SFCX26","FSGX"),("FMINZ26","FERX")]
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

client = create_client("overseas")
out = DATA_DIR / "week05_probe"; out.mkdir(parents=True, exist_ok=True)
for sym, exch in ITEMS:
    try:
        bars = fetch_chart(client, "/trade/v1/overseas/chart-minute", {"exch_cd": exch, "sym": sym, "dcnt": "1"}, COUNT)
    except Exception as e:
        print(f"{sym}: 오류 {type(e).__name__} {str(e)[:120]}"); continue
    if not bars:
        print(f"{sym}: 데이터 없음"); continue
    save_bars(bars, out / f"{sym}_1min.csv")
    days = sorted({b.date for b in bars})
    hours = collections.Counter(b.time[:2] for b in bars)
    vol0 = sum(1 for i in range(1, len(bars)) if bars[i].cum_volume == bars[i-1].cum_volume and bars[i].date == bars[i-1].date)
    hrs = " ".join(f"{h}:{hours[h]}" for h in sorted(hours))
    print(f"{sym}: {len(bars)}봉, {days[0]} {bars[0].time} ~ {days[-1]} {bars[-1].time}, {len(days)}일 | 시간대별 봉 수 {hrs}")
