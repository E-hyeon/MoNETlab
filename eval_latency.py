"""
eval_latency.py — Chronos TCN dApp vs EWMA dApp vs MLP dApp vs Reactive Baseline
predict_next() 추론 지연시간 비교.

실제 컨트롤러 클래스를 그대로 임포트해서 predict_next를 반복 호출하며 측정한다
(재구현이 아니라 운영 코드 경로 그대로 측정).
"""

import time
import numpy as np
import pandas as pd

from dapp_controller_chronos import ChronosDAppController
from dapp_controller_ewma import EWMADAppController
from dapp_controller_mlp import MLPDAppController
from baseline_controller_reactive import ReactiveBaselineController

KPI_CSV  = 'kpi_live.csv'
FEATURES = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
N_WARMUP = 15
N_TRIALS = 200


def bench(name: str, ctrl, ue: str, sub: pd.DataFrame):
    # LOOKBACK 워밍업 (Reactive Baseline은 워밍업 불필요하지만 동일 절차로 통일)
    for i in range(min(N_WARMUP, len(sub))):
        row = sub.iloc[i]
        kpi = {f: float(row[f]) for f in FEATURES}
        ctrl.predict_next(ue, kpi)

    lats = []
    for i in range(N_TRIALS):
        row = sub.iloc[i % len(sub)]
        kpi = {f: float(row[f]) for f in FEATURES}
        t0 = time.perf_counter()
        ctrl.predict_next(ue, kpi)
        lats.append((time.perf_counter() - t0) * 1000)

    lats = np.array(lats)
    print(f"  {name:<20} 평균={lats.mean():>8.4f}ms  중앙값={np.median(lats):>8.4f}ms  "
          f"P95={np.percentile(lats, 95):>8.4f}ms  P99={np.percentile(lats, 99):>8.4f}ms")


def main():
    df = pd.read_csv(KPI_CSV)
    df['rnti'] = df['rnti'].astype(str)
    ue = df['rnti'].value_counts().idxmax()
    sub = df[df['rnti'] == ue].reset_index(drop=True)
    print(f"대상 UE: {ue}  (표본 {len(sub)}행, {KPI_CSV})\n")

    print("모델 로드 중...")
    controllers = [
        ("Chronos TCN dApp", ChronosDAppController(ues=[ue], csv_path=KPI_CSV)),
        ("EWMA dApp",        EWMADAppController(ues=[ue], csv_path=KPI_CSV)),
        ("MLP dApp",         MLPDAppController(ues=[ue], csv_path=KPI_CSV)),
        ("Reactive Baseline", ReactiveBaselineController(ues=[ue], csv_path=KPI_CSV)),
    ]

    print(f"\n추론 지연시간 비교 (predict_next() 1회, n={N_TRIALS}회)")
    print("=" * 90)
    for name, ctrl in controllers:
        bench(name, ctrl, ue, sub)
    print("=" * 90)


if __name__ == '__main__':
    main()
