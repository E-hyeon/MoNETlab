"""
baseline_controller_reactive.py
────────────────────────────────────────────────────────────────────
예측 없이 현재 측정값만으로 실시간 가중치를 조정하는
Reactive Baseline (no AI prediction) 컨트롤러.

AI 예측이 전혀 없는 규칙 기반 로직이므로 dApp(CU/DU 단 AI 기반 실시간 제어)이
아니라 baseline/비교군으로 취급한다.

dapp_controller_chronos.py 와 동일한 구조이며,
TCN 예측값이 들어가던 자리에 "방금 측정한 현재 KPI"를 그대로 넣는다.
  - 모델/스케일러 로드 없음
  - LOOKBACK 워밍업 없음 (첫 사이클부터 즉시 동작)
compute_weights / apply_weights / run 루프는 Chronos 버전과 동일하게 유지하여
가중치 산식 자체는 공정하게 비교되도록 한다.
"""

import time
import json
import os
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────
FEATURES = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']


# ── CSV에서 최신 KPI 읽기 ─────────────────────────────────────────
def get_latest_kpi(csv_path: str, ue: str, features: list) -> dict:
    if not os.path.exists(csv_path):
        return {f: 0.0 for f in features}
    try:
        df = pd.read_csv(csv_path)
        rows = df[df['rnti'].astype(str) == str(ue)]
        if rows.empty:
            return {f: 0.0 for f in features}
        row = rows.iloc[-1]
        return {f: float(row.get(f, 0.0)) for f in features}
    except Exception as e:
        print(f"[KPI 읽기 오류] {e}")
        return {f: 0.0 for f in features}


# ── Reactive Baseline 컨트롤러 (AI 예측 없음) ──────────────────────
class ReactiveBaselineController:
    """
    예측기 없이 현재 SNR/BLER 측정값만으로 가중치를 조정하는 baseline 컨트롤러.
    predict_next 자리에 현재값을 그대로 반환하는 것만 다르고 나머지는 동일.
    """

    def __init__(self, ues: list, csv_path='kpi_live.csv'):
        self.ues      = ues
        self.csv_path = csv_path
        self.history  = {ue: [] for ue in ues}

        print("Reactive Baseline (no AI prediction) — 예측 모델 없음")
        print(f"대상 UE: {self.ues}")

    # ── 단일 UE 관측 (예측 대신 현재값 사용) ─────────────────────
    def predict_next(self, ue: str, kpi: dict):
        row = [kpi.get(f, 0.0) for f in FEATURES]
        self.history[ue].append(row)

        # 히스토리는 기록용으로만 유지 (제어에는 사용하지 않음)
        if len(self.history[ue]) > 20:
            self.history[ue] = self.history[ue][-10:]

        # 예측값 자리에 방금 측정한 현재값을 그대로 반환
        return {f: kpi.get(f, 0.0) for f in FEATURES}

    # ── 가중치 계산 (Chronos 버전과 동일) ────────────────────────
    def compute_weights(self, predictions: dict) -> dict:
        valid = {ue: p for ue, p in predictions.items() if p}
        if not valid:
            return {ue: round(1.0 / len(self.ues), 4) for ue in self.ues}

        scores = {}
        for ue, pred in valid.items():
            snr  = max(pred.get('snr',  1.0), 0.1)
            bler = pred.get('bler', 0.0)
            scores[ue] = (1.0 / snr) * (1.0 + bler * 5)

        total = sum(scores.values()) or 1.0
        return {ue: round(scores.get(ue, 0.0) / total, 4) for ue in self.ues}

    # ── 가중치 적용 (Chronos 버전과 동일) ────────────────────────
    def apply_weights(self, weights: dict):
        with open('/tmp/dapp_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        print("  → /tmp/dapp_weights.json 저장 완료")

    # ── 메인 루프 (Chronos 버전과 동일) ──────────────────────────
    def run(self, interval=1.0):
        print("=" * 55)
        print(f"Reactive Baseline 시작 | UE 수: {len(self.ues)}")
        print("=" * 55)
        cycle = 0
        try:
            while True:
                cycle += 1
                predictions = {}
                for ue in self.ues:
                    kpi  = get_latest_kpi(self.csv_path, ue, FEATURES)
                    pred = self.predict_next(ue, kpi)
                    predictions[ue] = pred
                    if pred:
                        print(f"  [{ue}] SNR현재={pred.get('snr', 0):.1f}  "
                              f"BLER현재={pred.get('bler', 0):.4f}")

                weights = self.compute_weights(predictions)
                self.apply_weights(weights)
                print(f"[Cycle {cycle}] 가중치: {weights}\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n종료")


# ── 엔트리포인트 ──────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('kpi_live.csv')
    ues = [str(u) for u in df['rnti'].unique()]
    ctrl = ReactiveBaselineController(ues=ues, csv_path='kpi_live.csv')
    ctrl.run(interval=1.0)
