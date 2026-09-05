"""
dapp_controller_ewma.py
────────────────────────────────────────────────────────────────────
지수가중이동평균(EWMA, alpha=0.3) 기반 dApp 컨트롤러.

dapp_controller_chronos.py 와 동일한 구조이며,
predict_next의 "스케일링 → TCN 추론 → 역정규화" 부분만
EWMA(alpha=0.3)로 다음 값을 추정하는 로직으로 교체했다.
여전히 과거 데이터로부터 다음 값을 "예측"하므로 dApp으로 분류한다
(예측이 전혀 없는 baseline_controller_reactive.py와는 다름).

compute_weights / apply_weights / run 루프는 Chronos 버전과 동일하게 유지하여
가중치 산식 자체는 공정하게 비교되도록 한다.
"""

import time
import json
import os
import numpy as np
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────
FEATURES   = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK   = 10
EWMA_ALPHA = 0.3


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


# ── EWMA dApp 컨트롤러 ────────────────────────────────────────────
class EWMADAppController:
    """
    EWMA(alpha=0.3) 기반 dApp 컨트롤러.
    predict_next만 EWMA로 교체, 나머지 제어 로직은 Chronos 버전과 동일.
    """

    def __init__(self, ues: list, csv_path='kpi_live.csv'):
        self.ues      = ues
        self.csv_path = csv_path
        self.history  = {ue: [] for ue in ues}

        print(f"EWMA(alpha={EWMA_ALPHA}) dApp 컨트롤러 — TCN 모델 없음")
        print(f"대상 UE: {self.ues}")

    # ── 단일 UE 예측 (Chronos의 TCN 추론 자리를 EWMA로 교체) ─────
    def predict_next(self, ue: str, kpi: dict):
        row = [kpi.get(f, 0.0) for f in FEATURES]
        self.history[ue].append(row)

        # 히스토리 길이 관리 (Chronos 버전과 동일)
        if len(self.history[ue]) > LOOKBACK * 2:
            self.history[ue] = self.history[ue][-LOOKBACK:]
        if len(self.history[ue]) < LOOKBACK:
            return None

        seq = np.array(self.history[ue][-LOOKBACK:])   # (10, 6)

        # ── EWMA(alpha=0.3)로 다음 값 예측 (TCN 추론 대체) ──────
        level = seq[0]
        for row_t in seq[1:]:
            level = EWMA_ALPHA * row_t + (1 - EWMA_ALPHA) * level
        pred = level

        return dict(zip(FEATURES, pred))

    # ── 가중치 계산 (Fairness 우선, Chronos 버전과 동일) ─────────
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

    # ── 가중치 적용 (Chronos 버전과 동일) ─────────────────────────
    def apply_weights(self, weights: dict):
        with open('/tmp/dapp_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        print("  → /tmp/dapp_weights.json 저장 완료")

    # ── 메인 루프 (Chronos 버전과 동일) ───────────────────────────
    def run(self, interval=1.0):
        print("=" * 55)
        print(f"EWMA dApp 시작 | UE 수: {len(self.ues)}")
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
                        print(f"  [{ue}] SNR예측={pred.get('snr', 0):.1f}  "
                              f"BLER예측={pred.get('bler', 0):.4f}")

                weights = self.compute_weights(predictions)
                self.apply_weights(weights)
                print(f"[Cycle {cycle}] 가중치: {weights}\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\ndApp 종료")


# ── 엔트리포인트 ──────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('kpi_baseline.csv')
    ues = [str(u) for u in df['rnti'].unique()]
    ctrl = EWMADAppController(ues=ues, csv_path='kpi_live.csv')
    ctrl.run(interval=1.0)
