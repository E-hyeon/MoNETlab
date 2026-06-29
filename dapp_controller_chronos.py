"""
dapp_controller_chronos.py
────────────────────────────────────────────────────────────────────
BigDL Chronos TCNForecaster 기반 dApp 컨트롤러.
  - chronos_forecaster/  : 학습된 TCN 모델 (chronos_final.py로 생성)
  - scaler_chronos.pkl   : StandardScaler

predict_next만 Chronos로 교체.
compute_weights / apply_weights / run 루프는 기존 그대로 유지.
"""

import time
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster import TCNForecaster

# ── 설정 ──────────────────────────────────────────────────────────
FEATURES      = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK      = 10
FORECASTER_PATH = 'chronos_forecaster'
SCALER_PATH   = 'scaler_chronos.pkl'


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


# ── Chronos dApp 컨트롤러 ─────────────────────────────────────────
class ChronosDAppController:
    """
    BigDL Chronos TCNForecaster 기반 dApp 컨트롤러.
    predict_next만 Chronos로 교체, 나머지 제어 로직은 동일.
    """

    def __init__(self, ues: list, csv_path='kpi_live.csv'):
        self.ues      = ues
        self.csv_path = csv_path
        self.history  = {ue: [] for ue in ues}

        # ── 모델 & 스케일러 로드 ───────────────────────────────
        self.scaler = joblib.load(SCALER_PATH)

        # 인스턴스 생성 후 load (Chronos 2.2.0 방식)
        from bigdl.chronos.data import TSDataset
        df_tmp = pd.read_csv('kpi_baseline.csv', parse_dates=['timestamp'])
        df_tmp = df_tmp.sort_values('timestamp').reset_index(drop=True)
        df_tmp[FEATURES] = df_tmp[FEATURES].fillna(0)
        tsdata_tmp, _, _ = TSDataset.from_pandas(
            df_tmp, dt_col='timestamp', target_col=FEATURES,
            with_split=True, val_ratio=0.15, test_ratio=0.15
        )
        scaler_tmp = StandardScaler()
        tsdata_tmp.scale(scaler_tmp, fit=True)
        tsdata_tmp.roll(lookback=LOOKBACK, horizon=1)

        self.forecaster = TCNForecaster.from_tsdataset(tsdata_tmp)
        self.forecaster.load(FORECASTER_PATH)
        print(f"Chronos TCN 모델 로드 완료")
        print(f"대상 UE: {self.ues}")

    # ── 단일 UE 예측 (Chronos로 교체된 부분) ─────────────────────
    def predict_next(self, ue: str, kpi: dict):
        row = [kpi.get(f, 0.0) for f in FEATURES]
        self.history[ue].append(row)

        # 히스토리 길이 관리
        if len(self.history[ue]) > LOOKBACK * 2:
            self.history[ue] = self.history[ue][-LOOKBACK:]
        if len(self.history[ue]) < LOOKBACK:
            return None

        # 스케일링
        seq = np.array(self.history[ue][-LOOKBACK:])         # (10, 6)
        seq_scaled = self.scaler.transform(seq)              # (10, 6)

        # Chronos 입력 형식: (1, lookback, n_features)
        x = seq_scaled[np.newaxis, :, :].astype(np.float32)                     # (1, 10, 6)

        # 예측
        pred_scaled = self.forecaster.predict(x)             # (1, 1, 6)
        pred_scaled = pred_scaled.reshape(1, len(FEATURES))  # (1, 6)

        # 역정규화
        pred = self.scaler.inverse_transform(pred_scaled)[0]
        return dict(zip(FEATURES, pred))

    # ── 가중치 계산 (Fairness 우선, 기존과 동일) ─────────────────
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

    # ── 가중치 적용 (기존과 동일) ─────────────────────────────────
    def apply_weights(self, weights: dict):
        with open('/tmp/dapp_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        print("  → /tmp/dapp_weights.json 저장 완료")

    # ── 메인 루프 (기존과 동일) ───────────────────────────────────
    def run(self, interval=1.0):
        print("=" * 55)
        print(f"Chronos dApp 시작 | UE 수: {len(self.ues)}")
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
    df = pd.read_csv('kpi_live.csv')
    ues = [str(u) for u in df['rnti'].unique()]
    ctrl = ChronosDAppController(ues=ues, csv_path='kpi_live.csv')
    ctrl.run(interval=1.0)
