import torch
import numpy as np
import joblib
import time
import json
import os
import pandas as pd

# ── 모델 클래스 ────────────────────────────────
class ChannelRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, output_size=None):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = torch.nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── CSV에서 최신 KPI 읽기 ──────────────────────
def get_latest_kpi(csv_path, ue, features):
    """kpi_baseline.csv 또는 gnb_live.log 파싱 결과에서 최신 행 반환"""
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


# ── dApp 컨트롤러 ─────────────────────────────
class DAppController:
    def __init__(self, ues, csv_path='kpi_live.csv', seq_length=10):
        self.ues = ues
        self.csv_path = csv_path
        self.seq_length = seq_length
        self.features = joblib.load('features.pkl')
        self.models = {}
        self.scalers = {}
        self.history = {ue: [] for ue in ues}

        for ue in ues:
            mpath = f'model_ue_{ue}.pth'
            spath = f'scaler_ue_{ue}.pkl'
            if not os.path.exists(mpath):
                print(f"[경고] {mpath} 없음 — train_rnn.py 먼저 실행")
                continue
            model = ChannelRNN(len(self.features), 64, 2, len(self.features))
            model.load_state_dict(torch.load(mpath, map_location='cpu'))
            model.eval()
            self.models[ue] = model
            self.scalers[ue] = joblib.load(spath)
            print(f"[{ue}] 모델 로드 완료")

    def predict_next(self, ue, kpi: dict):
        if ue not in self.models:
            return None

        row = [kpi.get(f, 0.0) for f in self.features]
        self.history[ue].append(row)
        if len(self.history[ue]) > self.seq_length * 2:
            self.history[ue] = self.history[ue][-self.seq_length:]
        if len(self.history[ue]) < self.seq_length:
            return None

        seq = np.array(self.history[ue][-self.seq_length:])
        seq_scaled = self.scalers[ue].transform(seq)
        x = torch.FloatTensor(seq_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = self.models[ue](x).numpy()
        pred = self.scalers[ue].inverse_transform(pred_scaled)[0]
        return dict(zip(self.features, pred))

    def compute_weights(self, predictions: dict) -> dict:
        valid = {ue: p for ue, p in predictions.items() if p}
        if not valid:
            return {ue: round(1.0 / len(self.ues), 4) for ue in self.ues}

        scores = {}
        for ue, pred in valid.items():
            snr  = max(pred.get('snr',  1.0), 0.1)
            bler = pred.get('bler', 0.0)
            scores[ue] = (1.0 / snr) * (1.0 + bler * 5)  # Fairness 우선

        total = sum(scores.values()) or 1.0
        return {ue: round(scores.get(ue, 0.0) / total, 4) for ue in self.ues}

    def apply_weights(self, weights: dict):
        with open('/tmp/dapp_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        print(f"  → /tmp/dapp_weights.json 저장 완료")

    def run(self, interval=1.0):
        print("=" * 50)
        print(f"dApp 시작 | UE: {self.ues} | Features: {self.features}")
        print("=" * 50)
        cycle = 0
        try:
            while True:
                cycle += 1
                predictions = {}
                for ue in self.ues:
                    kpi = get_latest_kpi(self.csv_path, ue, self.features)
                    pred = self.predict_next(ue, kpi)
                    predictions[ue] = pred
                    if pred:
                        print(f"  [{ue}] SNR예측={pred.get('snr',0):.1f} "
                              f"BLER예측={pred.get('bler',0):.4f}")

                weights = self.compute_weights(predictions)
                self.apply_weights(weights)
                print(f"[Cycle {cycle}] 가중치: {weights}\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\ndApp 종료")


if __name__ == '__main__':
    df = pd.read_csv('kpi_baseline.csv')
    ues = [str(u) for u in df['rnti'].unique()]
    ctrl = DAppController(ues=ues, csv_path='kpi_live.csv')
    ctrl.run(interval=1.0)
