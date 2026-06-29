"""
dapp_controller_global.py
────────────────────────────────────────────────────────────────────
글로벌 단일 모델 기반 dApp 컨트롤러.
  - model_global.pth       : 전체 UE 학습 글로벌 모델 (Attention 구조)
  - scaler_global.pkl      : 글로벌 MinMaxScaler
  - features_global.pkl    : feature 이름 리스트
  - ue_encoder.pkl         : LabelEncoder (미지 UE → padding_idx 처리)

신규 UE(학습 때 없던 UE)는 UE 임베딩 padding_idx(=zero vector)로
처리하므로, 재학습 없이 즉시 예측 가능.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import time
import json
import os
import pandas as pd


# ── 모델 클래스 [수정: train_rnn_global.py와 완전히 동일한 Attention 구조 복원] ──
class GlobalChannelRNN(nn.Module):
    def __init__(self, n_features, n_ues,
                 hidden_size=128, num_layers=2, ue_emb_dim=8):
        super().__init__()
        self.ue_emb = nn.Embedding(
            num_embeddings=n_ues + 1,
            embedding_dim=ue_emb_dim,
            padding_idx=n_ues
        )
        lstm_input = n_features + ue_emb_dim
        self.lstm = nn.LSTM(
            lstm_input, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, n_features)
        self.attention = nn.Linear(hidden_size, 1)  # Attention 레이어 추가

    def forward(self, x, ue_idx):
        # UE 임베딩을 시퀀스 전체에 브로드캐스트
        emb = self.ue_emb(ue_idx)                         # (B, emb_dim)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, emb_dim)
        x_in = torch.cat([x, emb_exp], dim=-1)            # (B, T, feat+emb)
        out, _ = self.lstm(x_in)

        # 각 스텝마다 중요도 점수 계산
        attn_score = self.attention(out)                  # (B, T, 1)
        attn_weight = torch.softmax(attn_score, dim=1)    # 합=1로 정규화

        # 중요도 가중 평균 (Context Vector 추출)
        context = (out * attn_weight).sum(dim=1)          # (B, hidden)
        return self.fc(context)                           # (B, n_features)


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


# ── 글로벌 dApp 컨트롤러 ─────────────────────────────────────────
class GlobalDAppController:
    """
    단일 글로벌 모델로 모든 UE를 처리.
    학습 때 없던 신규 UE도 padding_idx 임베딩으로 즉시 대응.
    """

    MODEL_PATH    = 'model_global.pth'
    SCALER_PATH   = 'scaler_global.pkl'
    FEATURES_PATH = 'features_global.pkl'
    ENCODER_PATH  = 'ue_encoder.pkl'

    def __init__(self, ues: list, csv_path='kpi_live.csv', seq_length=10):
        self.ues        = ues
        self.csv_path   = csv_path
        self.seq_length = seq_length
        self.history    = {ue: [] for ue in ues}

        # ── 아티팩트 로드 ───────────────────────────────────────
        self.features = joblib.load(self.FEATURES_PATH)
        self.scaler   = joblib.load(self.SCALER_PATH)
        self.encoder  = joblib.load(self.ENCODER_PATH)   # LabelEncoder
        n_ues         = len(self.encoder.classes_)

        self.model = GlobalChannelRNN(
            n_features  = len(self.features),
            n_ues       = n_ues,
            hidden_size = 128,
            num_layers  = 2,
            ue_emb_dim  = 8,
        )
        self.model.load_state_dict(
            torch.load(self.MODEL_PATH, map_location='cpu')
        )
        self.model.eval()
        print(f"글로벌 모델 로드 완료  (UE 수: {n_ues}, "
              f"features: {self.features})")

        # 패딩 인덱스: 미지 UE에 사용
        self._unk_idx = n_ues

    # ── UE 인덱스 변환 ────────────────────────────────────────────
    def _ue_to_idx(self, ue: str) -> int:
        """학습 때 없던 UE면 padding_idx(unknown) 반환."""
        try:
            return int(self.encoder.transform([str(ue)])[0])
        except ValueError:
            print(f"  [미지 UE] {ue} → unknown 임베딩 사용")
            return self._unk_idx

    # ── 단일 UE 예측 ──────────────────────────────────────────────
    def predict_next(self, ue: str, kpi: dict):
        row = [kpi.get(f, 0.0) for f in self.features]
        self.history[ue].append(row)

        # 히스토리 길이 관리
        if len(self.history[ue]) > self.seq_length * 2:
            self.history[ue] = self.history[ue][-self.seq_length:]
        if len(self.history[ue]) < self.seq_length:
            return None

        seq = np.array(self.history[ue][-self.seq_length:])
        seq_scaled = self.scaler.transform(seq)

        x      = torch.FloatTensor(seq_scaled).unsqueeze(0)   # (1, T, F)
        ue_idx = torch.LongTensor([self._ue_to_idx(ue)])      # (1,)

        with torch.no_grad():
            pred_scaled = self.model(x, ue_idx).numpy()

        pred = self.scaler.inverse_transform(pred_scaled)[0]
        return dict(zip(self.features, pred))

    # ── 가중치 계산 (Fairness 우선) ───────────────────────────────
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

    # ── 가중치 적용 ───────────────────────────────────────────────
    def apply_weights(self, weights: dict):
        with open('/tmp/dapp_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        print("  → /tmp/dapp_weights.json 저장 완료")

    # ── 메인 루프 ─────────────────────────────────────────────────
    def run(self, interval=1.0):
        print("=" * 55)
        print(f"글로벌 dApp 시작 | UE: {self.ues}")
        print("=" * 55)
        cycle = 0
        try:
            while True:
                cycle += 1
                predictions = {}
                for ue in self.ues:
                    kpi  = get_latest_kpi(self.csv_path, ue, self.features)
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
    # 시스템 상황에 맞는 기저 데이터 파일 확인 후 기동
    if os.path.exists('kpi_baseline.csv'):
        df = pd.read_csv('kpi_baseline.csv')
        ues = [str(u) for u in df['rnti'].unique()]
    else:
        # 혹시 몰라 하드코딩 백업이나 빈 리스트 방지 로직 구현
        ues = []
        print("[경고] kpi_baseline.csv 파일이 없어 타겟 UE 목록을 자동으로 인지하지 못했습니다.")

    if ues:
        ctrl = GlobalDAppController(ues=ues, csv_path='kpi_live.csv')
        ctrl.run(interval=1.0)