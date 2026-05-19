import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 device:", device)

# ── 설정 ──────────────────────────────────────
FEATURES = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
SEQ_LENGTH = 10
HIDDEN_SIZE = 64
EPOCHS = 300

# ── 모델 정의 ──────────────────────────────────
class ChannelRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# ── 데이터 로드 ────────────────────────────────
df = pd.read_csv('kpi_combined.csv')
available = [f for f in FEATURES if f in df.columns]
print(f"사용 feature: {available}")

ues = df['rnti'].unique()
print(f"학습 UE 목록: {ues}")

# ── UE별 학습 ──────────────────────────────────
for ue in ues:
    ue_df = df[df['rnti'] == ue][available].dropna()
    if len(ue_df) < SEQ_LENGTH + 10:
        print(f"UE {ue}: 데이터 부족 ({len(ue_df)}개), 스킵")
        continue

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(ue_df.values)

    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).to(device)

    model = ChannelRNN(
    len(available),
    HIDDEN_SIZE,
    2,
    len(available)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"\n[UE {ue}] 학습 시작 ({len(ue_df)}개 샘플)")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), f'model_ue_{ue}.pth')
    joblib.dump(scaler, f'scaler_ue_{ue}.pkl')
    joblib.dump(available, 'features.pkl')  # feature 목록 저장
    print(f"[UE {ue}] 저장 완료 → model_ue_{ue}.pth / scaler_ue_{ue}.pkl")

print("Model Ready for Training!")
