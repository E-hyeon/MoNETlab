"""
train_rnn_global.py
────────────────────────────────────────────────────────────────────
전체 UE 데이터를 하나의 모델로 학습.
UE ID는 Label Encoding → Embedding 레이어로 처리하여
어떤 UE(신규 포함)가 들어와도 대응 가능.

저장 파일
  model_global.pth     ← 글로벌 모델 가중치
  scaler_global.pkl    ← 전체 데이터 기준 MinMaxScaler
  features_global.pkl  ← feature 이름 리스트
  ue_encoder.pkl       ← UE rnti → int index LabelEncoder
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# ── 설정 ──────────────────────────────────────────────────────────
FEATURES    = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
SEQ_LENGTH  = 10
HIDDEN_SIZE = 128      # 글로벌 모델이므로 용량 약간 확대
UE_EMB_DIM  = 8        # UE 임베딩 차원
NUM_LAYERS  = 2
EPOCHS      = 300
LR          = 0.001
BATCH_SIZE  = 256
PATIENCE    = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 device:", device)


# ── 모델 정의 ─────────────────────────────────────────────────────
class GlobalChannelRNN(nn.Module):
    """
    KPI 시퀀스 + UE 임베딩을 함께 받아 다음 KPI를 예측.

    입력:
      x      : (batch, seq_len, n_features)   ← 정규화된 KPI
      ue_idx : (batch,)                        ← UE 정수 인덱스
    출력:
      (batch, n_features)                      ← 다음 스텝 KPI 예측
    """
    def __init__(self, n_features, n_ues,
                 hidden_size=128, num_layers=2, ue_emb_dim=8):
        super().__init__()
        self.ue_emb = nn.Embedding(
            num_embeddings=n_ues + 1,   # +1: 미지 UE용 패딩 인덱스
            embedding_dim=ue_emb_dim,
            padding_idx=n_ues           # 미지 UE → zero vector
        )
        lstm_input = n_features + ue_emb_dim
        self.lstm = nn.LSTM(
            lstm_input, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, n_features)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x, ue_idx):
        # UE 임베딩을 시퀀스 전체에 브로드캐스트
        emb = self.ue_emb(ue_idx)                         # (B, emb_dim)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, emb_dim)
        x_in = torch.cat([x, emb_exp], dim=-1)            # (B, T, feat+emb)
        out, _ = self.lstm(x_in)

        # 각 스텝마다 중요도 점수 계산
        attn_score = self.attention(out)      # (B, T, 1)
        attn_weight = torch.softmax(attn_score, dim=1)  # 합=1로 정규화

        # 중요도 가중 평균
        context = (out * attn_weight).sum(dim=1)  # (B, hidden)
        return self.fc(context)                  # (B, n_features)


# ── 시퀀스 생성 ───────────────────────────────────────────────────
def create_sequences(data: np.ndarray, ue_indices: np.ndarray, seq_len: int):
    X, Y, UE = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len])
        UE.append(ue_indices[i + seq_len])   # 예측 대상 스텝의 UE
    return np.array(X), np.array(Y), np.array(UE)


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────
df = pd.read_csv('kpi_baseline.csv')
available = [f for f in FEATURES if f in df.columns]
print(f"사용 feature: {available}")

# UE 인코딩
le = LabelEncoder()
df['ue_idx'] = le.fit_transform(df['rnti'].astype(str))
n_ues = len(le.classes_)
print(f"학습 UE 수: {n_ues}  →  {list(le.classes_)}")

# 글로벌 스케일러 (전체 데이터 기준)
scaler = MinMaxScaler()
df[available] = scaler.fit_transform(df[available].fillna(0))

# UE별로 시퀀스 생성 후 합치기
all_X, all_Y, all_UE = [], [], []
for ue_val, grp in df.groupby('ue_idx'):
    grp = grp.sort_index()          # 시간순 보장
    kpi_arr = grp[available].values
    ue_arr  = grp['ue_idx'].values
    if len(kpi_arr) < SEQ_LENGTH + 1:
        print(f"  UE idx {ue_val}: 데이터 부족 ({len(kpi_arr)}개), 스킵")
        continue
    X, Y, U = create_sequences(kpi_arr, ue_arr, SEQ_LENGTH)
    all_X.append(X); all_Y.append(Y); all_UE.append(U)

X_all  = np.concatenate(all_X,  axis=0)
Y_all  = np.concatenate(all_Y,  axis=0)
UE_all = np.concatenate(all_UE, axis=0)
print(f"총 시퀀스 수: {len(X_all)}")

# ── 시간순 분할 (Time Split) ───────────────────────────────
def time_split(X, Y, UE, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    return (X[:train_end],   Y[:train_end],   UE[:train_end],    # 훈련
            X[train_end:val_end], Y[train_end:val_end], UE[train_end:val_end],  # 검증
            X[val_end:],     Y[val_end:],     UE[val_end:])      # 테스트

X_tr, Y_tr, UE_tr, X_val, Y_val, UE_val, X_te, Y_te, UE_te = time_split(X_all, Y_all, UE_all)


# ── 각각 별도로 텐서 변환 및 .to(device) 적용 ────────────────
X_tr_t,  Y_tr_t,  UE_tr_t  = torch.FloatTensor(X_tr).to(device),  torch.FloatTensor(Y_tr).to(device),  torch.LongTensor(UE_tr).to(device)
X_val_t, Y_val_t, UE_val_t = torch.FloatTensor(X_val).to(device), torch.FloatTensor(Y_val).to(device), torch.LongTensor(UE_val).to(device)
X_te_t,  Y_te_t,  UE_te_t  = torch.FloatTensor(X_te).to(device),  torch.FloatTensor(Y_te).to(device),  torch.LongTensor(UE_te).to(device)

# ── 모델 / 옵티마이저 ─────────────────────────────────────────────
model = GlobalChannelRNN(
    n_features  = len(available),
    n_ues       = n_ues,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    ue_emb_dim  = UE_EMB_DIM,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 훈련용 덤프셋은 훈련 데이터(X_tr_t, ...)만 사용하고 shuffle=True 유지
train_dataset = torch.utils.data.TensorDataset(X_tr_t, Y_tr_t, UE_tr_t)
train_loader  = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)


# ── Early Stopping 포함 학습 루프 ───────────────────────────
print("\n글로벌 모델 학습 시작 (with Early Stopping)")
best_val_loss = float('inf')
wait          = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb, ub in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb, ub), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    avg_train_loss = total_loss / len(X_tr)
    # 검증 (가중치 업데이트 없음)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t, UE_val_t)
        val_loss = criterion(val_pred, Y_val_t).item()

    # 주기적인 로그 출력 (예: 10 에폭마다 또는 성능 갱신 시)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early Stopping 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_global.pth')  # 파일명을 요구사항에 맞춰 model_global.pth로 저장
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early Stopping @ Epoch {epoch+1}")
            break

# ── [추가] 테스트 (딱 한 번) ──────────────────────────────────────
print("\n최적 모델 복원 및 테스트 평가 중...")
model.load_state_dict(torch.load('model_global.pth'))  # 저장했던 최적 시점 가중치 복원
model.eval()

with torch.no_grad():
    test_pred  = model(X_te_t, UE_te_t)
    test_loss  = criterion(test_pred, Y_te_t).item()

    # 역정규화 후 실제 단위로 성능 측정 (CPU로 이동하여 numpy 변환)
    pred_real  = scaler.inverse_transform(test_pred.cpu().numpy())
    true_real  = scaler.inverse_transform(Y_te_t.cpu().numpy())
    mae        = np.abs(pred_real - true_real).mean(axis=0)

print(f"최종 테스트 Loss (MSE): {test_loss:.6f}")
print("Feature별 MAE:")
for f, e in zip(available, mae):
    print(f"  {f:12s}: {e:.4f}")

# ── 저장 (피처 및 인코더) ─────────────────────────────────────────
# 모델 가중치는 Early Stopping 루프 내에서 이미 'model_global.pth'로 저장되었습니다.
joblib.dump(scaler,    'scaler_global.pkl')
joblib.dump(available, 'features_global.pkl')
joblib.dump(le,        'ue_encoder.pkl')

print("\n저장 완료:")
print("  model_global.pth  /  scaler_global.pkl")
print("  features_global.pkl  /  ue_encoder.pkl")
print("Model Ready!")