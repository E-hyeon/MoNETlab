"""
mlp_train.py — 얕은 MLP(은닉층 1개, 뉴런 12개) 기반 다음 스텝 예측기 학습.

dapp_controller_chronos.py의 TCNForecaster 자리를 대체할 경량 모델을 만든다.
LOOKBACK=10 윈도우(6피처 × 10스텝 = 60차원, 스케일링 후 flatten)를 입력으로,
다음 스텝의 6피처 값을 한 번에 예측하는 다중출력 회귀 모델.

학습 데이터: kpi_live.csv (현재 세션의 실측 KPI, UE 구분 없이 윈도우 단위로 합산 학습
             — chronos_forecaster와 동일하게 UE 무관 단일 공용 모델).
저장: mlp_forecaster.pkl (MLPRegressor), scaler_mlp.pkl (StandardScaler)
"""

import time
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

FEATURES       = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK       = 10
KPI_CSV        = 'kpi_live.csv'
HIDDEN_NEURONS = 12   # 은닉층 1개, 8~16 범위 내 얕은 MLP

print("=" * 55)
df = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
df['rnti'] = df['rnti'].astype(str)
print(f"데이터: {len(df)}행  UE: {df['rnti'].nunique()}개 ({KPI_CSV})")

# ── 스케일러: chronos_forecaster와 동일하게 전체 데이터에 공용으로 fit ──
scaler = StandardScaler()
scaler.fit(df[FEATURES].values)

# ── UE별로 슬라이딩 윈도 생성 (LOOKBACK → 다음 스텝) ────────────────
X, y = [], []
for ue in df['rnti'].unique():
    sub = df[df['rnti'] == ue].sort_values('timestamp').reset_index(drop=True)
    vals = scaler.transform(sub[FEATURES].values)
    for i in range(LOOKBACK, len(vals)):
        X.append(vals[i - LOOKBACK:i].flatten())   # (60,)
        y.append(vals[i])                           # (6,)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"학습 샘플: {len(X)}개 (입력 차원 {X.shape[1]})")

# ── 시간순 train/test 분할 (마지막 15%를 테스트로) ──────────────────
n_test = max(1, int(len(X) * 0.15))
X_train, X_test = X[:-n_test], X[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]
print(f"train={len(X_train)}  test={len(X_test)}\n")

# ── 얕은 MLP 학습 ────────────────────────────────────────────────────
model = MLPRegressor(
    hidden_layer_sizes=(HIDDEN_NEURONS,),
    activation='relu',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,
)
print(f"학습 시작 (hidden_layer_sizes=({HIDDEN_NEURONS},))")
model.fit(X_train, y_train)
print(f"학습 완료 (iterations={model.n_iter_})")

# ── 평가 ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n" + "=" * 55)
print("Feature별 MAE (스케일 공간, 표준편차=1 기준)")
maes = []
for i, f in enumerate(FEATURES):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    maes.append(mae)
    print(f"  {f:10s}: {mae:.4f}")
print(f"  {'평균':10s}: {np.mean(maes):.4f}")

# ── 저장 ──────────────────────────────────────────────────────────────
joblib.dump(model, 'mlp_forecaster.pkl')
joblib.dump(scaler, 'scaler_mlp.pkl')
print("\n모델 저장 완료: mlp_forecaster.pkl, scaler_mlp.pkl")

# ── 추론 속도 (단일 윈도우 1회 예측) ─────────────────────────────────
print("\n" + "=" * 55)
x1 = X_test[:1]
model.predict(x1)  # warm-up
start = time.time()
for _ in range(200):
    model.predict(x1)
t_ms = (time.time() - start) / 200 * 1000
print(f"MLP 단일 추론 지연: {t_ms:.4f} ms  (n=200회 평균)")
