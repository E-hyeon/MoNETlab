"""
chronos_train.py
────────────────────────────────────────────────────────────────────
BigDL Chronos TCNForecaster 기반 UE KPI 예측 모델 학습
  - TSDataset: 전처리 + 시간순 분할 자동
  - TCNForecaster: 학습 + Early Stopping 내장
  - evaluate(): MAE / MSE 자동 계산
  - optimize(): OpenVINO 추론 가속 (추론 시간 비교)

저장 파일
  chronos_forecaster/   ← 모델 디렉토리
  scaler_chronos.pkl    ← 스케일러 (dApp 추론 시 역정규화용)
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.data import TSDataset
from bigdl.chronos.forecaster import TCNForecaster

# ── 설정 ──────────────────────────────────────────────────────────
FEATURES   = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK   = 10      # 입력 시퀀스 길이 (과거 10스텝)
HORIZON    = 1       # 예측 스텝 수 (다음 1스텝)
EPOCHS     = 300
BATCH_SIZE = 256
VAL_RATIO  = 0.15
TEST_RATIO = 0.15

# ── 데이터 로드 ───────────────────────────────────────────────────
print("=" * 55)
print("데이터 로드 중...")
df = pd.read_csv('kpi_baseline.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 결측치 처리
df[FEATURES] = df[FEATURES].fillna(0)

print(f"전체 데이터: {len(df)}행  UE 수: {df['rnti'].nunique()}")
print(f"컬럼: {list(df.columns)}")
print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# ── TSDataset 생성 ────────────────────────────────────────────────
# with_split=True → 시간순으로 train/val/test 자동 분할
# 랜덤 분할 안 함 → 시계열 Data Leakage 방지
print("\n데이터셋 분할 중... (시간순 70/15/15)")
tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(
    df,
    dt_col="timestamp",
    target_col=FEATURES,
    with_split=True,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO
)

# ── 전처리 ────────────────────────────────────────────────────────
# train 기준 스케일러 → val, test에 동일 적용
scaler = StandardScaler()
tsdata_train.scale(scaler, fit=True)
tsdata_val.scale(scaler,   fit=False)
tsdata_test.scale(scaler,  fit=False)

# 슬라이딩 윈도우 (lookback=10스텝 → horizon=1스텝 예측)
for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.roll(lookback=LOOKBACK, horizon=HORIZON)

print(f"Train: {len(tsdata_train.to_pandas())}행")
print(f"Val:   {len(tsdata_val.to_pandas())}행")
print(f"Test:  {len(tsdata_test.to_pandas())}행")

# ── 모델 생성 ─────────────────────────────────────────────────────
print("\n모델 생성 중...")
forecaster = TCNForecaster.from_tsdataset(
    tsdata_train,
    past_seq_len=LOOKBACK,
    future_seq_len=HORIZON,
)

# ── 학습 ──────────────────────────────────────────────────────────
print(f"\n학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
print("Early Stopping 내장 → 최적 시점 자동 저장")
print("-" * 55)

forecaster.fit(
    tsdata_train,
    validation_data=tsdata_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# ── 성능 평가 ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("테스트 성능 평가 중...")
result = forecaster.evaluate(tsdata_test, metrics=['mse', 'mae'])
print(f"MSE: {result[0]:.6f}")
print(f"MAE: {result[1]:.6f}")

# Feature별 MAE (역정규화 후 실제 단위)
test_pred  = forecaster.predict(tsdata_test)
x_test, y_test = tsdata_test.roll(lookback=LOOKBACK,
                                   horizon=HORIZON,
                                   is_predict=False)

pred_inv  = scaler.inverse_transform(
    test_pred.reshape(-1, len(FEATURES)))
true_inv  = scaler.inverse_transform(
    y_test.reshape(-1, len(FEATURES)))
mae_each  = np.abs(pred_inv - true_inv).mean(axis=0)

print("\nFeature별 MAE (실제 단위):")
for f, e in zip(FEATURES, mae_each):
    print(f"  {f:12s}: {e:.4f}")

# ── 추론 속도 비교 (일반 vs OpenVINO) ────────────────────────────
print("\n" + "=" * 55)
print("추론 속도 측정 중...")

# 일반 PyTorch 추론
start = time.time()
for _ in range(100):
    _ = forecaster.predict(tsdata_test)
t_pytorch = (time.time() - start) / 100 * 1000
print(f"PyTorch  추론: {t_pytorch:.2f} ms")

# OpenVINO 가속 추론
try:
    forecaster.optimize(
        target_data=tsdata_test,
        backend="openvino",
        precision="fp32"
    )
    start = time.time()
    for _ in range(100):
        _ = forecaster.predict(tsdata_test)
    t_openvino = (time.time() - start) / 100 * 1000
    print(f"OpenVINO 추론: {t_openvino:.2f} ms")
    print(f"속도 향상:     {t_pytorch/t_openvino:.1f}배  "
          f"({(1-t_openvino/t_pytorch)*100:.0f}% 단축)")
except Exception as e:
    print(f"OpenVINO 가속 실패: {e}")

# ── 저장 ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
forecaster.save('chronos_forecaster')
joblib.dump(scaler, 'scaler_chronos.pkl')
print("저장 완료:")
print("  chronos_forecaster/  ← TCN 모델")
print("  scaler_chronos.pkl   ← 스케일러")
print("\n학습 완료!")
