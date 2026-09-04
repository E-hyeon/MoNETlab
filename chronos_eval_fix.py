import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.data import TSDataset
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator

FEATURES   = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK   = 10
HORIZON    = 1
EPOCHS     = 300
BATCH_SIZE = 256
VAL_RATIO  = 0.15
TEST_RATIO = 0.15

print("=" * 55)
print("데이터 로드 중...")
df = pd.read_csv('kpi_baseline.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df[FEATURES] = df[FEATURES].fillna(0)
print(f"전체 데이터: {len(df)}행  UE 수: {df['rnti'].nunique()}")

print("\n데이터셋 분할 중... (시간순 70/15/15)")
tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(
    df,
    dt_col="timestamp",
    target_col=FEATURES,
    with_split=True,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO
)

scaler = StandardScaler()
tsdata_train.scale(scaler, fit=True)
tsdata_val.scale(scaler,   fit=False)
tsdata_test.scale(scaler,  fit=False)

for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.roll(lookback=LOOKBACK, horizon=HORIZON)

print("모델 생성 중...")
forecaster = TCNForecaster.from_tsdataset(
    tsdata_train,
    past_seq_len=LOOKBACK,
    future_seq_len=HORIZON,
)

print(f"\n학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
forecaster.fit(
    tsdata_train,
    validation_data=tsdata_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# ── 성능 평가 (수정된 API) ────────────────────────────────────────
print("\n" + "=" * 55)
print("테스트 성능 평가 중...")

y_hat          = forecaster.predict(tsdata_test)
y_hat_unscaled = tsdata_test.unscale_numpy(y_hat)
y_unscaled     = tsdata_test.unscale_numpy(tsdata_test.to_numpy()[1])

mae = Evaluator.evaluate(metric="mae", y=y_unscaled,
                          yhat=y_hat_unscaled, multioutput="raw_values")
mse = Evaluator.evaluate(metric="mse", y=y_unscaled,
                          yhat=y_hat_unscaled, multioutput="raw_values")

print(f"MSE: {mse.mean():.6f}")
print(f"MAE: {mae.mean():.6f}")
print("\nFeature별 MAE (실제 단위):")
for f, e in zip(FEATURES, mae):
    print(f"  {f:12s}: {e:.4f}")

# ── 추론 속도 비교 ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("추론 속도 측정 중...")

start = time.time()
for _ in range(100):
    forecaster.predict(tsdata_test)
t_pytorch = (time.time() - start) / 100 * 1000
print(f"PyTorch  추론: {t_pytorch:.2f} ms")

try:
    forecaster.optimize(
        target_data=tsdata_test,
        backend="openvino",
        precision="fp32"
    )
    start = time.time()
    for _ in range(100):
        forecaster.predict(tsdata_test)
    t_openvino = (time.time() - start) / 100 * 1000
    print(f"OpenVINO 추론: {t_openvino:.2f} ms")
    print(f"속도 향상:     {t_pytorch/t_openvino:.1f}배  "
          f"({(1-t_openvino/t_pytorch)*100:.0f}% 단축)")
except Exception as e:
    print(f"OpenVINO 실패: {e}")

# ── 저장 ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
forecaster.save('chronos_forecaster')
joblib.dump(scaler, 'scaler_chronos.pkl')
print("저장 완료: chronos_forecaster/  scaler_chronos.pkl")
