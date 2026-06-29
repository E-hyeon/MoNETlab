"""
eval_dapp.py — dApp 채널 예측 성능 평가 (논문용)

비교:
  - Chronos TCN (dApp)  : 과거 10스텝 → 다음 스텝 예측
  - Persistence (OAI 기준) : 현재값 = 다음 스텝 (예측 없음)

출력:
  - Feature별 MAE / RMSE
  - 개선율 (%)
  - Inference latency (ms / 예측 1회)
"""

import os, time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import TSDataset

# ── 설정 ──────────────────────────────────────────────────────────────
FEATURES        = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK        = 10
FORECASTER_PATH = 'chronos_forecaster'
SCALER_PATH     = 'scaler_chronos.pkl'
KPI_CSV         = 'kpi_live.csv'
TARGET_UES      = ['21ab', 'f402']   # 동시 활성 세션만

# ── 데이터 로드 + 전처리 (chronos_retrain.py와 동일) ─────────────────
print("=" * 60)
print("데이터 로드 및 전처리 중...")

df_raw = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_raw[f] = pd.to_numeric(df_raw[f], errors='coerce').fillna(0)
df_raw['rnti'] = df_raw['rnti'].astype(str)
df_raw = df_raw[df_raw['rnti'].isin(TARGET_UES)].copy()

# 학습 때와 동일한 1s resample + interpolate
frames = []
for ue in TARGET_UES:
    sub = df_raw[df_raw['rnti'] == ue].set_index('timestamp')[FEATURES].copy()
    sub = sub.resample('1s').mean().interpolate('linear').bfill().fillna(0)
    sub = sub.reset_index()
    sub['rnti'] = ue
    frames.append(sub)
    print(f"  UE {ue}: {len(sub)}행 (1s resample)")

df = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ues = TARGET_UES
print(f"전처리 완료: 총 {len(df)}행\n")

# ── 모델 로드 ─────────────────────────────────────────────────────────
print("모델 로드 중...")
scaler = joblib.load(SCALER_PATH)

# TCNForecaster 복원용 더미 tsdata — 전처리된 데이터, id_col='rnti' (학습과 동일 구조)
tsdata_tmp, _, _ = TSDataset.from_pandas(
    df, dt_col='timestamp', target_col=FEATURES, id_col='rnti',
    with_split=True, val_ratio=0.15, test_ratio=0.15,
)
sc_tmp = StandardScaler()
tsdata_tmp.scale(sc_tmp, fit=True)
tsdata_tmp.roll(lookback=LOOKBACK, horizon=1)

forecaster = TCNForecaster.from_tsdataset(tsdata_tmp)
forecaster.load(FORECASTER_PATH)
print("Chronos TCN 로드 완료")

# ── 슬라이딩 윈도 평가 ────────────────────────────────────────────────
all_y_true, all_y_tcn, all_y_pers = [], [], []
latencies = []

for ue in ues:
    sub = df[df['rnti'] == str(ue)][FEATURES].values.astype(np.float32)
    if len(sub) < LOOKBACK + 1:
        print(f"  UE {ue}: 데이터 부족 ({len(sub)}행), 스킵")
        continue

    for i in range(LOOKBACK, len(sub)):
        window = sub[i - LOOKBACK:i]          # (10, 6)
        y_true = sub[i]                        # (6,)

        # Persistence (OAI 기준: 마지막 관측값 그대로)
        y_pers = sub[i - 1]

        # Chronos TCN 예측
        window_scaled = scaler.transform(window)
        x = window_scaled[np.newaxis, :, :].astype(np.float32)

        t0 = time.perf_counter()
        pred_scaled = forecaster.predict(x).reshape(1, len(FEATURES))
        latencies.append((time.perf_counter() - t0) * 1000)

        y_tcn = scaler.inverse_transform(pred_scaled)[0]

        all_y_true.append(y_true)
        all_y_tcn.append(y_tcn)
        all_y_pers.append(y_pers)

Y      = np.array(all_y_true)
Y_tcn  = np.array(all_y_tcn)
Y_pers = np.array(all_y_pers)

# ── 결과 출력 ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Feature':<12} {'MAE(TCN)':>10} {'MAE(Pers)':>10} {'개선율':>8}  "
      f"{'RMSE(TCN)':>10} {'RMSE(Pers)':>10}")
print("-" * 60)

mae_tcn_all, mae_pers_all = [], []
for i, feat in enumerate(FEATURES):
    mae_t = mean_absolute_error(Y[:, i], Y_tcn[:, i])
    mae_p = mean_absolute_error(Y[:, i], Y_pers[:, i])
    rmse_t = np.sqrt(mean_squared_error(Y[:, i], Y_tcn[:, i]))
    rmse_p = np.sqrt(mean_squared_error(Y[:, i], Y_pers[:, i]))
    improvement = (1 - mae_t / mae_p) * 100 if mae_p > 0 else 0.0
    mae_tcn_all.append(mae_t)
    mae_pers_all.append(mae_p)
    print(f"{feat:<12} {mae_t:>10.4f} {mae_p:>10.4f} {improvement:>7.1f}%  "
          f"{rmse_t:>10.4f} {rmse_p:>10.4f}")

print("-" * 60)
avg_imp = (1 - np.mean(mae_tcn_all) / np.mean(mae_pers_all)) * 100
print(f"{'평균':<12} {np.mean(mae_tcn_all):>10.4f} {np.mean(mae_pers_all):>10.4f} "
      f"{avg_imp:>7.1f}%")

print("\n" + "=" * 60)
lat = np.array(latencies)
print(f"Inference Latency (Chronos TCN, n={len(lat)}회)")
print(f"  평균: {lat.mean():.2f} ms")
print(f"  중앙값: {np.median(lat):.2f} ms")
print(f"  P95: {np.percentile(lat, 95):.2f} ms")
print(f"  P99: {np.percentile(lat, 99):.2f} ms")
print("=" * 60)
