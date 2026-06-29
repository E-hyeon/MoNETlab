"""
chronos_retrain.py — kpi_live.csv로 Chronos TCN 재학습
50 epochs마다 체크포인트 저장 → 중단 후 이어서 학습 가능
"""

import os
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
CHUNK      = 50   # 몇 epoch마다 체크포인트 저장

CKPT_MODEL = 'chronos_checkpoint'
CKPT_EPOCH = 'chronos_ckpt_epoch.txt'

print("=" * 60)
print("kpi_live.csv 로드 중...")
df_all = pd.read_csv('kpi_live.csv', parse_dates=['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_all[f] = pd.to_numeric(df_all[f], errors='coerce').fillna(0)
df_all['rnti'] = df_all['rnti'].astype(str)

print(f"전체: {len(df_all)}행  UE: {list(df_all['rnti'].unique())}")

# ── UE별 분리 후 concat ──────────────────────────────────────────────
frames = []
for ue, grp in df_all.groupby('rnti'):
    g = grp.set_index('timestamp')[FEATURES].copy()
    g = g.resample('1s').mean().interpolate('linear').fillna(method='bfill').fillna(0)
    g = g.reset_index()
    g['rnti'] = ue
    frames.append(g)
    print(f"  UE {ue}: {len(g)}행 (1s resample)")

df = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
print(f"\n학습용 데이터: {len(df)}행\n")

# ── TSDataset 구성 ────────────────────────────────────────────────────
tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(
    df,
    dt_col='timestamp',
    target_col=FEATURES,
    id_col='rnti',
    with_split=True,
    val_ratio=0.15,
    test_ratio=0.15,
)

scaler = StandardScaler()
tsdata_train.scale(scaler, fit=True)
tsdata_val.scale(scaler,   fit=False)
tsdata_test.scale(scaler,  fit=False)
for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
    tsdata.roll(lookback=LOOKBACK, horizon=HORIZON)

# ── 모델 생성 ────────────────────────────────────────────────────────
print("=" * 60)
forecaster = TCNForecaster.from_tsdataset(
    tsdata_train, past_seq_len=LOOKBACK, future_seq_len=HORIZON
)

# ── 체크포인트에서 재시작 ─────────────────────────────────────────────
start_epoch = 0
if os.path.exists(CKPT_MODEL) and os.path.exists(CKPT_EPOCH):
    try:
        start_epoch = int(open(CKPT_EPOCH).read().strip())
        forecaster.load(CKPT_MODEL)
        print(f"체크포인트 로드 완료 → Epoch {start_epoch}부터 재시작")
    except Exception as e:
        print(f"체크포인트 로드 실패 ({e}) → 처음부터 시작")
        start_epoch = 0
else:
    print(f"체크포인트 없음 → 처음부터 시작 (총 {EPOCHS} epochs)")

# ── 청크 단위 학습 + 체크포인트 저장 ─────────────────────────────────
print("=" * 60)
epoch = start_epoch
while epoch < EPOCHS:
    chunk_epochs = min(CHUNK, EPOCHS - epoch)
    print(f"\n[Epoch {epoch+1}~{epoch+chunk_epochs} / {EPOCHS}] 학습 중...")
    forecaster.fit(tsdata_train, validation_data=tsdata_val,
                   epochs=chunk_epochs, batch_size=256)
    epoch += chunk_epochs
    forecaster.save(CKPT_MODEL)
    joblib.dump(scaler, 'scaler_chronos.pkl')
    with open(CKPT_EPOCH, 'w') as f:
        f.write(str(epoch))
    print(f"[체크포인트 저장] Epoch {epoch}/{EPOCHS} 완료")

# ── 최종 모델 저장 ────────────────────────────────────────────────────
forecaster.save('chronos_forecaster')
joblib.dump(scaler, 'scaler_chronos.pkl')
# 체크포인트 파일 정리
if os.path.exists(CKPT_EPOCH):
    os.remove(CKPT_EPOCH)
print("\n모델 저장 완료: chronos_forecaster / scaler_chronos.pkl")

# ── 평가 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("테스트셋 평가 중...")
y_hat          = forecaster.predict(tsdata_test)
y_hat_unscaled = tsdata_test.unscale_numpy(y_hat)
y_unscaled     = tsdata_test.unscale_numpy(tsdata_test.to_numpy()[1])

mae = Evaluator.evaluate(metrics='mae', y_true=y_unscaled,
                          y_pred=y_hat_unscaled, aggregate=None)
mse = Evaluator.evaluate(metrics='mse', y_true=y_unscaled,
                          y_pred=y_hat_unscaled, aggregate=None)

print(f"\nFeature별 MAE (재학습 후):")
for f, e in zip(FEATURES, mae):
    print(f"  {f:12s}: {float(np.array(e).mean()):.4f}")
print(f"\n평균 MAE: {float(np.array(mae).mean()):.4f}")
print(f"평균 MSE: {float(np.array(mse).mean()):.6f}")

# ── 추론 latency ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
latencies = []
x_single = tsdata_test.to_numpy()[0][:1]
for _ in range(200):
    t0 = time.perf_counter()
    forecaster.predict(x_single)
    latencies.append((time.perf_counter() - t0) * 1000)
lat = np.array(latencies)
print(f"Inference Latency (n=200):")
print(f"  평균:   {lat.mean():.2f} ms")
print(f"  중앙값: {np.median(lat):.2f} ms")
print(f"  P95:    {np.percentile(lat, 95):.2f} ms")
print(f"  P99:    {np.percentile(lat, 99):.2f} ms")
print("=" * 60)
