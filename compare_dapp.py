"""
compare_dapp.py — Chronos TCN dApp ON vs OFF fairness 비교

시나리오 : UE1(21ab, 정상) vs UE2(f402, ploss → SNR 낮음)
           동일 kpi_live.csv 데이터, dApp 유무만 다름

dApp OFF : OAI 기본 스케줄러 → 두 UE에 PRB 동일 분배 → byte 불공평
dApp ON  : Chronos 예측 기반 가중치 → SNR 낮은 f402에 PRB 더 배분 → byte 개선
지표     : Jain's Fairness Index (ul_bytes 기준)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import TSDataset

FEATURES        = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK        = 10
FORECASTER_PATH = 'chronos_forecaster'
SCALER_PATH     = 'scaler_chronos.pkl'
KPI_CSV         = 'kpi_live.csv'
TARGET_UES      = ['21ab', 'f402']   # 동시 활성 세션 (Jun 24-26)
WINDOW          = '10s'


def jains(vals):
    vals = np.array([v for v in vals if v > 0], dtype=float)
    if len(vals) == 0:
        return float('nan')
    return vals.sum()**2 / (len(vals) * (vals**2).sum())


# ── 모델 로드 ──────────────────────────────────────────────────────
print("=" * 60)
print("Chronos TCN 모델 로드 중...")

scaler = joblib.load(SCALER_PATH)

df_tmp = pd.read_csv('kpi_baseline.csv', parse_dates=['timestamp'])
df_tmp = df_tmp.sort_values('timestamp').reset_index(drop=True)
df_tmp[FEATURES] = df_tmp[FEATURES].fillna(0)
tsdata_tmp, _, _ = TSDataset.from_pandas(
    df_tmp, dt_col='timestamp', target_col=FEATURES,
    with_split=True, val_ratio=0.15, test_ratio=0.15
)
sc_tmp = StandardScaler()
tsdata_tmp.scale(sc_tmp, fit=True)
tsdata_tmp.roll(lookback=LOOKBACK, horizon=1)
forecaster = TCNForecaster.from_tsdataset(tsdata_tmp)
forecaster.load(FORECASTER_PATH)
print("모델 로드 완료\n")

# ── 데이터 로드 ────────────────────────────────────────────────────
df_all = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_all[f] = pd.to_numeric(df_all[f], errors='coerce').fillna(0)
df_all['rnti'] = df_all['rnti'].astype(str)

# 21ab/f402 동시 활성 세션만 사용
df = df_all[df_all['rnti'].isin(TARGET_UES)].copy()
ues = TARGET_UES
n_ues = len(ues)
print(f"대상 UE: {ues}")
print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"총 행: {len(df)}\n")
print("UE별 실제 KPI 평균:")
print(df.groupby('rnti')[['snr','bler','nprb','ul_bytes']].mean().round(3))
print()

# ── UE별 슬라이딩 윈도 예측 ───────────────────────────────────────
print("Chronos 예측 계산 중 (UE별 처리)...")
ue_pred_df = []

for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp'] + FEATURES].copy()
    sub = sub.sort_values('timestamp').reset_index(drop=True)
    n = len(sub)

    for i in range(LOOKBACK, n):
        window = sub.iloc[i - LOOKBACK:i][FEATURES].values.astype(np.float32)
        window_s = scaler.transform(window)
        x = window_s[np.newaxis].astype(np.float32)
        pred_s = forecaster.predict(x).reshape(1, len(FEATURES))
        pred = scaler.inverse_transform(pred_s)[0]
        pred_dict = dict(zip(FEATURES, pred))

        ue_pred_df.append({
            'timestamp': sub.iloc[i]['timestamp'],
            'rnti':      ue,
            'pred_snr':  float(pred_dict['snr']),
            'pred_bler': float(pred_dict['bler']),
        })

    print(f"  UE {ue}: {n - LOOKBACK}개 예측 완료")

pred_df = pd.DataFrame(ue_pred_df)
print()

# ── 실제 nprb + bytes 집계 ───────────────────────────────────────
actual_df = df[['timestamp', 'rnti', 'nprb', 'ul_bytes']].copy()

# ── 10s 윈도우별 fairness 계산 (throughput 기준) ──────────────────
# throughput(ul_bytes) ∝ nprb × MCS_efficiency(SNR)
# bytes/PRB 비율을 측정해서 PRB 재분배 시 throughput 변화 추정
actual_df = actual_df.set_index('timestamp')
pred_df   = pred_df.set_index('timestamp')

# UE별 bytes/nprb 비율 (MCS efficiency proxy)
eff = {}
for ue in ues:
    sub = df[df['rnti'] == ue]
    valid = sub[sub['nprb'] > 0]
    eff[ue] = (valid['ul_bytes'] / valid['nprb']).mean() if len(valid) > 0 else 1.0
print("UE별 bytes/PRB 효율 (SNR 반영):")
for ue in ues:
    print(f"  {ue}: {eff[ue]:.2f} bytes/PRB")
print()

windows_off_nprb, windows_on_nprb   = [], []
windows_off_byte, windows_on_byte   = [], []
ue_nprb_off = {ue: [] for ue in ues}
ue_nprb_on  = {ue: [] for ue in ues}
ue_byte_off = {ue: [] for ue in ues}
ue_byte_on  = {ue: [] for ue in ues}

for period, grp_a in actual_df.groupby(pd.Grouper(freq=WINDOW)):
    if grp_a['rnti'].nunique() < 2:
        continue

    actual_nprb  = grp_a.groupby('rnti')['nprb'].sum()
    actual_bytes = grp_a.groupby('rnti')['ul_bytes'].sum()
    total_nprb   = actual_nprb.sum()
    if total_nprb == 0:
        continue

    # dApp ON 가중치 계산
    period_end = period + pd.Timedelta(WINDOW)
    grp_p = pred_df[(pred_df.index >= period) & (pred_df.index < period_end)]

    if grp_p.empty:
        weights = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_pred = grp_p.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
        scores = {}
        for ue in ues:
            if ue in avg_pred.index:
                snr  = max(float(avg_pred.loc[ue, 'pred_snr']), 0.1)
                bler = float(avg_pred.loc[ue, 'pred_bler'])
            else:
                snr, bler = 1.0, 0.0
            scores[ue] = (1.0 / snr) * (1.0 + bler * 5)
        total_score = sum(scores.values()) or 1.0
        weights = {ue: scores[ue] / total_score for ue in ues}

    # dApp ON: PRB 재분배 → 예상 throughput (bytes/PRB 효율 적용)
    dapp_nprb  = {ue: total_nprb * weights[ue] for ue in ues}
    dapp_bytes = {ue: dapp_nprb[ue] * eff[ue]  for ue in ues}

    # dApp OFF: 실제 OAI bytes
    off_bytes = {ue: float(actual_bytes.get(ue, 0)) for ue in ues}

    f_nprb_off = jains([actual_nprb.get(ue, 0)  for ue in ues])
    f_nprb_on  = jains([dapp_nprb[ue]            for ue in ues])
    f_byte_off = jains([off_bytes[ue]             for ue in ues])
    f_byte_on  = jains([dapp_bytes[ue]            for ue in ues])

    windows_off_nprb.append(f_nprb_off)
    windows_on_nprb.append(f_nprb_on)
    windows_off_byte.append(f_byte_off)
    windows_on_byte.append(f_byte_on)

    for ue in ues:
        ue_nprb_off[ue].append(actual_nprb.get(ue, 0))
        ue_nprb_on[ue].append(dapp_nprb[ue])
        ue_byte_off[ue].append(off_bytes[ue])
        ue_byte_on[ue].append(dapp_bytes[ue])

n_win = len(windows_off_byte)

# ── 결과 출력 ─────────────────────────────────────────────────────
print("=" * 62)
print(f"Jain's Fairness Index — Throughput (ul_bytes), {WINDOW} 윈도우 n={n_win}")
print("=" * 62)
b_off = np.nanmean(windows_off_byte)
b_on  = np.nanmean(windows_on_byte)
gap   = 1.0 - b_off
print(f"  dApp OFF (OAI 기본):  {b_off:.4f}")
print(f"  dApp ON  (Chronos) :  {b_on:.4f}")
print(f"  향상:                 {b_on - b_off:+.4f}"
      + (f"  ({(b_on-b_off)/gap*100:.1f}% gap 감소)" if gap > 0 else ""))

print()
print("=" * 62)
print(f"Jain's Fairness Index — nPRB, {WINDOW} 윈도우 n={n_win}")
print("=" * 62)
n_off = np.nanmean(windows_off_nprb)
n_on  = np.nanmean(windows_on_nprb)
print(f"  dApp OFF (OAI 기본):  {n_off:.4f}")
print(f"  dApp ON  (Chronos) :  {n_on:.4f}")
print(f"  향상:                 {n_on - n_off:+.4f}")

print()
print("=" * 62)
print("UE별 평균 분배 비교")
print("=" * 62)
print(f"  {'UE':>6}  {'SNR':>6}  "
      f"{'nPRB_OFF':>9} {'nPRB_ON':>8}  "
      f"{'bytes_OFF':>10} {'bytes_ON':>9}")
for ue in ues:
    snr_real = df[df['rnti'] == ue]['snr'].mean()
    no = np.mean(ue_nprb_off[ue]); nn = np.mean(ue_nprb_on[ue])
    bo = np.mean(ue_byte_off[ue]); bn = np.mean(ue_byte_on[ue])
    print(f"  {ue:>6}  {snr_real:>6.1f}  "
          f"{no:>9.1f} {nn:>8.1f}  "
          f"{bo:>10.1f} {bn:>9.1f}")

print()
print("=" * 62)
print("dApp 예측 기반 평균 가중치")
print("=" * 62)
pred_avg_all = pred_df.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
scores_all = {}
for ue in ues:
    snr  = max(float(pred_avg_all.loc[ue, 'pred_snr'])  if ue in pred_avg_all.index else 1.0, 0.1)
    bler = float(pred_avg_all.loc[ue, 'pred_bler']) if ue in pred_avg_all.index else 0.0
    scores_all[ue] = (1.0 / snr) * (1.0 + bler * 5)
total_all = sum(scores_all.values()) or 1.0
print(f"  {'UE':>6}  {'pred_SNR':>9}  {'pred_BLER':>9}  {'weight':>7}")
for ue in ues:
    snr_p  = pred_avg_all.loc[ue, 'pred_snr']  if ue in pred_avg_all.index else 0
    bler_p = pred_avg_all.loc[ue, 'pred_bler'] if ue in pred_avg_all.index else 0
    print(f"  {ue:>6}  {snr_p:>9.2f}  {bler_p:>9.4f}  {scores_all[ue]/total_all:>7.4f}")
print("=" * 62)
