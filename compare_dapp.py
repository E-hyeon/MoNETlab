"""
compare_dapp.py — OAI 기본 / Chronos dApp(TCN 예측) / MLP dApp / EWMA dApp / Reactive Baseline(no AI) 5자 비교

시나리오 : kpi_live.csv에 동시 활성인 UE 중 평균 SNR 차이가 가장 큰 두 개를
           자동으로 선택 (정상 UE vs SNR 낮은 UE), 스케줄링 방식만 다름

OAI 기본(OFF)                        : 두 UE에 PRB 동일 분배 → byte 불공평
Chronos dApp                        : TCN 예측 SNR/BLER 기반 가중치 → 예측적으로 PRB 재분배
MLP dApp                            : 얕은 MLP(은닉층 1개, 뉴런 12개) 예측 기반 가중치
EWMA dApp                           : 지수가중이동평균(alpha=0.3)으로 다음 값을 예측해 가중치 적용
Reactive Baseline (no AI prediction) : 예측 없이 "현재" 측정 SNR/BLER만으로 같은 가중치 산식 적용
                                        (AI 예측이 없어 dApp이 아니라 baseline/비교군으로 분류)
지표                                 : Jain's Fairness Index (ul_bytes / nPRB 기준)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import TSDataset

FEATURES         = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK         = 10
FORECASTER_PATH  = 'chronos_forecaster'
SCALER_PATH      = 'scaler_chronos.pkl'
MLP_MODEL_PATH   = 'mlp_forecaster.pkl'
MLP_SCALER_PATH  = 'scaler_mlp.pkl'
KPI_CSV          = 'kpi_live.csv'
MIN_SAMPLES      = 50   # 자동 UE 선택 시 최소 표본 수 (너무 적은 UE 제외)
WINDOW           = '10s'
EWMA_ALPHA       = 0.3


def jains(vals):
    vals = np.array([v for v in vals if v > 0], dtype=float)
    if len(vals) == 0:
        return float('nan')
    return vals.sum()**2 / (len(vals) * (vals**2).sum())


def pick_target_ues(df_all: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> list:
    """평균 SNR 차이가 가장 큰 두 UE를 자동 선택 (표본 부족 UE는 제외)."""
    counts = df_all['rnti'].value_counts()
    candidates = counts[counts >= min_samples].index.tolist()
    if len(candidates) < 2:
        candidates = counts.index.tolist()
    snr_mean = df_all[df_all['rnti'].isin(candidates)].groupby('rnti')['snr'].mean()
    hi_ue, lo_ue = snr_mean.idxmax(), snr_mean.idxmin()
    return [hi_ue, lo_ue]


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
print("Chronos 모델 로드 완료")

mlp_scaler = joblib.load(MLP_SCALER_PATH)
mlp_model  = joblib.load(MLP_MODEL_PATH)
print(f"MLP 모델 로드 완료 (hidden_layer_sizes={mlp_model.hidden_layer_sizes})\n")

# ── 데이터 로드 ────────────────────────────────────────────────────
df_all = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_all[f] = pd.to_numeric(df_all[f], errors='coerce').fillna(0)
df_all['rnti'] = df_all['rnti'].astype(str)

# 평균 SNR 차이가 가장 큰 두 UE를 자동 선택
TARGET_UES = pick_target_ues(df_all)
df = df_all[df_all['rnti'].isin(TARGET_UES)].copy()
ues = TARGET_UES
n_ues = len(ues)
print(f"자동 선택된 대상 UE (SNR 차이 최대): {ues}")
print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"총 행: {len(df)}\n")
print("UE별 실제 KPI 평균:")
print(df.groupby('rnti')[['snr','bler','nprb','ul_bytes']].mean().round(3))
print()

# ── UE별 슬라이딩 윈도 예측 (Chronos TCN) ──────────────────────────
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

# ── UE별 슬라이딩 윈도 예측 (MLP) ───────────────────────────────────
print("MLP 예측 계산 중 (UE별 처리)...")
mlp_pred_rows = []

for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp'] + FEATURES].copy()
    sub = sub.sort_values('timestamp').reset_index(drop=True)
    n = len(sub)

    for i in range(LOOKBACK, n):
        window = sub.iloc[i - LOOKBACK:i][FEATURES].values.astype(np.float32)
        window_s = mlp_scaler.transform(window)
        x = window_s.flatten()[np.newaxis, :].astype(np.float32)
        pred_s = mlp_model.predict(x).reshape(1, len(FEATURES))
        pred = mlp_scaler.inverse_transform(pred_s)[0]
        pred_dict = dict(zip(FEATURES, pred))

        mlp_pred_rows.append({
            'timestamp': sub.iloc[i]['timestamp'],
            'rnti':      ue,
            'mlp_snr':   float(pred_dict['snr']),
            'mlp_bler':  float(pred_dict['bler']),
        })

    print(f"  UE {ue}: {n - LOOKBACK}개 예측 완료")

mlp_pred_df = pd.DataFrame(mlp_pred_rows)
print()

# ── UE별 EWMA(alpha=0.3) 다음 값 예측 ──────────────────────────────
print(f"EWMA(alpha={EWMA_ALPHA}) 예측 계산 중 (UE별 처리)...")
ewma_pred_rows = []

for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp', 'snr', 'bler']].copy()
    sub = sub.sort_values('timestamp').reset_index(drop=True)
    # level[i] = alpha*값[i] + (1-alpha)*level[i-1] (level[0]=값[0])
    # shift(1)한 값이 "이 시점 이전까지의 데이터로 예측한 다음 값"이 된다.
    level = sub[['snr', 'bler']].ewm(alpha=EWMA_ALPHA, adjust=False).mean().shift(1)

    n_valid = 0
    for i in range(len(sub)):
        if pd.isna(level.loc[i, 'snr']):
            continue
        ewma_pred_rows.append({
            'timestamp':  sub.iloc[i]['timestamp'],
            'rnti':       ue,
            'ewma_snr':   float(level.loc[i, 'snr']),
            'ewma_bler':  float(level.loc[i, 'bler']),
        })
        n_valid += 1

    print(f"  UE {ue}: {n_valid}개 예측 완료")

ewma_pred_df = pd.DataFrame(ewma_pred_rows)
print()

# ── 실제 nprb + bytes 집계 (snr/bler는 Reactive Baseline용으로 함께 보관) ──
actual_df = df[['timestamp', 'rnti', 'nprb', 'ul_bytes', 'snr', 'bler']].copy()

# ── 10s 윈도우별 fairness 계산 (throughput 기준) ──────────────────
# throughput(ul_bytes) ∝ nprb × MCS_efficiency(SNR)
# bytes/PRB 비율을 측정해서 PRB 재분배 시 throughput 변화 추정
actual_df    = actual_df.set_index('timestamp')
pred_df      = pred_df.set_index('timestamp')
mlp_pred_df  = mlp_pred_df.set_index('timestamp')
ewma_pred_df = ewma_pred_df.set_index('timestamp')

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

def score_weights(snr_bler: dict) -> dict:
    """{ue: (snr, bler)} → fairness 가중치 (compute_weights와 동일 산식)"""
    scores = {}
    for ue in ues:
        snr, bler = snr_bler.get(ue, (1.0, 0.0))
        snr = max(snr, 0.1)
        scores[ue] = (1.0 / snr) * (1.0 + bler * 5)
    total_score = sum(scores.values()) or 1.0
    return {ue: scores[ue] / total_score for ue in ues}


windows_off_nprb, windows_on_nprb, windows_mlp_nprb, windows_react_nprb, windows_ewma_nprb = [], [], [], [], []
windows_off_byte, windows_on_byte, windows_mlp_byte, windows_react_byte, windows_ewma_byte = [], [], [], [], []
ue_nprb_off   = {ue: [] for ue in ues}
ue_nprb_on    = {ue: [] for ue in ues}
ue_nprb_mlp   = {ue: [] for ue in ues}
ue_nprb_react = {ue: [] for ue in ues}
ue_nprb_ewma  = {ue: [] for ue in ues}
ue_byte_off   = {ue: [] for ue in ues}
ue_byte_on    = {ue: [] for ue in ues}
ue_byte_mlp   = {ue: [] for ue in ues}
ue_byte_react = {ue: [] for ue in ues}
ue_byte_ewma  = {ue: [] for ue in ues}

for period, grp_a in actual_df.groupby(pd.Grouper(freq=WINDOW)):
    if grp_a['rnti'].nunique() < 2:
        continue

    actual_nprb  = grp_a.groupby('rnti')['nprb'].sum()
    actual_bytes = grp_a.groupby('rnti')['ul_bytes'].sum()
    total_nprb   = actual_nprb.sum()
    if total_nprb == 0:
        continue

    period_end = period + pd.Timedelta(WINDOW)

    # Chronos dApp 가중치: 슬라이딩 윈도로 예측된 다음 스텝 SNR/BLER 사용
    grp_p = pred_df[(pred_df.index >= period) & (pred_df.index < period_end)]
    if grp_p.empty:
        weights_on = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_pred = grp_p.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
        weights_on = score_weights({
            ue: (float(avg_pred.loc[ue, 'pred_snr']), float(avg_pred.loc[ue, 'pred_bler']))
            for ue in ues if ue in avg_pred.index
        })

    # MLP dApp 가중치: 슬라이딩 윈도로 예측된 다음 스텝 SNR/BLER 사용
    grp_m = mlp_pred_df[(mlp_pred_df.index >= period) & (mlp_pred_df.index < period_end)]
    if grp_m.empty:
        weights_mlp = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_mlp = grp_m.groupby('rnti')[['mlp_snr', 'mlp_bler']].mean()
        weights_mlp = score_weights({
            ue: (float(avg_mlp.loc[ue, 'mlp_snr']), float(avg_mlp.loc[ue, 'mlp_bler']))
            for ue in ues if ue in avg_mlp.index
        })

    # Reactive Baseline 가중치 (AI 예측 없음): "현재" 윈도우의 실측 SNR/BLER 평균만 사용
    avg_actual = grp_a.groupby('rnti')[['snr', 'bler']].mean()
    weights_react = score_weights({
        ue: (float(avg_actual.loc[ue, 'snr']), float(avg_actual.loc[ue, 'bler']))
        for ue in ues if ue in avg_actual.index
    })

    # EWMA dApp 가중치: alpha=0.3 지수가중이동평균으로 예측된 다음 값 사용
    grp_e = ewma_pred_df[(ewma_pred_df.index >= period) & (ewma_pred_df.index < period_end)]
    if grp_e.empty:
        weights_ewma = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_ewma = grp_e.groupby('rnti')[['ewma_snr', 'ewma_bler']].mean()
        weights_ewma = score_weights({
            ue: (float(avg_ewma.loc[ue, 'ewma_snr']), float(avg_ewma.loc[ue, 'ewma_bler']))
            for ue in ues if ue in avg_ewma.index
        })

    # PRB 재분배 → 예상 throughput (bytes/PRB 효율 적용)
    dapp_nprb    = {ue: total_nprb * weights_on[ue]    for ue in ues}
    dapp_bytes   = {ue: dapp_nprb[ue] * eff[ue]         for ue in ues}
    mlp_nprb     = {ue: total_nprb * weights_mlp[ue]   for ue in ues}
    mlp_bytes    = {ue: mlp_nprb[ue] * eff[ue]          for ue in ues}
    react_nprb   = {ue: total_nprb * weights_react[ue] for ue in ues}
    react_bytes  = {ue: react_nprb[ue] * eff[ue]        for ue in ues}
    ewma_nprb    = {ue: total_nprb * weights_ewma[ue]   for ue in ues}
    ewma_bytes   = {ue: ewma_nprb[ue] * eff[ue]          for ue in ues}

    # OAI 기본(OFF): 실제 bytes
    off_bytes = {ue: float(actual_bytes.get(ue, 0)) for ue in ues}

    f_nprb_off   = jains([actual_nprb.get(ue, 0) for ue in ues])
    f_nprb_on    = jains([dapp_nprb[ue]          for ue in ues])
    f_nprb_mlp   = jains([mlp_nprb[ue]           for ue in ues])
    f_nprb_react = jains([react_nprb[ue]         for ue in ues])
    f_nprb_ewma  = jains([ewma_nprb[ue]          for ue in ues])
    f_byte_off   = jains([off_bytes[ue]          for ue in ues])
    f_byte_on    = jains([dapp_bytes[ue]         for ue in ues])
    f_byte_mlp   = jains([mlp_bytes[ue]          for ue in ues])
    f_byte_react = jains([react_bytes[ue]        for ue in ues])
    f_byte_ewma  = jains([ewma_bytes[ue]         for ue in ues])

    windows_off_nprb.append(f_nprb_off)
    windows_on_nprb.append(f_nprb_on)
    windows_mlp_nprb.append(f_nprb_mlp)
    windows_react_nprb.append(f_nprb_react)
    windows_ewma_nprb.append(f_nprb_ewma)
    windows_off_byte.append(f_byte_off)
    windows_on_byte.append(f_byte_on)
    windows_mlp_byte.append(f_byte_mlp)
    windows_react_byte.append(f_byte_react)
    windows_ewma_byte.append(f_byte_ewma)

    for ue in ues:
        ue_nprb_off[ue].append(actual_nprb.get(ue, 0))
        ue_nprb_on[ue].append(dapp_nprb[ue])
        ue_nprb_mlp[ue].append(mlp_nprb[ue])
        ue_nprb_react[ue].append(react_nprb[ue])
        ue_nprb_ewma[ue].append(ewma_nprb[ue])
        ue_byte_off[ue].append(off_bytes[ue])
        ue_byte_on[ue].append(dapp_bytes[ue])
        ue_byte_mlp[ue].append(mlp_bytes[ue])
        ue_byte_react[ue].append(react_bytes[ue])
        ue_byte_ewma[ue].append(ewma_bytes[ue])

n_win = len(windows_off_byte)

# ── 결과 출력 ─────────────────────────────────────────────────────
print("=" * 86)
print(f"Jain's Fairness Index — Throughput (ul_bytes), {WINDOW} 윈도우 n={n_win}")
print("=" * 86)
b_off   = np.nanmean(windows_off_byte)
b_on    = np.nanmean(windows_on_byte)
b_mlp   = np.nanmean(windows_mlp_byte)
b_react = np.nanmean(windows_react_byte)
b_ewma  = np.nanmean(windows_ewma_byte)
gap = 1.0 - b_off

def gap_pct(val):
    return f"  ({(val - b_off) / gap * 100:.1f}% gap 감소)" if gap > 0 else ""

print(f"  {'OAI 기본':<20}:  {b_off:.4f}")
print(f"  {'Chronos dApp':<20}:  {b_on:.4f}   향상: {b_on - b_off:+.4f}" + gap_pct(b_on))
print(f"  {'MLP dApp':<20}:  {b_mlp:.4f}   향상: {b_mlp - b_off:+.4f}" + gap_pct(b_mlp))
print(f"  {'EWMA dApp':<20}:  {b_ewma:.4f}   향상: {b_ewma - b_off:+.4f}" + gap_pct(b_ewma))
print(f"  {'Reactive Baseline':<20}:  {b_react:.4f}   향상: {b_react - b_off:+.4f}" + gap_pct(b_react))

print()
print("=" * 86)
print(f"Jain's Fairness Index — nPRB, {WINDOW} 윈도우 n={n_win}")
print("=" * 86)
n_off   = np.nanmean(windows_off_nprb)
n_on    = np.nanmean(windows_on_nprb)
n_mlp   = np.nanmean(windows_mlp_nprb)
n_react = np.nanmean(windows_react_nprb)
n_ewma  = np.nanmean(windows_ewma_nprb)
print(f"  {'OAI 기본':<20}:  {n_off:.4f}")
print(f"  {'Chronos dApp':<20}:  {n_on:.4f}   향상: {n_on - n_off:+.4f}")
print(f"  {'MLP dApp':<20}:  {n_mlp:.4f}   향상: {n_mlp - n_off:+.4f}")
print(f"  {'EWMA dApp':<20}:  {n_ewma:.4f}   향상: {n_ewma - n_off:+.4f}")
print(f"  {'Reactive Baseline':<20}:  {n_react:.4f}   향상: {n_react - n_off:+.4f}")

print()
print("=" * 86)
print("UE별 평균 분배 비교")
print("=" * 86)
print(f"  {'UE':>6}  {'SNR':>6}  "
      f"{'nPRB_OFF':>9} {'nPRB_ON':>8} {'nPRB_MLP':>9} {'nPRB_EWMA':>10} {'nPRB_RCT':>9}  "
      f"{'bytes_OFF':>10} {'bytes_ON':>9} {'bytes_MLP':>10} {'bytes_EWMA':>11} {'bytes_RCT':>10}")
for ue in ues:
    snr_real = df[df['rnti'] == ue]['snr'].mean()
    no = np.mean(ue_nprb_off[ue]); nn = np.mean(ue_nprb_on[ue]); nm = np.mean(ue_nprb_mlp[ue])
    ne = np.mean(ue_nprb_ewma[ue]); nr = np.mean(ue_nprb_react[ue])
    bo = np.mean(ue_byte_off[ue]); bn = np.mean(ue_byte_on[ue]); bm = np.mean(ue_byte_mlp[ue])
    be = np.mean(ue_byte_ewma[ue]); br = np.mean(ue_byte_react[ue])
    print(f"  {ue:>6}  {snr_real:>6.1f}  "
          f"{no:>9.1f} {nn:>8.1f} {nm:>9.1f} {ne:>10.1f} {nr:>9.1f}  "
          f"{bo:>10.1f} {bn:>9.1f} {bm:>10.1f} {be:>11.1f} {br:>10.1f}")

print()
print("=" * 86)
print("Chronos dApp — 예측(pred_snr/pred_bler) 기반 평균 가중치")
print("=" * 86)
pred_avg_all = pred_df.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
weights_on_all = score_weights({
    ue: (float(pred_avg_all.loc[ue, 'pred_snr']), float(pred_avg_all.loc[ue, 'pred_bler']))
    for ue in ues if ue in pred_avg_all.index
})
print(f"  {'UE':>6}  {'pred_SNR':>9}  {'pred_BLER':>9}  {'weight':>7}")
for ue in ues:
    snr_p  = pred_avg_all.loc[ue, 'pred_snr']  if ue in pred_avg_all.index else 0
    bler_p = pred_avg_all.loc[ue, 'pred_bler'] if ue in pred_avg_all.index else 0
    print(f"  {ue:>6}  {snr_p:>9.2f}  {bler_p:>9.4f}  {weights_on_all[ue]:>7.4f}")

print()
print("=" * 86)
print("MLP dApp — 예측(mlp_snr/mlp_bler) 기반 평균 가중치")
print("=" * 86)
mlp_avg_all = mlp_pred_df.groupby('rnti')[['mlp_snr', 'mlp_bler']].mean()
weights_mlp_all = score_weights({
    ue: (float(mlp_avg_all.loc[ue, 'mlp_snr']), float(mlp_avg_all.loc[ue, 'mlp_bler']))
    for ue in ues if ue in mlp_avg_all.index
})
print(f"  {'UE':>6}  {'mlp_SNR':>9}  {'mlp_BLER':>9}  {'weight':>7}")
for ue in ues:
    snr_m  = mlp_avg_all.loc[ue, 'mlp_snr']  if ue in mlp_avg_all.index else 0
    bler_m = mlp_avg_all.loc[ue, 'mlp_bler'] if ue in mlp_avg_all.index else 0
    print(f"  {ue:>6}  {snr_m:>9.2f}  {bler_m:>9.4f}  {weights_mlp_all[ue]:>7.4f}")

print()
print("=" * 86)
print(f"EWMA dApp — 예측(ewma_snr/ewma_bler, alpha={EWMA_ALPHA}) 기반 평균 가중치")
print("=" * 86)
ewma_avg_all = ewma_pred_df.groupby('rnti')[['ewma_snr', 'ewma_bler']].mean()
weights_ewma_all = score_weights({
    ue: (float(ewma_avg_all.loc[ue, 'ewma_snr']), float(ewma_avg_all.loc[ue, 'ewma_bler']))
    for ue in ues if ue in ewma_avg_all.index
})
print(f"  {'UE':>6}  {'ewma_SNR':>9}  {'ewma_BLER':>9}  {'weight':>7}")
for ue in ues:
    snr_e  = ewma_avg_all.loc[ue, 'ewma_snr']  if ue in ewma_avg_all.index else 0
    bler_e = ewma_avg_all.loc[ue, 'ewma_bler'] if ue in ewma_avg_all.index else 0
    print(f"  {ue:>6}  {snr_e:>9.2f}  {bler_e:>9.4f}  {weights_ewma_all[ue]:>7.4f}")

print()
print("=" * 86)
print("Reactive Baseline (no AI prediction) — 실측(현재 snr/bler) 기반 평균 가중치")
print("=" * 86)
actual_avg_all = df.groupby('rnti')[['snr', 'bler']].mean()
weights_react_all = score_weights({
    ue: (float(actual_avg_all.loc[ue, 'snr']), float(actual_avg_all.loc[ue, 'bler']))
    for ue in ues if ue in actual_avg_all.index
})
print(f"  {'UE':>6}  {'SNR':>9}  {'BLER':>9}  {'weight':>7}")
for ue in ues:
    snr_a  = actual_avg_all.loc[ue, 'snr']  if ue in actual_avg_all.index else 0
    bler_a = actual_avg_all.loc[ue, 'bler'] if ue in actual_avg_all.index else 0
    print(f"  {ue:>6}  {snr_a:>9.2f}  {bler_a:>9.4f}  {weights_react_all[ue]:>7.4f}")
print("=" * 86)
