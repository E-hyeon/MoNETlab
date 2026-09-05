"""
analyze_variability_segments.py — 약한 UE(SNR 낮은 쪽)의 SNR/BLER 변동성 구간 분석

compare_dapp.py와 동일한 파이프라인(Chronos/EWMA/Reactive Baseline 가중치 계산)을
10s 윈도우 단위로 재현하면서, 약한 UE의 SNR 표준편차(rolling std)를 함께 추적한다.
변동이 큰 구간과 안정적인 구간이 실제로 나뉘는지 먼저 확인하고,
나뉜다면 각 구간에서 Chronos/EWMA/Reactive Baseline의 Jain's Fairness를 따로 비교한다.
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
MIN_SAMPLES     = 50
WINDOW          = '10s'
EWMA_ALPHA      = 0.3
ROLL_WIN        = 6   # 60s rolling std 기준


def jains(vals):
    vals = np.array([v for v in vals if v > 0], dtype=float)
    if len(vals) == 0:
        return float('nan')
    return vals.sum()**2 / (len(vals) * (vals**2).sum())


def pick_target_ues(df_all, min_samples=MIN_SAMPLES):
    counts = df_all['rnti'].value_counts()
    candidates = counts[counts >= min_samples].index.tolist()
    if len(candidates) < 2:
        candidates = counts.index.tolist()
    snr_mean = df_all[df_all['rnti'].isin(candidates)].groupby('rnti')['snr'].mean()
    hi_ue, lo_ue = snr_mean.idxmax(), snr_mean.idxmin()
    return [hi_ue, lo_ue]


def score_weights(snr_bler, ues):
    scores = {}
    for ue in ues:
        snr, bler = snr_bler.get(ue, (1.0, 0.0))
        snr = max(snr, 0.1)
        scores[ue] = (1.0 / snr) * (1.0 + bler * 5)
    total = sum(scores.values()) or 1.0
    return {ue: scores[ue] / total for ue in ues}


print("모델 로드 중...")
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
print("완료\n")

df_all = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_all[f] = pd.to_numeric(df_all[f], errors='coerce').fillna(0)
df_all['rnti'] = df_all['rnti'].astype(str)

ues = pick_target_ues(df_all)
df = df_all[df_all['rnti'].isin(ues)].copy()
n_ues = len(ues)
snr_mean_by_ue = df.groupby('rnti')['snr'].mean()
weak_ue = snr_mean_by_ue.idxmin()
strong_ue = snr_mean_by_ue.idxmax()
print(f"대상 UE: {ues}  (약한 UE={weak_ue}, 강한 UE={strong_ue})\n")

# ── Chronos 예측 ────────────────────────────────────────────────────
print("Chronos 예측 계산 중...")
ue_pred_rows = []
for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp'] + FEATURES].sort_values('timestamp').reset_index(drop=True)
    for i in range(LOOKBACK, len(sub)):
        window = sub.iloc[i - LOOKBACK:i][FEATURES].values.astype(np.float32)
        pred_s = forecaster.predict(scaler.transform(window)[np.newaxis].astype(np.float32)).reshape(1, len(FEATURES))
        pred = scaler.inverse_transform(pred_s)[0]
        pd_dict = dict(zip(FEATURES, pred))
        ue_pred_rows.append({'timestamp': sub.iloc[i]['timestamp'], 'rnti': ue,
                              'pred_snr': pd_dict['snr'], 'pred_bler': pd_dict['bler']})
pred_df = pd.DataFrame(ue_pred_rows).set_index('timestamp')
print("완료\n")

# ── EWMA(alpha=0.3) 예측 ─────────────────────────────────────────────
print("EWMA 예측 계산 중...")
ewma_rows = []
for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp', 'snr', 'bler']].sort_values('timestamp').reset_index(drop=True)
    level = sub[['snr', 'bler']].ewm(alpha=EWMA_ALPHA, adjust=False).mean().shift(1)
    for i in range(len(sub)):
        if pd.isna(level.loc[i, 'snr']):
            continue
        ewma_rows.append({'timestamp': sub.iloc[i]['timestamp'], 'rnti': ue,
                           'ewma_snr': float(level.loc[i, 'snr']), 'ewma_bler': float(level.loc[i, 'bler'])})
ewma_pred_df = pd.DataFrame(ewma_rows).set_index('timestamp')
print("완료\n")

actual_df = df[['timestamp', 'rnti', 'nprb', 'ul_bytes', 'snr', 'bler']].set_index('timestamp')

eff = {}
for ue in ues:
    sub = df[df['rnti'] == ue]
    valid = sub[sub['nprb'] > 0]
    eff[ue] = (valid['ul_bytes'] / valid['nprb']).mean() if len(valid) > 0 else 1.0

# ── 10s 윈도우별 계산 ─────────────────────────────────────────────
rows = []
for period, grp_a in actual_df.groupby(pd.Grouper(freq=WINDOW)):
    if grp_a['rnti'].nunique() < 2:
        continue
    actual_nprb  = grp_a.groupby('rnti')['nprb'].sum()
    actual_bytes = grp_a.groupby('rnti')['ul_bytes'].sum()
    total_nprb   = actual_nprb.sum()
    if total_nprb == 0:
        continue

    period_end = period + pd.Timedelta(WINDOW)

    grp_p = pred_df[(pred_df.index >= period) & (pred_df.index < period_end)]
    if grp_p.empty:
        weights_on = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_pred = grp_p.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
        weights_on = score_weights({ue: (float(avg_pred.loc[ue, 'pred_snr']), float(avg_pred.loc[ue, 'pred_bler']))
                                     for ue in ues if ue in avg_pred.index}, ues)

    avg_actual = grp_a.groupby('rnti')[['snr', 'bler']].mean()
    weights_react = score_weights({ue: (float(avg_actual.loc[ue, 'snr']), float(avg_actual.loc[ue, 'bler']))
                                    for ue in ues if ue in avg_actual.index}, ues)

    grp_e = ewma_pred_df[(ewma_pred_df.index >= period) & (ewma_pred_df.index < period_end)]
    if grp_e.empty:
        weights_ewma = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_ewma = grp_e.groupby('rnti')[['ewma_snr', 'ewma_bler']].mean()
        weights_ewma = score_weights({ue: (float(avg_ewma.loc[ue, 'ewma_snr']), float(avg_ewma.loc[ue, 'ewma_bler']))
                                       for ue in ues if ue in avg_ewma.index}, ues)

    dapp_bytes  = {ue: total_nprb * weights_on[ue]    * eff[ue] for ue in ues}
    react_bytes = {ue: total_nprb * weights_react[ue] * eff[ue] for ue in ues}
    ewma_bytes  = {ue: total_nprb * weights_ewma[ue]  * eff[ue] for ue in ues}
    off_bytes   = {ue: float(actual_bytes.get(ue, 0)) for ue in ues}

    rows.append({
        'period':    period,
        'weak_snr':  float(avg_actual.loc[weak_ue, 'snr'])  if weak_ue in avg_actual.index else np.nan,
        'weak_bler': float(avg_actual.loc[weak_ue, 'bler']) if weak_ue in avg_actual.index else np.nan,
        'f_off':   jains([off_bytes[ue]   for ue in ues]),
        'f_on':    jains([dapp_bytes[ue]  for ue in ues]),
        'f_react': jains([react_bytes[ue] for ue in ues]),
        'f_ewma':  jains([ewma_bytes[ue]  for ue in ues]),
    })

res = pd.DataFrame(rows).set_index('period')
print(f"윈도우 수: {len(res)}\n")

# ── 변동성(rolling std) 계산 ─────────────────────────────────────────
res['snr_roll_std']  = res['weak_snr'].rolling(ROLL_WIN, min_periods=3).std()
res['snr_roll_rate'] = res['weak_snr'].diff().abs().rolling(ROLL_WIN, min_periods=3).mean()
res['bler_roll_std'] = res['weak_bler'].rolling(ROLL_WIN, min_periods=3).std()

print("=" * 78)
print(f"약한 UE({weak_ue}) SNR — 전체 통계")
print("=" * 78)
print(f"  전체 SNR 평균={res['weak_snr'].mean():.2f}dB  표준편차(전체)={res['weak_snr'].std():.2f}dB")
print(f"  {ROLL_WIN}윈도우({ROLL_WIN*10}s) rolling std — 평균={res['snr_roll_std'].mean():.3f}  "
      f"중앙값={res['snr_roll_std'].median():.3f}  최대={res['snr_roll_std'].max():.3f}  최소={res['snr_roll_std'].min():.3f}")
print(f"  rolling 변화율(|diff| 평균) — 평균={res['snr_roll_rate'].mean():.3f}  최대={res['snr_roll_rate'].max():.3f}")
print()

# ── rolling std 시계열 출력 (구간 확인용) ────────────────────────────
print("=" * 78)
print(f"시간대별 SNR / rolling std ({ROLL_WIN*10}s 창) — 5개 윈도우 간격 샘플링")
print("=" * 78)
for i in range(0, len(res), 5):
    row = res.iloc[i]
    ts = res.index[i]
    std_val = row['snr_roll_std']
    marker = '*' if pd.notna(std_val) and std_val > res['snr_roll_std'].median() * 1.5 else ' '
    print(f"  {ts.strftime('%H:%M:%S')}  SNR={row['weak_snr']:6.2f}dB  "
          f"roll_std={std_val if pd.isna(std_val) else round(std_val,3):>6}  {marker}")

# ── 변동 구간 vs 안정 구간 분리 ───────────────────────────────────────
threshold = res['snr_roll_std'].median() * 1.5
res['volatile'] = res['snr_roll_std'] > threshold

# 연속 구간(run) 추출
segments = []
cur_state = None
seg_start = None
for ts, is_vol in res['volatile'].items():
    if pd.isna(is_vol):
        continue
    if cur_state is None:
        cur_state, seg_start = is_vol, ts
    elif is_vol != cur_state:
        segments.append((cur_state, seg_start, ts))
        cur_state, seg_start = is_vol, ts
if cur_state is not None:
    segments.append((cur_state, seg_start, res.index[-1]))

print()
print("=" * 78)
print(f"변동 구간 판정 (rolling std > 중앙값×1.5 = {threshold:.3f})")
print("=" * 78)
long_segments = [(s, a, b) for s, a, b in segments if (b - a).total_seconds() >= 60]
for is_vol, a, b in long_segments:
    label = "변동 큼" if is_vol else "안정적"
    print(f"  [{label}] {a.strftime('%H:%M:%S')} ~ {b.strftime('%H:%M:%S')}  ({(b-a).total_seconds():.0f}s)")

n_vol_windows = int(res['volatile'].sum())
n_stable_windows = int((~res['volatile'].fillna(False)).sum()) - int(res['volatile'].isna().sum())
print(f"\n  전체 {len(res)}개 윈도우 중 변동 큰 윈도우 {n_vol_windows}개 / 안정적 윈도우 {n_stable_windows}개")

# ── 구간별 Chronos/EWMA/Reactive 성능 비교 ───────────────────────────
print()
print("=" * 78)
print("구간별 Jain's Fairness (Throughput) 비교")
print("=" * 78)
for label, mask in [("변동 큰 구간", res['volatile'] == True), ("안정적 구간", res['volatile'] == False)]:
    sub = res[mask]
    if len(sub) == 0:
        print(f"  [{label}] 윈도우 없음")
        continue
    print(f"  [{label}] n={len(sub)}개 윈도우")
    print(f"    OAI 기본            : {sub['f_off'].mean():.4f}")
    print(f"    Chronos dApp        : {sub['f_on'].mean():.4f}   향상: {sub['f_on'].mean()-sub['f_off'].mean():+.4f}")
    print(f"    Reactive Baseline   : {sub['f_react'].mean():.4f}   향상: {sub['f_react'].mean()-sub['f_off'].mean():+.4f}")
    print(f"    EWMA dApp           : {sub['f_ewma'].mean():.4f}   향상: {sub['f_ewma'].mean()-sub['f_off'].mean():+.4f}")
    print()

print("=" * 78)
