"""
plot_dapp_compare.py — OAI 기본 / Chronos dApp(TCN 예측) / MLP dApp / EWMA dApp / Reactive Baseline(no AI) 5자 비교 그래프
출력: dapp_compare.png (3개 subplot)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
COLOR_OFF        = '#5B8DB8'
COLOR_ON         = '#E07B54'
COLOR_MLP        = '#D4B23C'
COLOR_REACT      = '#5FA777'
COLOR_EWMA       = '#9B7FC7'


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

mlp_scaler = joblib.load(MLP_SCALER_PATH)
mlp_model  = joblib.load(MLP_MODEL_PATH)
print("완료\n")

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
ue_snr_mean = {ue: df[df['rnti'] == ue]['snr'].mean() for ue in ues}

# ── UE별 예측 (Chronos TCN) ─────────────────────────────────────────
print("Chronos 예측 중...")
ue_pred_df = []
for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp'] + FEATURES].sort_values('timestamp').reset_index(drop=True)
    for i in range(LOOKBACK, len(sub)):
        window = sub.iloc[i - LOOKBACK:i][FEATURES].values.astype(np.float32)
        pred_s = forecaster.predict(scaler.transform(window)[np.newaxis].astype(np.float32)).reshape(1, len(FEATURES))
        pred   = scaler.inverse_transform(pred_s)[0]
        pd_dict = dict(zip(FEATURES, pred))
        ue_pred_df.append({'timestamp': sub.iloc[i]['timestamp'], 'rnti': ue,
                           'pred_snr': pd_dict['snr'], 'pred_bler': pd_dict['bler']})
    print(f"  UE {ue}: {len(sub)-LOOKBACK}개 완료")

pred_df = pd.DataFrame(ue_pred_df).set_index('timestamp')

# ── UE별 예측 (MLP) ──────────────────────────────────────────────────
print("MLP 예측 중...")
mlp_pred_rows = []
for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp'] + FEATURES].sort_values('timestamp').reset_index(drop=True)
    for i in range(LOOKBACK, len(sub)):
        window = sub.iloc[i - LOOKBACK:i][FEATURES].values.astype(np.float32)
        window_s = mlp_scaler.transform(window)
        x = window_s.flatten()[np.newaxis, :].astype(np.float32)
        pred_s = mlp_model.predict(x).reshape(1, len(FEATURES))
        pred = mlp_scaler.inverse_transform(pred_s)[0]
        pd_dict = dict(zip(FEATURES, pred))
        mlp_pred_rows.append({'timestamp': sub.iloc[i]['timestamp'], 'rnti': ue,
                               'mlp_snr': pd_dict['snr'], 'mlp_bler': pd_dict['bler']})
    print(f"  UE {ue}: {len(sub)-LOOKBACK}개 완료")

mlp_pred_df = pd.DataFrame(mlp_pred_rows).set_index('timestamp')

# ── UE별 EWMA(alpha=0.3) 다음 값 예측 ──────────────────────────────
print(f"EWMA(alpha={EWMA_ALPHA}) 예측 중...")
ewma_pred_rows = []
for ue in ues:
    sub = df[df['rnti'] == ue][['timestamp', 'snr', 'bler']].sort_values('timestamp').reset_index(drop=True)
    level = sub[['snr', 'bler']].ewm(alpha=EWMA_ALPHA, adjust=False).mean().shift(1)
    n_valid = 0
    for i in range(len(sub)):
        if pd.isna(level.loc[i, 'snr']):
            continue
        ewma_pred_rows.append({'timestamp': sub.iloc[i]['timestamp'], 'rnti': ue,
                                'ewma_snr': float(level.loc[i, 'snr']),
                                'ewma_bler': float(level.loc[i, 'bler'])})
        n_valid += 1
    print(f"  UE {ue}: {n_valid}개 완료")

ewma_pred_df = pd.DataFrame(ewma_pred_rows).set_index('timestamp')

# ── bytes/PRB 효율 ─────────────────────────────────────────────────
eff = {}
for ue in ues:
    sub = df[df['rnti'] == ue]
    valid = sub[sub['nprb'] > 0]
    eff[ue] = (valid['ul_bytes'] / valid['nprb']).mean() if len(valid) > 0 else 1.0

def score_weights(snr_bler: dict) -> dict:
    """{ue: (snr, bler)} → fairness 가중치 (dApp 컨트롤러와 동일 산식)"""
    scores = {}
    for ue in ues:
        snr, bler = snr_bler.get(ue, (1.0, 0.0))
        snr = max(snr, 0.1)
        scores[ue] = (1.0 / snr) * (1.0 + bler * 5)
    total_score = sum(scores.values()) or 1.0
    return {ue: scores[ue] / total_score for ue in ues}


# ── 10s 윈도우별 계산 (snr/bler는 Reactive Baseline용으로 함께 보관) ──
actual_df = df[['timestamp', 'rnti', 'nprb', 'ul_bytes', 'snr', 'bler']].set_index('timestamp')

ts_list = []
f_byte_off_list, f_byte_on_list, f_byte_mlp_list, f_byte_react_list, f_byte_ewma_list = [], [], [], [], []
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

    # Chronos dApp: 슬라이딩 윈도로 예측된 다음 스텝 SNR/BLER 기반 가중치
    grp_p = pred_df[(pred_df.index >= period) & (pred_df.index < period_end)]
    if grp_p.empty:
        weights_on = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_pred = grp_p.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
        weights_on = score_weights({
            ue: (float(avg_pred.loc[ue, 'pred_snr']), float(avg_pred.loc[ue, 'pred_bler']))
            for ue in ues if ue in avg_pred.index
        })

    # MLP dApp: 슬라이딩 윈도로 예측된 다음 스텝 SNR/BLER 기반 가중치
    grp_m = mlp_pred_df[(mlp_pred_df.index >= period) & (mlp_pred_df.index < period_end)]
    if grp_m.empty:
        weights_mlp = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_mlp = grp_m.groupby('rnti')[['mlp_snr', 'mlp_bler']].mean()
        weights_mlp = score_weights({
            ue: (float(avg_mlp.loc[ue, 'mlp_snr']), float(avg_mlp.loc[ue, 'mlp_bler']))
            for ue in ues if ue in avg_mlp.index
        })

    # Reactive Baseline (AI 예측 없음): "현재" 윈도우의 실측 SNR/BLER 평균만 사용
    avg_actual = grp_a.groupby('rnti')[['snr', 'bler']].mean()
    weights_react = score_weights({
        ue: (float(avg_actual.loc[ue, 'snr']), float(avg_actual.loc[ue, 'bler']))
        for ue in ues if ue in avg_actual.index
    })

    # EWMA dApp: alpha=0.3 지수가중이동평균으로 예측된 다음 값 사용
    grp_e = ewma_pred_df[(ewma_pred_df.index >= period) & (ewma_pred_df.index < period_end)]
    if grp_e.empty:
        weights_ewma = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_ewma = grp_e.groupby('rnti')[['ewma_snr', 'ewma_bler']].mean()
        weights_ewma = score_weights({
            ue: (float(avg_ewma.loc[ue, 'ewma_snr']), float(avg_ewma.loc[ue, 'ewma_bler']))
            for ue in ues if ue in avg_ewma.index
        })

    dapp_nprb    = {ue: total_nprb * weights_on[ue]    for ue in ues}
    dapp_bytes   = {ue: dapp_nprb[ue] * eff[ue]         for ue in ues}
    mlp_nprb     = {ue: total_nprb * weights_mlp[ue]   for ue in ues}
    mlp_bytes    = {ue: mlp_nprb[ue] * eff[ue]          for ue in ues}
    react_nprb   = {ue: total_nprb * weights_react[ue] for ue in ues}
    react_bytes  = {ue: react_nprb[ue] * eff[ue]        for ue in ues}
    ewma_nprb    = {ue: total_nprb * weights_ewma[ue]   for ue in ues}
    ewma_bytes   = {ue: ewma_nprb[ue] * eff[ue]          for ue in ues}
    off_bytes    = {ue: float(actual_bytes.get(ue, 0)) for ue in ues}

    f_off   = jains([off_bytes[ue]   for ue in ues])
    f_on    = jains([dapp_bytes[ue]  for ue in ues])
    f_mlp   = jains([mlp_bytes[ue]   for ue in ues])
    f_react = jains([react_bytes[ue] for ue in ues])
    f_ewma  = jains([ewma_bytes[ue]  for ue in ues])

    ts_list.append(period)
    f_byte_off_list.append(f_off)
    f_byte_on_list.append(f_on)
    f_byte_mlp_list.append(f_mlp)
    f_byte_react_list.append(f_react)
    f_byte_ewma_list.append(f_ewma)
    for ue in ues:
        ue_byte_off[ue].append(off_bytes[ue])
        ue_byte_on[ue].append(dapp_bytes[ue])
        ue_byte_mlp[ue].append(mlp_bytes[ue])
        ue_byte_react[ue].append(react_bytes[ue])
        ue_byte_ewma[ue].append(ewma_bytes[ue])

print(f"\n윈도우 수: {len(ts_list)}")

# ── 집계 ──────────────────────────────────────────────────────────
mean_off   = np.nanmean(f_byte_off_list)
mean_on    = np.nanmean(f_byte_on_list)
mean_mlp   = np.nanmean(f_byte_mlp_list)
mean_react = np.nanmean(f_byte_react_list)
mean_ewma  = np.nanmean(f_byte_ewma_list)
ue_avg_off   = {ue: np.mean(ue_byte_off[ue])   for ue in ues}
ue_avg_on    = {ue: np.mean(ue_byte_on[ue])    for ue in ues}
ue_avg_mlp   = {ue: np.mean(ue_byte_mlp[ue])   for ue in ues}
ue_avg_react = {ue: np.mean(ue_byte_react[ue]) for ue in ues}
ue_avg_ewma  = {ue: np.mean(ue_byte_ewma[ue])  for ue in ues}

# ── 그래프 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("OAI Default vs Chronos vs MLP vs EWMA dApp vs Reactive Baseline — Fairness Comparison\n"
             f"(UE1: {ues[0]} SNR={ue_snr_mean[ues[0]]:.1f} dB  |  "
             f"UE2: {ues[1]} SNR={ue_snr_mean[ues[1]]:.1f} dB)",
             fontsize=12, fontweight='bold')

LABELS  = ['OAI default', 'Chronos dApp\n(TCN prediction)', 'MLP dApp\n(1 hidden layer)',
           'EWMA dApp\n(alpha=0.3)', 'Reactive Baseline\n(no AI prediction)']
COLORS  = [COLOR_OFF, COLOR_ON, COLOR_MLP, COLOR_EWMA, COLOR_REACT]

# ── (1) Jain's Fairness 막대 ──────────────────────────────────────
ax = axes[0]
vals = [mean_off, mean_on, mean_mlp, mean_ewma, mean_react]
bars = ax.bar(LABELS, vals, color=COLORS, width=0.6, edgecolor='white', linewidth=1.2)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.set_title("Throughput Fairness", fontsize=11, fontweight='bold')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=7.5)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# ── (2) UE별 Throughput 막대 ─────────────────────────────────────
ax = axes[1]
ue_labels = [f'UE1\n({ues[0]})\nSNR={ue_snr_mean[ues[0]]:.1f}dB',
             f'UE2\n({ues[1]})\nSNR={ue_snr_mean[ues[1]]:.1f}dB']
x = np.arange(2)
w = 0.16
b1 = ax.bar(x - 2*w, [ue_avg_off[ue]   for ue in ues], w,
            label=LABELS[0], color=COLOR_OFF,   edgecolor='white')
b2 = ax.bar(x - 1*w, [ue_avg_on[ue]    for ue in ues], w,
            label=LABELS[1], color=COLOR_ON,    edgecolor='white')
b3 = ax.bar(x,       [ue_avg_mlp[ue]   for ue in ues], w,
            label=LABELS[2], color=COLOR_MLP,   edgecolor='white')
b4 = ax.bar(x + 1*w, [ue_avg_ewma[ue]  for ue in ues], w,
            label=LABELS[3], color=COLOR_EWMA,  edgecolor='white')
b5 = ax.bar(x + 2*w, [ue_avg_react[ue] for ue in ues], w,
            label=LABELS[4], color=COLOR_REACT, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(ue_labels, fontsize=10)
ax.set_ylabel("Avg UL Throughput (bytes/10s)", fontsize=11)
ax.set_title("Per-UE Throughput", fontsize=11, fontweight='bold')
for bar in list(b1) + list(b2) + list(b3) + list(b4) + list(b5):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
ax.legend(fontsize=6.5)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# ── (3) 시간축 Fairness 추이 ─────────────────────────────────────
ax = axes[2]
ts = pd.Series(ts_list)
roll = 20   # 200s 이동평균
f_off_s   = pd.Series(f_byte_off_list).rolling(roll, min_periods=1).mean()
f_on_s    = pd.Series(f_byte_on_list).rolling(roll, min_periods=1).mean()
f_mlp_s   = pd.Series(f_byte_mlp_list).rolling(roll, min_periods=1).mean()
f_react_s = pd.Series(f_byte_react_list).rolling(roll, min_periods=1).mean()
f_ewma_s  = pd.Series(f_byte_ewma_list).rolling(roll, min_periods=1).mean()

ax.plot(ts, f_off_s,   color=COLOR_OFF,   linewidth=1.5, label=LABELS[0], alpha=0.9)
ax.plot(ts, f_on_s,    color=COLOR_ON,    linewidth=1.5, label=LABELS[1], alpha=0.9)
ax.plot(ts, f_mlp_s,   color=COLOR_MLP,   linewidth=1.5, label=LABELS[2], alpha=0.9)
ax.plot(ts, f_ewma_s,  color=COLOR_EWMA,  linewidth=1.5, label=LABELS[3], alpha=0.9)
ax.plot(ts, f_react_s, color=COLOR_REACT, linewidth=1.5, label=LABELS[4], alpha=0.9)
ax.fill_between(ts, f_off_s, f_on_s,
                where=(f_on_s >= f_off_s), alpha=0.08, color=COLOR_ON, label='Chronos gain')
ax.fill_between(ts, f_off_s, f_mlp_s,
                where=(f_mlp_s >= f_off_s), alpha=0.08, color=COLOR_MLP, label='MLP gain')
ax.fill_between(ts, f_off_s, f_ewma_s,
                where=(f_ewma_s >= f_off_s), alpha=0.08, color=COLOR_EWMA, label='EWMA gain')
ax.fill_between(ts, f_off_s, f_react_s,
                where=(f_react_s >= f_off_s), alpha=0.08, color=COLOR_REACT, label='Reactive gain')
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.set_title(f"Fairness Over Time ({roll*10}s rolling avg)", fontsize=11, fontweight='bold')
ax.axhline(mean_off,   color=COLOR_OFF,   linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(mean_on,    color=COLOR_ON,    linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(mean_mlp,   color=COLOR_MLP,   linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(mean_ewma,  color=COLOR_EWMA,  linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(mean_react, color=COLOR_REACT, linestyle=':', linewidth=1, alpha=0.7)
ax.legend(fontsize=6.5)
ax.tick_params(axis='x', rotation=20)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d\n%H:%M'))
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dapp_compare.png', dpi=150, bbox_inches='tight')
print("\n저장 완료: dapp_compare.png")
