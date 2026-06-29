"""
plot_dapp_compare.py — dApp ON vs OFF 비교 그래프 생성
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

FEATURES        = ['snr', 'bler', 'nprb', 'mcs_ul', 'ul_bytes', 'dl_bytes']
LOOKBACK        = 10
FORECASTER_PATH = 'chronos_forecaster'
SCALER_PATH     = 'scaler_chronos.pkl'
KPI_CSV         = 'kpi_live.csv'
TARGET_UES      = ['21ab', 'f402']
WINDOW          = '10s'
COLOR_OFF       = '#5B8DB8'
COLOR_ON        = '#E07B54'


def jains(vals):
    vals = np.array([v for v in vals if v > 0], dtype=float)
    if len(vals) == 0:
        return float('nan')
    return vals.sum()**2 / (len(vals) * (vals**2).sum())


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
print("완료\n")

# ── 데이터 로드 ────────────────────────────────────────────────────
df_all = pd.read_csv(KPI_CSV, parse_dates=['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)
for f in FEATURES:
    df_all[f] = pd.to_numeric(df_all[f], errors='coerce').fillna(0)
df_all['rnti'] = df_all['rnti'].astype(str)
df = df_all[df_all['rnti'].isin(TARGET_UES)].copy()
ues = TARGET_UES
n_ues = len(ues)

# ── UE별 예측 ─────────────────────────────────────────────────────
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

# ── bytes/PRB 효율 ─────────────────────────────────────────────────
eff = {}
for ue in ues:
    sub = df[df['rnti'] == ue]
    valid = sub[sub['nprb'] > 0]
    eff[ue] = (valid['ul_bytes'] / valid['nprb']).mean() if len(valid) > 0 else 1.0

# ── 10s 윈도우별 계산 ─────────────────────────────────────────────
actual_df = df[['timestamp', 'rnti', 'nprb', 'ul_bytes']].set_index('timestamp')

ts_list, f_byte_off_list, f_byte_on_list = [], [], []
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

    period_end = period + pd.Timedelta(WINDOW)
    grp_p = pred_df[(pred_df.index >= period) & (pred_df.index < period_end)]
    if grp_p.empty:
        weights = {ue: 1.0 / n_ues for ue in ues}
    else:
        avg_pred = grp_p.groupby('rnti')[['pred_snr', 'pred_bler']].mean()
        scores = {}
        for ue in ues:
            snr  = max(float(avg_pred.loc[ue, 'pred_snr'])  if ue in avg_pred.index else 1.0, 0.1)
            bler = float(avg_pred.loc[ue, 'pred_bler']) if ue in avg_pred.index else 0.0
            scores[ue] = (1.0 / snr) * (1.0 + bler * 5)
        total_score = sum(scores.values()) or 1.0
        weights = {ue: scores[ue] / total_score for ue in ues}

    dapp_nprb  = {ue: total_nprb * weights[ue] for ue in ues}
    dapp_bytes = {ue: dapp_nprb[ue] * eff[ue]  for ue in ues}
    off_bytes  = {ue: float(actual_bytes.get(ue, 0)) for ue in ues}

    f_off = jains([off_bytes[ue]  for ue in ues])
    f_on  = jains([dapp_bytes[ue] for ue in ues])

    ts_list.append(period)
    f_byte_off_list.append(f_off)
    f_byte_on_list.append(f_on)
    for ue in ues:
        ue_byte_off[ue].append(off_bytes[ue])
        ue_byte_on[ue].append(dapp_bytes[ue])

print(f"\n윈도우 수: {len(ts_list)}")

# ── 집계 ──────────────────────────────────────────────────────────
mean_off = np.nanmean(f_byte_off_list)
mean_on  = np.nanmean(f_byte_on_list)
ue_avg_off = {ue: np.mean(ue_byte_off[ue]) for ue in ues}
ue_avg_on  = {ue: np.mean(ue_byte_on[ue])  for ue in ues}

# ── 그래프 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Chronos TCN dApp ON vs OFF — Fairness Comparison\n"
             "(UE1: 21ab SNR=23.8 dB  |  UE2: f402 SNR=15.7 dB with packet loss)",
             fontsize=12, fontweight='bold')

# ── (1) Jain's Fairness 막대 ──────────────────────────────────────
ax = axes[0]
bars = ax.bar(['dApp OFF\n(OAI default)', 'dApp ON\n(Chronos TCN)'],
              [mean_off, mean_on],
              color=[COLOR_OFF, COLOR_ON], width=0.45, edgecolor='white', linewidth=1.2)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.set_title("Throughput Fairness", fontsize=11, fontweight='bold')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
for bar, val in zip(bars, [mean_off, mean_on]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
arrow_x = 0.72
ax.annotate('', xy=(arrow_x, mean_on - 0.001), xytext=(arrow_x, mean_off + 0.001),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(arrow_x + 0.05, (mean_off + mean_on)/2,
        f'+{mean_on-mean_off:.3f}', color='green', fontsize=10, va='center')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# ── (2) UE별 Throughput 막대 ─────────────────────────────────────
ax = axes[1]
ue_labels = [f'UE1\n(21ab)\nSNR={23.8}dB', f'UE2\n(f402)\nSNR={15.7}dB']
x = np.arange(2)
w = 0.3
b1 = ax.bar(x - w/2, [ue_avg_off[ue] for ue in ues], w,
            label='dApp OFF', color=COLOR_OFF, edgecolor='white')
b2 = ax.bar(x + w/2, [ue_avg_on[ue]  for ue in ues], w,
            label='dApp ON',  color=COLOR_ON,  edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(ue_labels, fontsize=10)
ax.set_ylabel("Avg UL Throughput (bytes/10s)", fontsize=11)
ax.set_title("Per-UE Throughput", fontsize=11, fontweight='bold')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# ── (3) 시간축 Fairness 추이 ─────────────────────────────────────
ax = axes[2]
ts = pd.Series(ts_list)
roll = 20   # 200s 이동평균
f_off_s = pd.Series(f_byte_off_list).rolling(roll, min_periods=1).mean()
f_on_s  = pd.Series(f_byte_on_list).rolling(roll, min_periods=1).mean()

ax.plot(ts, f_off_s, color=COLOR_OFF, linewidth=1.5, label='dApp OFF', alpha=0.9)
ax.plot(ts, f_on_s,  color=COLOR_ON,  linewidth=1.5, label='dApp ON',  alpha=0.9)
ax.fill_between(ts, f_off_s, f_on_s,
                where=(f_on_s >= f_off_s), alpha=0.15, color='green', label='dApp gain')
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.set_title(f"Fairness Over Time ({roll*10}s rolling avg)", fontsize=11, fontweight='bold')
ax.axhline(mean_off, color=COLOR_OFF, linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(mean_on,  color=COLOR_ON,  linestyle=':', linewidth=1, alpha=0.7)
ax.legend(fontsize=9)
ax.tick_params(axis='x', rotation=20)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d\n%H:%M'))
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dapp_compare.png', dpi=150, bbox_inches='tight')
print("\n저장 완료: dapp_compare.png")
