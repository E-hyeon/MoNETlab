# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoNETlab is an O-RAN research testbed implementing an AI-based RAN scheduling **dApp** (distributed Application). The system predicts per-UE channel KPIs using LSTM/TCN models and computes fairness-aware PRB scheduling weights in real time.

## Key Commands

### KPI Collection (run against live gNB log)
```bash
# Collect baseline data (writes kpi_baseline.csv)
tail -f gnb_live.log | python3 collect_kpi.py

# Collect live data (writes kpi_live.csv)
tail -f gnb_live.log | python3 collect_kpi_live.py
```

### Model Training
```bash
# Build kpi_combined.csv (required by live/retrain scripts)
head -1 kpi_baseline.csv > kpi_combined.csv
tail -n +2 kpi_baseline.csv >> kpi_combined.csv
tail -n +2 kpi_live.csv    >> kpi_combined.csv

# Train global attention-LSTM from baseline data
python3 train_rnn_global.py

# Retrain/fine-tune global model from combined data
python3 train_rnn_global_live.py   # reads kpi_combined.csv

# Retrain legacy per-UE models from combined data
python3 train_rnn_live.py          # writes model_ue_<rnti>.pth per UE

# Train BigDL Chronos TCN model (with OpenVINO benchmark)
python3 chronos_train.py    # full version with OpenVINO speed comparison
python3 chronos_final.py    # streamlined version
```

### Running the dApp Controller (pick one)
```bash
python3 dapp_controller_global.py   # global attention-LSTM (recommended)
python3 dapp_controller_chronos.py  # BigDL TCN variant
python3 dapp_controller.py          # legacy per-UE LSTM
```

### Testbed Setup
```bash
bash full_restart.sh        # restart UE tunnels, iperf3, and KPI collector
bash setup_ue_traffic.sh    # configure routing/iperf3 only (no KPI restart)
```

### Fairness Analysis
```bash
python3 fairness.py      # prints Jain's fairness index + throughput over kpi_live.csv
python3 plot_fairness.py # generates kpi_analysis_plot.png
```

## Architecture

### Data Flow
```
OAI gNB log (gnb_live.log)
  └─► collect_kpi.py / collect_kpi_live.py
        └─► kpi_baseline.csv / kpi_live.csv
              └─► train_rnn_global.py  →  model artifacts (*.pth, *.pkl)
              └─► dapp_controller_global.py
                    └─► /tmp/dapp_weights.json  (scheduling weights per UE RNTI)
```

### KPI Features
All models share the same 6 features: `snr`, `bler`, `nprb`, `mcs_ul`, `ul_bytes`, `dl_bytes`.

### Model Variants

| Script | Model | Notes |
|---|---|---|
| `train_rnn_global.py` | `GlobalChannelRNN` (LSTM + attention) | Single model for all UEs; UE identity via embedding |
| `train_rnn_global_live.py` | Same architecture | Fine-tunes from existing `model_global.pth`; expands `ue_encoder.pkl` for new UEs |
| `chronos_train.py` | BigDL `TCNForecaster` | Full version: MAE/MSE eval + OpenVINO speed benchmark |
| `chronos_final.py` | BigDL `TCNForecaster` | Streamlined version; same output as `chronos_train.py` |
| `train_rnn.py` | `ChannelRNN` (plain LSTM) | Legacy per-UE baseline training from `kpi_baseline.csv` |
| `train_rnn_live.py` | `ChannelRNN` (plain LSTM) | Legacy per-UE live retrain from `kpi_combined.csv` |

### `GlobalChannelRNN` Architecture
- Input: KPI sequence `(B, T=10, 6)` concatenated with UE embedding `(B, T, 8)`
- LSTM: 2 layers, hidden=128, dropout=0.2
- Attention: linear scoring over time steps → weighted sum context vector
- Output: next-step KPI prediction `(B, 6)`
- Unknown UEs (not seen at training) map to `padding_idx = n_ues` → zero embedding vector

### Fairness Weight Formula
```python
score[ue] = (1 / snr) * (1 + bler * 5)   # higher score → more PRB weight needed
weight[ue] = score[ue] / sum(scores)
```
Equal weights are applied when no predictions are available yet (first `seq_length=10` steps).

### Model Artifacts
| File | Contents |
|---|---|
| `model_global.pth` | GlobalChannelRNN state dict |
| `scaler_global.pkl` | `MinMaxScaler` fitted on training data |
| `features_global.pkl` | Ordered feature name list |
| `ue_encoder.pkl` | `LabelEncoder` mapping RNTI string → int index |
| `chronos_forecaster` | BigDL TCN model (single file in repo) |
| `scaler_chronos.pkl` | `StandardScaler` for Chronos |

### Python Environment
- Virtual env at `.venv/` (Python 3.12) — activate with `source .venv/bin/activate`
- Key deps: `torch`, `scikit-learn`, `joblib`, `pandas`, `bigdl-chronos` (for Chronos TCN scripts)

### Infrastructure Stack
- **OAI gNB**: `openairinterface5g/` — config at `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.e2.ej.conf` (Band 78, 106 PRB, rfsim, E2 agent enabled)
- **5G Core**: `oai-cn5g/docker-compose.yaml` — UPF, AMF, SMF, etc. on `192.168.70.0/24`; external DN at `192.168.70.135`
- **Near-RT RIC**: `ric-dep/` — O-RAN SC RIC deployed via Kubernetes Helm (release prefix `r4`)
- **UE namespaces**: `ue1` / `ue2` with tunnels `oaitun_ue1` / `oaitun_ue2`
- **Traffic**: iperf3 UDP 20 Mbps; ue1 on port 5201, ue2 on port 5202

### `oai-channel-prediction/` Subproject
Empty scaffold (`src/`, `oai-patch/`, `docker/`, `k8s/`, `tests/`, `docs/`) for packaging the ML pipeline as a containerised O-RAN xApp/dApp. Not yet implemented.
