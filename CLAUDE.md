# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoNETlab is an O-RAN research testbed implementing an AI-based RAN scheduling **dApp** (distributed Application). The system predicts per-UE channel KPIs using a BigDL Chronos TCN model and computes fairness-aware PRB scheduling weights in real time.

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
# Train Chronos TCN model from baseline data
conda run -n chronos python3 chronos_train.py    # full version with OpenVINO benchmark
conda run -n chronos python3 chronos_final.py    # streamlined version

# Retrain Chronos TCN from live data (with 50-epoch checkpoint support)
conda run -n chronos python3 chronos_retrain.py  # reads kpi_live.csv
```

### Running the dApp Controller
```bash
conda run -n chronos python3 dapp_controller_chronos.py  # Chronos TCN-based PRB scheduler
```

### Evaluation & Visualization
```bash
conda run -n chronos python3 eval_dapp.py         # Chronos TCN vs persistence MAE/latency
conda run -n chronos python3 compare_dapp.py      # dApp ON vs OFF Jain's fairness comparison
conda run -n chronos python3 plot_dapp_compare.py # generates dapp_compare.png
```

### Testbed Setup
```bash
bash setup_namespaces.sh  # create ue1/ue2 network namespaces
bash start_gnb.sh         # start OAI gNB
bash start_ue1.sh         # start UE1
bash start_ue2.sh         # start UE2
bash full_restart.sh      # restart UE tunnels, iperf3, and KPI collector
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
              └─► chronos_train.py / chronos_retrain.py  →  chronos_forecaster / scaler_chronos.pkl
              └─► dapp_controller_chronos.py
                    └─► /tmp/dapp_weights.json  (scheduling weights per UE RNTI)
                          └─► OAI gNB MAC scheduler (source patch required)
```

### KPI Features
All models share the same 6 features: `snr`, `bler`, `nprb`, `mcs_ul`, `ul_bytes`, `dl_bytes`.

### Model Scripts

| Script | Description |
|---|---|
| `chronos_train.py` | BigDL `TCNForecaster` 학습 (full: MAE/MSE eval + OpenVINO benchmark) |
| `chronos_final.py` | BigDL `TCNForecaster` 학습 (streamlined) |
| `chronos_retrain.py` | kpi_live.csv로 재학습; 50 epoch마다 체크포인트 저장 |
| `chronos_live.py` | 실시간 스트리밍 재학습 |
| `dapp_controller_chronos.py` | 예측 기반 PRB 가중치 컨트롤러 |

### Fairness Weight Formula
```python
score[ue] = (1 / snr) * (1 + bler * 5)   # higher score → more PRB weight needed
weight[ue] = score[ue] / sum(scores)
```
Equal weights are applied when no predictions are available yet (first `seq_length=10` steps).

### Model Artifacts
| File | Contents |
|---|---|
| `chronos_forecaster` | BigDL TCNForecaster model |
| `scaler_chronos.pkl` | `StandardScaler` fitted on training data |

### Python Environment
- Conda env `chronos` (Python 3.9) — `conda run -n chronos python3 <script>`
- Key deps: `bigdl-chronos`, `scikit-learn`, `joblib`, `pandas`

### Infrastructure Stack
- **OAI gNB**: `openairinterface5g/` — config at `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.e2.ej.conf` (Band 78, 106 PRB, rfsim, E2 agent enabled)
- **5G Core**: `oai-cn5g/docker-compose.yaml` — UPF, AMF, SMF, etc. on `192.168.70.0/24`; external DN at `192.168.70.135`
- **Near-RT RIC**: `ric-dep/` — O-RAN SC RIC deployed via Kubernetes Helm (release prefix `r4`)
- **UE namespaces**: `ue1` / `ue2` with tunnels `oaitun_ue1` / `oaitun_ue2`
- **Traffic**: iperf3 UDP 20 Mbps; ue1 on port 5201, ue2 on port 5202

### `oai-channel-prediction/` Subproject
Scaffold for packaging the ML pipeline as a containerised O-RAN xApp/dApp. Not yet implemented.
