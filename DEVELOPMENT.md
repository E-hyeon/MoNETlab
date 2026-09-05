# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoNETlab is an O-RAN research testbed implementing AI-based and baseline RAN scheduling logic for per-UE fairness-aware PRB weighting. Three interchangeable "next-value estimators" feed the same weighting formula:

| Controller | Next-value estimation | Classification |
|---|---|---|
| `dapp_controller_chronos.py` | BigDL Chronos TCN forecast (`LOOKBACK=10`) | dApp (AI prediction) |
| `dapp_controller_ewma.py` | Exponentially weighted moving average, `alpha=0.3` (`LOOKBACK=10`) | dApp (lightweight statistical prediction) |
| `baseline_controller_reactive.py` | No prediction — reuses the just-measured current value | baseline (no AI prediction, **not** a dApp) |

`compute_weights` / `apply_weights` / `run` are identical across all three (copy-pasted intentionally) so `compare_dapp.py` / `plot_dapp_compare.py` can attribute fairness differences purely to the estimation method.

**IMPORTANT — no live actuation exists yet.** All three controllers only write to `/tmp/dapp_weights.json`; nothing in the OAI MAC scheduler or an E2 xApp currently reads that file. Fairness numbers from `compare_dapp.py` are computed **offline** by re-weighting already-collected KPI data (`nprb × weights` → estimated `bytes`), not by observing a live scheduler switch. Don't assume "10 min OAI / 10 min Chronos / 10 min Reactive" style live A/B runs are possible without first wiring a real actuation path.

## Key Commands

### KPI Collection
Two independent data paths exist — pick based on what you need:
```bash
# Path 1: tail gnb_live.log (stdout). UE stats blocks here are triggered by a simulated
# frame-count condition, and on WSL2 (softmodem runs slower than real-time) this ends up
# updating only every ~5-8s in wall-clock time. Fine for coarse/long sessions.
tail -f gnb_live.log | python3 collect_kpi.py        # writes kpi_baseline.csv
tail -f gnb_live.log | python3 collect_kpi_live.py   # writes kpi_live.csv

# Path 2 (recommended for anything time-variation-sensitive): poll nrMAC_stats.log directly.
# nrmac_stats_thread() (openair2/LAYER2/NR_MAC_gNB/main.c) overwrites this file every real
# 1 second via sleep(1) — the SAME per-UE stats content, but at genuinely 1Hz.
python3 collect_kpi_fast.py   # polls .../cmake_targets/ran_build/build/nrMAC_stats.log → kpi_fast.csv
```
This distinction matters a lot: a 30-minute `kpi_live.csv` session showed a weak UE's SNR pinned to a single value for >95% of samples even with a time-varying channel model configured, while the same channel sampled via `kpi_fast.csv` (1Hz) showed real swings of several dB second-to-second. If SNR/BLER looks suspiciously flat, check which log you're reading before assuming the channel model isn't working.

### Model Training
```bash
# Train Chronos TCN model from baseline data
conda run -n chronos python3 chronos_train.py    # full version with OpenVINO benchmark
conda run -n chronos python3 chronos_final.py    # streamlined version

# Retrain Chronos TCN from live data (with 50-epoch checkpoint support)
conda run -n chronos python3 chronos_retrain.py  # reads kpi_live.csv
```

### Running the dApp / Baseline Controllers
```bash
conda run -n chronos python3 dapp_controller_chronos.py   # Chronos TCN-based PRB scheduler
conda run -n chronos python3 dapp_controller_ewma.py       # EWMA(alpha=0.3)-based PRB scheduler
python3 baseline_controller_reactive.py                    # no-prediction baseline (plain pandas, no chronos env needed)
```

### Evaluation & Visualization
```bash
conda run -n chronos python3 eval_dapp.py         # Chronos TCN vs persistence MAE/latency
conda run -n chronos python3 compare_dapp.py      # OAI default vs Chronos/EWMA dApp vs Reactive Baseline, Jain's fairness
conda run -n chronos python3 plot_dapp_compare.py # generates dapp_compare.png (4-way)
conda run -n chronos python3 analyze_variability_segments.py  # splits the weak UE's session by SNR rolling-std, compares per-segment
```
`compare_dapp.py`/`plot_dapp_compare.py` auto-select the two UEs with the largest mean-SNR gap from `kpi_live.csv` (`pick_target_ues()`, min 50 samples) rather than hardcoding RNTIs — RNTI is reassigned every session.

### Testbed Setup
```bash
bash setup_namespaces.sh  # create ue1/ue2 network namespaces (once)
bash start_gnb.sh         # start OAI gNB (foreground, own terminal)
bash start_ue1.sh         # start UE1 (foreground, own terminal)
bash start_ue2.sh         # start UE2 (foreground, own terminal)
bash full_restart.sh      # (re)start UE tunnels, iperf3 (8 Mbps/UE UDP), and collect_kpi.py — does NOT touch gNB/UE processes
```
**Start gNB + UE1 + UE2 together, with no intermediate restart of only one UE.** The rfsimulator's per-UE channel model (`rfsimu_channel_ue0`/`rfsimu_channel_ue1`) is assigned by connection order at the socket level. Restarting a single UE while the gNB keeps running can leave both UEs bound to the same channel-model slot (observed: `rfsimu_channel_ue1` activated 3× on one run after a UE was individually restarted, while `ue0` only activated once) — after that, the intended SNR asymmetry silently disappears even though the config is correct. Verify a clean 1:1 assignment before trusting any collected data:
```bash
grep "activated" gnb_live.log   # want exactly one "rfsimu_channel_ue0" and one "rfsimu_channel_ue1" line
```
Also: `oai-cn5g` (AMF/SMF/UPF/ext-dn, docker compose) must be up before UEs attempt Registration — if the core was down when a UE's NAS Registration Request went unanswered, the UE does not auto-retry; it needs an explicit restart once the core is back (`cd oai-cn5g && docker compose up -d`, then restart UE1/UE2).

### Fairness Analysis
```bash
python3 fairness.py      # prints Jain's fairness index + throughput over kpi_live.csv
python3 plot_fairness.py # generates kpi_analysis_plot.png
```

## Architecture

### Data Flow
```
OAI gNB (rfsim)
  ├─► gnb_live.log (stdout, ~5-8s cadence on WSL2)
  │     └─► collect_kpi.py / collect_kpi_live.py → kpi_baseline.csv / kpi_live.csv
  └─► nrMAC_stats.log (genuinely 1Hz, overwritten in place)
        └─► collect_kpi_fast.py → kpi_fast.csv   (use this for time-variation analysis)

kpi_*.csv
  ├─► chronos_train.py / chronos_retrain.py → chronos_forecaster / scaler_chronos.pkl
  ├─► dapp_controller_chronos.py ─┐
  ├─► dapp_controller_ewma.py      ├─► /tmp/dapp_weights.json (per-RNTI weights)
  ├─► baseline_controller_reactive.py ─┘     └─► OAI gNB MAC scheduler (source patch required — NOT implemented; see note above)
  └─► compare_dapp.py / plot_dapp_compare.py → offline 4-way Jain's fairness comparison
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
| `dapp_controller_chronos.py` | TCN 예측 기반 PRB 가중치 컨트롤러 |
| `dapp_controller_ewma.py` | EWMA(alpha=0.3) 예측 기반 PRB 가중치 컨트롤러 |
| `baseline_controller_reactive.py` | 예측 없음 — 실측 현재값 그대로 사용 (baseline, dApp 아님) |
| `collect_kpi_fast.py` | `nrMAC_stats.log` 1Hz 폴링 수집기 |
| `analyze_variability_segments.py` | 약한 UE SNR rolling-std 기준 변동/안정 구간 분리 + 구간별 fairness 비교 |

### Fairness Weight Formula
```python
score[ue] = (1 / snr) * (1 + bler * 5)   # higher score → more PRB weight needed
weight[ue] = score[ue] / sum(scores)
```
Shared by all three controllers. Equal weights are applied when no estimate is available yet (Chronos/EWMA: first `LOOKBACK=10` steps per UE; Reactive Baseline: only the very first cycle).

### Model Artifacts
| File | Contents |
|---|---|
| `chronos_forecaster` | BigDL TCNForecaster model |
| `scaler_chronos.pkl` | `StandardScaler` fitted on training data |

### Python Environment
- Conda env `chronos` (Python 3.9) — `conda run -n chronos python3 <script>`
- Key deps: `bigdl-chronos`, `scikit-learn`, `joblib`, `pandas`
- `baseline_controller_reactive.py` and `collect_kpi_fast.py` only need plain `pandas` — no need for the `chronos` env.

### Infrastructure Stack
- **OAI gNB**: `openairinterface5g/` — config at `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.e2.ej.conf` (Band 78, 106 PRB, rfsim, E2 agent enabled)
- **5G Core**: `oai-cn5g/docker-compose.yaml` — UPF, AMF, SMF, etc. on `192.168.70.0/24`; external DN at `192.168.70.135`
- **Near-RT RIC**: `ric-dep/` — O-RAN SC RIC deployed via Kubernetes Helm (release prefix `r4`)
- **UE namespaces**: `ue1` / `ue2` with tunnels `oaitun_ue1` / `oaitun_ue2`
- **Traffic**: iperf3 UDP **8 Mbps**/UE (lowered from 20 Mbps — at 20 Mbps both UEs saturated the cell and hit RLC `SDU rejected, SDU buffer full` continuously, which masks any PRB-reallocation effect); ue1 on port 5201, ue2 on port 5202
- **Channel model**: `channelmod_rfsimu_ue_diff.conf`, `@include`d at the end of `gnb.e2.ej.conf` with `rfsimulator.options = ("chanmod")` enabled. Two `TDL_C` entries, `rfsimu_channel_ue0` (`noise_power_dB=-4`, good) and `rfsimu_channel_ue1` (`noise_power_dB=-2`, poor, `forgetfact=0.5` for time variation — see below). This `@include` + `chanmod` option did **not** exist before commit `89da3c0` (2026-06-30); any `kpi_*.csv` timestamped before that used the default symmetric AWGN channel (no per-UE asymmetry), which explains near-identical SNR between UEs in older captures.
  - Only 7 keys are recognized by this OAI build's config parser: `model_name, type, ploss_dB, noise_power_dB, forgetfact, offset, ds_tdl` (`openair1/SIMULATION/TOOLS/sim.h` `CHANNELMOD_PARAMS_DESC`, `random_channel.c`). **`speed_min`/`speed_max`/Doppler-by-speed keys do not exist** — unrecognized keys are silently ignored, not errored. `forgetfact` (0=new channel every call, 1=frozen) is the only real time-variation knob for `TDL_x` models; Doppler (`maxDoppler`) is hardcoded per preset `type` (e.g. `Rayleigh1_800`), not user-settable for `TDL_C`.

### `oai-channel-prediction/` Subproject
Scaffold for packaging the ML pipeline as a containerised O-RAN xApp/dApp. Mirrors the root controllers (`dapp_controller_chronos.py`, `dapp_controller_ewma.py`, `baseline_controller_reactive.py` under `src/`, entrypoint reads `kpi_baseline.csv` instead of `kpi_live.csv`). Not yet implemented as an actual xApp/dApp package.
