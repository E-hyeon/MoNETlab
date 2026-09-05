# MoNETlab — AI-based O-RAN RAN Scheduling dApp

O-RAN 연구 테스트베드에서 채널 예측/반응형 기법을 활용한 공정성 인식 PRB 스케줄링 dApp과, 이를 실측 데이터로 비교·검증하는 파이프라인 구현.

## Overview

OAI gNB 로그에서 실시간으로 UE별 KPI(SNR, BLER, nPRB 등)를 수집하고, 네 가지 방식으로 "다음 스텝"을 추정해 채널 상태가 나쁜 UE에 더 많은 PRB 가중치를 배분합니다.

| 방식 | 다음 값 추정 방법 | 분류 |
|---|---|---|
| **Chronos dApp** | BigDL Chronos TCN 모델 예측 (LOOKBACK=10) | dApp (AI 예측, 가장 무거움) |
| **MLP dApp** | 얕은 MLP(은닉층 1개, 뉴런 12개) 예측 (LOOKBACK=10) | dApp (경량 AI 예측) |
| **EWMA dApp** | 지수가중이동평균(alpha=0.3)으로 다음 값 추정 | dApp (경량 통계적 예측) |
| **Reactive Baseline** | 예측 없이 방금 측정한 현재값을 그대로 사용 | baseline (AI 예측 없음, dApp 아님) |

네 방식 모두 `compute_weights`/`apply_weights` 로직은 동일하며(공정 비교를 위해), "다음 값을 어떻게 추정하는가"만 다릅니다.

```
OAI gNB (rfsim)
    │
    ├─► gnb_live.log (stdout, ~5-8s 간격*) ──► collect_kpi.py / collect_kpi_live.py ──► kpi_baseline.csv / kpi_live.csv
    │
    └─► nrMAC_stats.log (1초마다 덮어씀)   ──► collect_kpi_fast.py (1Hz 폴링)        ──► kpi_fast.csv  (권장: 실제 채널 변동 포착용)

kpi_*.csv
    ├─► chronos_train.py / chronos_retrain.py        ──► chronos_forecaster / scaler_chronos.pkl
    ├─► mlp_train.py                                 ──► mlp_forecaster.pkl / scaler_mlp.pkl
    ├─► dapp_controller_chronos.py                   ─┐
    ├─► dapp_controller_mlp.py                        │
    ├─► dapp_controller_ewma.py                       ├─► /tmp/dapp_weights.json  (scheduling weights per UE RNTI)
    ├─► baseline_controller_reactive.py               ─┘        └─► OAI gNB MAC scheduler (source patch required, 미구현)
    └─► compare_dapp.py / plot_dapp_compare.py  (5자 오프라인 비교: OAI 기본/Chronos/MLP/EWMA/Reactive)
```
\* `gnb_live.log`의 UE stats 블록은 시뮬레이션 프레임 카운트 기반 트리거라, WSL2에서 실시간보다 느리게 도는 소프트모뎀 환경에서는 실제로 5~8초에 한 번만 갱신됩니다. 시간에 따른 채널 변동을 제대로 보려면 `collect_kpi_fast.py`로 `nrMAC_stats.log`를 직접 폴링하세요.

## Key Result

실측 테스트베드(비대칭 + 시간변화 채널, 8 Mbps UDP × 2 UE, 1Hz 폴링 `kpi_fast.csv`, 5분)에서 측정한 Jain's Fairness Index (throughput 기준):

| 방식 | Jain's Fairness Index | 향상 | gap 감소 | 추론 지연(평균) |
|---|---|---|---|---|
| OAI 기본 | 0.9838 | — | — | — |
| Chronos dApp | 0.9934 | +0.0096 | 59.2% | 1.7903 ms |
| MLP dApp | 0.9949 | +0.0111 | 68.6% | 0.1903 ms |
| Reactive Baseline | 0.9952 | +0.0114 | 70.2% | 0.0028 ms |
| **EWMA dApp** | **0.9961** | **+0.0123** | **75.8%** | 0.0383 ms |

UE1 (SNR ≈23.5 dB) vs UE2 (SNR ≈20.0 dB, `forgetfact=0.5`로 시간에 따라 변동하는 채널) 환경에서 측정. `compare_dapp.py`가 두 UE 중 SNR 차이가 가장 큰 쌍을 자동으로 선택합니다. 추론 지연은 `eval_latency.py`로 각 컨트롤러의 `predict_next()`를 200회 반복 호출해 측정 (같은 UE, 동일 워밍업). MLP는 Chronos TCN보다 fairness는 비슷하거나 더 좋으면서 추론은 **약 9.4배** 빠릅니다 — 모델 복잡도(TCN > MLP > EWMA > 없음)와 지연시간이 정확히 비례합니다.

## System Architecture

- **OAI gNB**: Band 78, 106 PRB, rfsimulator, E2 agent
- **5G Core**: OAI CN5G (UPF/AMF/SMF) on `192.168.70.0/24`
- **Near-RT RIC**: O-RAN SC RIC (Kubernetes, Helm release `r4`)
- **UE**: 2개 UE 네임스페이스 (`ue1`, `ue2`), iperf3 UDP 8 Mbps/UE
- **채널 모델**: `channelmod_rfsimu_ue_diff.conf`에서 UE0/UE1에 서로 다른 TDL_C 파라미터 적용 (비대칭 SNR). `forgetfact`를 0.99(정적)→0.5(시간변화)로 낮추면 실제로 흔들리는 채널이 됨 — 단 `speed_min`/`speed_max` 같은 도플러/속도 파라미터는 이 OAI 버전에 없는 키이므로 사용 불가 (지원 파라미터: `model_name, type, ploss_dB, noise_power_dB, forgetfact, offset, ds_tdl`).

## Prerequisites

- OAI gNB + 5G Core 실행 환경
- Conda with `chronos` env

```bash
conda create -n chronos python=3.9
conda activate chronos
pip install bigdl-chronos scikit-learn pandas joblib matplotlib
```

## Quick Start

### 1. 테스트베드 시작
채널모델(UE0/UE1)이 매 연결마다 정확히 1:1로 배정되도록, gNB와 두 UE를 **중간 재시작 없이 한 번에** 띄우는 것을 권장합니다.
```bash
bash setup_namespaces.sh   # UE 네임스페이스 생성 (1회)

# 터미널 A
bash start_gnb.sh          # OAI gNB 시작 → gnb_live.log에 NGAP_REGISTER_GNB_CNF 뜨면 아래 진행

# 터미널 B, C를 거의 동시에
bash start_ue1.sh          # UE1 접속
bash start_ue2.sh          # UE2 접속
# → gnb_live.log에서 "rfsimu_channel_ue0"/"rfsimu_channel_ue1" activated가 각각 1번씩만 찍히는지 확인
```

### 2. 트래픽 + KPI 수집 시작
```bash
bash full_restart.sh   # iperf3 UDP 8Mbps/UE 트래픽 + collect_kpi.py(→kpi_baseline.csv) 시작
```
채널 변동을 정확히 보려면(권장) `nrMAC_stats.log`를 1초 간격으로 직접 폴링하는 fast 콜렉터를 병행 실행:
```bash
python3 collect_kpi_fast.py   # → kpi_fast.csv (1Hz)
```
`collect_kpi.py`/`collect_kpi_live.py`가 보는 `gnb_live.log`의 UE stats 블록은 시뮬레이션 프레임 카운트 기반이라 WSL2 환경에서 실제로는 5~8초에 한 번만 갱신됩니다 — 짧은 시간 단위의 채널 변동 분석에는 `collect_kpi_fast.py` 결과(`kpi_fast.csv`)를 쓰세요.

### 3. 모델 학습
```bash
# Chronos TCN — 최초 학습
conda run -n chronos python3 chronos_train.py

# Chronos TCN — 라이브 데이터로 재학습 (중단 후 재개 가능)
conda run -n chronos python3 chronos_retrain.py

# MLP(은닉층 1개, 뉴런 12개) — kpi_live.csv로 학습
python3 mlp_train.py
```

### 4. dApp / Baseline 컨트롤러 실행
```bash
conda run -n chronos python3 dapp_controller_chronos.py       # TCN 예측 기반
python3 dapp_controller_mlp.py                                 # 얕은 MLP 예측 기반 (chronos env 불필요)
conda run -n chronos python3 dapp_controller_ewma.py          # EWMA(alpha=0.3) 예측 기반
python3 baseline_controller_reactive.py                       # 예측 없음 (Reactive Baseline, AI 미사용)
# → 넷 다 /tmp/dapp_weights.json 에 UE별 PRB 가중치 실시간 업데이트
```

### 5. 성능 평가 (OAI 기본 / Chronos dApp / MLP dApp / EWMA dApp / Reactive Baseline 5자 비교)
```bash
conda run -n chronos python3 compare_dapp.py      # fairness 비교 수치 (SNR 차이 최대인 UE 쌍 자동 선택)
conda run -n chronos python3 plot_dapp_compare.py # dapp_compare.png 생성
conda run -n chronos python3 eval_latency.py      # predict_next() 추론 지연시간 비교
```

### 6. (선택) SNR/BLER 변동성 구간 분석
```bash
conda run -n chronos python3 analyze_variability_segments.py
# 약한 UE의 SNR rolling std로 변동 구간/안정 구간을 나눠 구간별 fairness를 비교
```

## File Structure

```
├── collect_kpi.py                    # gnb_live.log(stdout) → kpi_baseline.csv (~5-8s 간격)
├── collect_kpi_live.py               # gnb_live.log(stdout) → kpi_live.csv (~5-8s 간격)
├── collect_kpi_fast.py               # nrMAC_stats.log 1Hz 폴링 → kpi_fast.csv (권장)
├── chronos_train.py                  # TCN 학습 (full)
├── chronos_final.py                  # TCN 학습 (streamlined)
├── chronos_retrain.py                # TCN 재학습 (체크포인트 지원)
├── chronos_live.py                   # 실시간 스트리밍 재학습
├── mlp_train.py                      # 얕은 MLP(은닉층 1개, 뉴런 12개) 학습
├── dapp_controller_chronos.py        # PRB 가중치 컨트롤러 — TCN 예측
├── dapp_controller_mlp.py            # PRB 가중치 컨트롤러 — 얕은 MLP 예측
├── dapp_controller_ewma.py           # PRB 가중치 컨트롤러 — EWMA(alpha=0.3) 예측
├── baseline_controller_reactive.py   # PRB 가중치 컨트롤러 — 예측 없음 (Reactive Baseline)
├── eval_dapp.py                      # 예측 정확도 평가
├── eval_latency.py                   # 컨트롤러별 predict_next() 추론 지연시간 비교
├── compare_dapp.py                   # OAI 기본/Chronos/MLP/EWMA/Reactive 5자 fairness 비교
├── plot_dapp_compare.py              # 비교 그래프 생성 (dapp_compare.png)
├── analyze_variability_segments.py   # SNR/BLER 변동 구간 vs 안정 구간 분석
├── fairness.py                       # Jain's fairness index 출력
├── full_restart.sh                   # 트래픽/KPI 수집 재시작 (gNB/UE는 유지)
├── setup_namespaces.sh               # UE 네임스페이스 설정
├── start_gnb.sh / start_ue*.sh       # gNB / UE 시작 스크립트
└── openairinterface5g/               # OAI 소스 및 설정
    └── targets/PROJECTS/GENERIC-NR-5GC/CONF/
        ├── gnb.e2.ej.conf
        ├── ue1.conf / ue2.conf
        └── channelmod_rfsimu_ue_diff.conf   # UE0(양호)/UE1(열악) 비대칭 + forgetfact 시간변화 설정
```

## KPI Features

모든 모델이 공유하는 6개 피처: `snr`, `bler`, `nprb`, `mcs_ul`, `ul_bytes`, `dl_bytes`

## Fairness Weight Formula

`dapp_controller_chronos.py` / `dapp_controller_mlp.py` / `dapp_controller_ewma.py` / `baseline_controller_reactive.py` 네 컨트롤러가 모두 공유하는 산식입니다 (입력값 — 예측 SNR/BLER vs 실측 SNR/BLER — 만 다름).

```python
score[ue] = (1 / snr) * (1 + bler * 5)   # 채널 상태 나쁠수록 높은 score
weight[ue] = score[ue] / sum(scores)       # PRB 배분 가중치
```
