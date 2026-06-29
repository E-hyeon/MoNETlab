# MoNETlab — AI-based O-RAN RAN Scheduling dApp

O-RAN 연구 테스트베드에서 BigDL Chronos TCN 채널 예측 모델을 활용한 공정성 인식 PRB 스케줄링 dApp 구현.

## Overview

OAI gNB 로그에서 실시간으로 UE별 KPI(SNR, BLER, nPRB 등)를 수집하고, TCN 모델로 다음 스텝을 예측하여 채널 상태가 나쁜 UE에 더 많은 PRB를 배분합니다.

```
OAI gNB (rfsim)
    │
    ▼
collect_kpi_live.py  ──►  kpi_live.csv
    │
    ▼
chronos_retrain.py   ──►  chronos_forecaster
    │
    ▼
dapp_controller_chronos.py  ──►  /tmp/dapp_weights.json  ──►  OAI gNB MAC scheduler
                                                               (source patch required)
```

## Key Result

| | dApp OFF (OAI default) | dApp ON (Chronos TCN) |
|--|--|--|
| Jain's Fairness Index | 0.8670 | **0.9617** |
| Fairness gap 감소 | — | **71.2%** |
| Inference latency | — | 평균 1.21 ms |

UE1 (SNR 23.8 dB) vs UE2 (SNR 15.7 dB, packet loss) 환경에서 측정.

## System Architecture

- **OAI gNB**: Band 78, 106 PRB, rfsimulator, E2 agent
- **5G Core**: OAI CN5G (UPF/AMF/SMF) on `192.168.70.0/24`
- **Near-RT RIC**: O-RAN SC RIC (Kubernetes, Helm release `r4`)
- **UE**: 2개 UE 네임스페이스 (`ue1`, `ue2`), iperf3 UDP 20 Mbps

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
```bash
bash setup_namespaces.sh   # UE 네임스페이스 생성
bash start_gnb.sh          # OAI gNB 시작
bash start_ue1.sh          # UE1 접속
bash start_ue2.sh          # UE2 접속
```

### 2. KPI 수집
```bash
# 터미널 1
tail -f gnb_live.log | python3 collect_kpi.py        # baseline

# 터미널 2
tail -f gnb_live.log | python3 collect_kpi_live.py   # live
```

### 3. 모델 학습
```bash
# 최초 학습
conda run -n chronos python3 chronos_train.py

# 라이브 데이터로 재학습 (중단 후 재개 가능)
conda run -n chronos python3 chronos_retrain.py
```

### 4. dApp 실행
```bash
conda run -n chronos python3 dapp_controller_chronos.py
# → /tmp/dapp_weights.json 에 UE별 PRB 가중치 실시간 업데이트
```

### 5. 성능 평가
```bash
conda run -n chronos python3 compare_dapp.py      # fairness 비교 수치
conda run -n chronos python3 plot_dapp_compare.py # dapp_compare.png 생성
```

## File Structure

```
├── collect_kpi.py              # gNB 로그 → kpi_baseline.csv
├── collect_kpi_live.py         # gNB 로그 → kpi_live.csv
├── chronos_train.py            # TCN 학습 (full)
├── chronos_final.py            # TCN 학습 (streamlined)
├── chronos_retrain.py          # TCN 재학습 (체크포인트 지원)
├── chronos_live.py             # 실시간 스트리밍 재학습
├── dapp_controller_chronos.py  # PRB 가중치 컨트롤러
├── eval_dapp.py                # 예측 정확도 평가
├── compare_dapp.py             # dApp ON/OFF fairness 비교
├── plot_dapp_compare.py        # 비교 그래프 생성
├── fairness.py                 # Jain's fairness index 출력
├── full_restart.sh             # 테스트베드 재시작
├── setup_namespaces.sh         # UE 네임스페이스 설정
├── start_gnb.sh / start_ue*.sh # gNB / UE 시작 스크립트
└── openairinterface5g/         # OAI 소스 및 설정
    └── targets/PROJECTS/GENERIC-NR-5GC/CONF/
        ├── gnb.e2.ej.conf
        ├── ue1.conf / ue2.conf
        └── channelmod_rfsimu_ue_diff.conf
```

## KPI Features

모든 모델이 공유하는 6개 피처: `snr`, `bler`, `nprb`, `mcs_ul`, `ul_bytes`, `dl_bytes`

## Fairness Weight Formula

```python
score[ue] = (1 / snr) * (1 + bler * 5)   # 채널 상태 나쁠수록 높은 score
weight[ue] = score[ue] / sum(scores)       # PRB 배분 가중치
```
