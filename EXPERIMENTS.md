# Experiment Log

## Overview

9가지 모델 아키텍처 × 3 입력 길이 × 3 출력 길이 = **81개 실험**의 동일 조건 비교.

CSV 기반 30분 간격 태양풍 시계열 → ap30 회귀 예측.

데이터셋 상세: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## Common Settings

모든 81개 실험에서 공유하는 설정 (base.yaml + local.yaml).

| 항목 | 값 |
|------|-----|
| Optimizer | Adam (lr=2e-4, weight_decay=0.0) |
| LR Scheduler | ReduceOnPlateau (factor=0.5, patience=5) |
| Loss | SolarWindWeightedLoss (multi_tier, base=MSE) |
| Early Stopping | patience=10, min_delta=0.0 |
| Max Epochs | 30 |
| Batch Size | 128 |
| Gradient Clipping | max_norm=1.0 |
| d_model | 128 |
| Dropout | 0.1 (모델별 기본값) |
| Dataset Mode | Table (Parquet + index CSV) |
| Device | CUDA |
| Augmentation | 없음 (gaussian_noise_std=0.0) |

---

## Input Variables (22개)

hp30 제외. ap30은 입력과 타겟 모두에 포함.

| # | 변수 그룹 | 변수명 | 정규화 | GNN 노드 |
|---|----------|--------|--------|----------|
| 1-3 | 태양풍 속도 (V) | v_avg, v_min, v_max | log_zscore | v |
| 4-6 | 양성자 밀도 (Np) | np_avg, np_min, np_max | log_zscore | np |
| 7-9 | 양성자 온도 (T) | t_avg, t_min, t_max | log_zscore | t |
| 10-12 | IMF Bx | bx_avg, bx_min, bx_max | zscore | bx |
| 13-15 | IMF By (GSM) | by_avg, by_min, by_max | zscore | by |
| 16-18 | IMF Bz (GSM) | bz_avg, bz_min, bz_max | zscore | bz |
| 19-21 | IMF 크기 (Bt) | bt_avg, bt_min, bt_max | log_zscore | bt |
| 22 | ap30 지수 | ap30 | log1p_zscore | ap30 |

Target: ap30 (log1p_zscore)

---

## Models (9종)

| # | Config 접미사 | model_type | 설명 |
|---|-------------|-----------|------|
| 1 | (없음) | `linear` | Linear Encoder — MLP baseline |
| 2 | `_transformer` | `transformer` | Transformer Encoder |
| 3 | `_tcn` | `tcn` | Temporal Convolutional Network |
| 4 | `_patchtst` | `patchtst` | PatchTST (patch=16, stride=8) |
| 5 | `_timesnet` | `timesnet` | TimesNet (FFT + 2D Inception Conv) |
| 6 | `_gnn_transformer` | `gnn` (transformer) | GNN (8-node) + Transformer |
| 7 | `_gnn_tcn` | `gnn` (tcn) | GNN + TCN |
| 8 | `_gnn_bilstm` | `gnn` (bilstm) | GNN + BiLSTM |
| 9 | `_gnn_patchtst` | `gnn` (patch_transformer) | GNN + PatchTransformer |

GNN 노드: 7개 물리 변수 그룹 + ap30 = **8 노드** (config 기반 동적 구성)

---

## Input/Output Matrix

| 입력 \ 출력 | out6h (12ts) | out12h (24ts) | out24h (48ts) |
|-----------|-------------|-------------|-------------|
| in1d (48ts) | `in1d_out6h` | `in1d_out12h` | `in1d_out24h` |
| in2d (96ts) | `in2d_out6h` | `in2d_out12h` | `in2d_out24h` |
| in3d (144ts) | `in3d_out6h` | `in3d_out12h` | `in3d_out24h` |

---

## Experiment Results

### 6h 예측

| 모델 \ 입력 | in1d | in2d | in3d |
|------------|------|------|------|
| Linear (baseline) | | | |
| Transformer | | | |
| TCN | | | |
| PatchTST | | | |
| TimesNet | | | |
| GNN+Transformer | | | |
| GNN+TCN | | | |
| GNN+BiLSTM | | | |
| GNN+PatchTST | | | |

### 12h 예측

| 모델 \ 입력 | in1d | in2d | in3d |
|------------|------|------|------|
| Linear (baseline) | | | |
| Transformer | | | |
| TCN | | | |
| PatchTST | | | |
| TimesNet | | | |
| GNN+Transformer | | | |
| GNN+TCN | | | |
| GNN+BiLSTM | | | |
| GNN+PatchTST | | | |

### 24h 예측

| 모델 \ 입력 | in1d | in2d | in3d |
|------------|------|------|------|
| Linear (baseline) | | | |
| Transformer | | | |
| TCN | | | |
| PatchTST | | | |
| TimesNet | | | |
| GNN+Transformer | | | |
| GNN+TCN | | | |
| GNN+BiLSTM | | | |
| GNN+PatchTST | | | |

---

## Changelog

| 날짜 | 내용 |
|------|------|
| 2025-04-08 | 실험 매트릭스 초기화. 9모델 × 9입출력 = 81개 동일 파라미터 비교 |
| 2025-04-08 | hp30 입력 제외 (23→22변수). GNN 노드 그룹 동적화 (config 기반 검증) |
| 2025-04-08 | out6h 추가 (6h/12h/24h). 네이밍 통일 (접미사 없음=linear baseline) |
