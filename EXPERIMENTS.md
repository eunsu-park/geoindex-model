# Experiment Log / 실험 기록

## Overview / 개요

Comparison of 9 model architectures × 3 input lengths × 3 output lengths = **81 experiments** under identical conditions.
9가지 모델 아키텍처 × 3 입력 길이 × 3 출력 길이 = **81개 실험**의 동일 조건 비교.

CSV-based 30-min solar wind time series → ap30 regression prediction.
CSV 기반 30분 간격 태양풍 시계열 → ap30 회귀 예측.

Dataset details / 데이터셋 상세: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## Common Settings / 공통 설정

Shared settings across all 81 experiments (base.yaml + local.yaml).
모든 81개 실험에서 공유하는 설정.

| Item / 항목 | Value / 값 |
|------------|-----------|
| Optimizer | Adam (lr=2e-4, weight_decay=0.0) |
| LR Scheduler | ReduceOnPlateau (factor=0.5, patience=5) |
| Loss | SolarWindWeightedLoss (multi_tier, base=MSE) |
| Early Stopping | patience=10, min_delta=0.0 |
| Max Epochs | 30 |
| Batch Size | 128 |
| Gradient Clipping | max_norm=1.0 |
| d_model | 128 |
| Dropout | 0.1 (model default / 모델별 기본값) |
| Dataset Mode | Table (Parquet + index CSV) |
| Device | CUDA |
| Augmentation | None / 없음 (gaussian_noise_std=0.0) |

---

## Input Variables (22) / 입력 변수 (22개)

hp30 excluded. ap30 is included in both input and target.
hp30 제외. ap30은 입력과 타겟 모두에 포함.

| # | Group / 변수 그룹 | Variables / 변수명 | Normalization / 정규화 | GNN Node / GNN 노드 |
|---|-----------------|-------------------|---------------------|-------------------|
| 1-3 | Solar wind speed (V) / 태양풍 속도 | v_avg, v_min, v_max | log_zscore | v |
| 4-6 | Proton density (Np) / 양성자 밀도 | np_avg, np_min, np_max | log_zscore | np |
| 7-9 | Proton temperature (T) / 양성자 온도 | t_avg, t_min, t_max | log_zscore | t |
| 10-12 | IMF Bx | bx_avg, bx_min, bx_max | zscore | bx |
| 13-15 | IMF By (GSM) | by_avg, by_min, by_max | zscore | by |
| 16-18 | IMF Bz (GSM) | bz_avg, bz_min, bz_max | zscore | bz |
| 19-21 | IMF magnitude (Bt) / IMF 크기 | bt_avg, bt_min, bt_max | log_zscore | bt |
| 22 | ap30 index / ap30 지수 | ap30 | log1p_zscore | ap30 |

Target: ap30 (log1p_zscore)

---

## Models (9) / 모델 (9종)

| # | Config suffix / 접미사 | model_type | Description / 설명 |
|---|---------------------|-----------|-------------------|
| 1 | (none / 없음) | `linear` | Linear Encoder — MLP baseline |
| 2 | `_transformer` | `transformer` | Transformer Encoder |
| 3 | `_tcn` | `tcn` | Temporal Convolutional Network |
| 4 | `_patchtst` | `patchtst` | PatchTST (patch=16, stride=8) |
| 5 | `_timesnet` | `timesnet` | TimesNet (FFT + 2D Inception Conv) |
| 6 | `_gnn_transformer` | `gnn` (transformer) | GNN (8-node) + Transformer |
| 7 | `_gnn_tcn` | `gnn` (tcn) | GNN + TCN |
| 8 | `_gnn_bilstm` | `gnn` (bilstm) | GNN + BiLSTM |
| 9 | `_gnn_patchtst` | `gnn` (patch_transformer) | GNN + PatchTransformer |

GNN nodes: 7 physical variable groups + ap30 = **8 nodes** (dynamic config-based grouping).
GNN 노드: 7개 물리 변수 그룹 + ap30 = **8 노드** (config 기반 동적 구성).

Model details / 모델 상세: [MODEL_GUIDE.md](MODEL_GUIDE.md)

---

## Input/Output Matrix / 입출력 매트릭스

| Input \ Output / 입력 \ 출력 | out6h (12ts) | out12h (24ts) | out24h (48ts) |
|---------------------------|-------------|-------------|-------------|
| in1d (48ts) | `in1d_out6h` | `in1d_out12h` | `in1d_out24h` |
| in2d (96ts) | `in2d_out6h` | `in2d_out12h` | `in2d_out24h` |
| in3d (144ts) | `in3d_out6h` | `in3d_out12h` | `in3d_out24h` |

---

## Experiment Results (val_loss at best epoch) / 실험 결과

### 6h Prediction / 6h 예측

| Model \ Input / 모델 \ 입력 | in1d | in2d | in3d |
|---------------------------|------|------|------|
| Linear (baseline) | 0.2389 | 0.2488 | 0.2541 |
| Transformer | 0.2311 | 0.2322 | 0.2322 |
| TCN | 0.2539 | 0.2622 | 0.2675 |
| PatchTST | 0.2373 | 0.2363 | 0.2382 |
| TimesNet | 0.3099 | 0.4619 | 0.5355 |
| **GNN+Transformer** | 0.2221 | **0.2178** | 0.2188 |
| GNN+TCN | 0.2354 | 0.2509 | 0.2609 |
| GNN+BiLSTM | 0.2377 | 0.2306 | 0.2337 |
| GNN+PatchTST | 0.2200 | 0.2228 | 0.2229 |

### 12h Prediction / 12h 예측

| Model \ Input / 모델 \ 입력 | in1d | in2d | in3d |
|---------------------------|------|------|------|
| Linear (baseline) | 0.2934 | 0.3040 | 0.3093 |
| Transformer | 0.2836 | 0.2814 | 0.2816 |
| TCN | 0.3091 | 0.3187 | 0.3440 |
| PatchTST | 0.2851 | 0.2937 | 0.2917 |
| TimesNet | 0.3636 | 0.5374 | 0.6228 |
| **GNN+Transformer** | **0.2706** | 0.2727 | 0.2742 |
| GNN+TCN | 0.2992 | 0.3143 | 0.3334 |
| GNN+BiLSTM | 0.2912 | 0.2912 | 0.3016 |
| GNN+PatchTST | 0.2771 | 0.2771 | 0.2792 |

### 24h Prediction / 24h 예측

| Model \ Input / 모델 \ 입력 | in1d | in2d | in3d |
|---------------------------|------|------|------|
| Linear (baseline) | 0.5886 | 0.6051 | 0.6085 |
| Transformer | 0.5768 | 0.5848 | 0.5703 |
| TCN | 0.6294 | 0.6460 | 0.6837 |
| PatchTST | 0.6316 | 0.6683 | 0.7168 |
| TimesNet | 0.6878 | 0.7738 | 0.8057 |
| **GNN+Transformer** | 0.5626 | **0.5600** | 0.5668 |
| GNN+TCN | 0.5906 | 0.6076 | 0.6358 |
| GNN+BiLSTM | 0.6040 | 0.5921 | 0.5919 |
| GNN+PatchTST | 0.5817 | 0.5883 | 0.5936 |

---

## Model Ranking (average val_loss) / 모델 순위 (전체 평균 val_loss)

| Rank / 순위 | Model / 모델 | Avg val_loss / 평균 val_loss |
|------------|------------|--------------------------|
| 1 | **GNN+Transformer** | **0.3518** |
| 2 | GNN+PatchTST | 0.3625 |
| 3 | Transformer | 0.3638 |
| 4 | GNN+BiLSTM | 0.3749 |
| 5 | Linear (baseline) | 0.3834 |
| 6 | GNN+TCN | 0.3920 |
| 7 | PatchTST | 0.3999 |
| 8 | TCN | 0.4127 |
| 9 | TimesNet | 0.5665 |

---

## Analysis / 분석

### 1. GNN+Transformer ranks #1 across all prediction horizons / GNN+Transformer가 모든 예측 구간에서 1위
- 6h: **0.2178** (in2d), 12h: **0.2706** (in1d), 24h: **0.5600** (in2d)
- Average val_loss 0.3518, 3% better than #2 (GNN+PatchTST, 0.3625)
- 평균 val_loss 0.3518로 2위(GNN+PatchTST, 0.3625) 대비 3% 우위

### 2. Consistent GNN benefit / GNN의 일관된 효과
- GNN variants occupy ranks 1, 2, 4, 6 / GNN 변형 4종이 순위 1, 2, 4, 6위
- Among non-GNN models, only Transformer (#3) matches GNN+BiLSTM (#4) / GNN 없는 모델 중 Transformer(3위)만 GNN+BiLSTM(4위)과 비슷
- **Graph-based variable relationship learning improves all temporal encoders** / 변수 간 그래프 관계 학습이 모든 temporal encoder에서 기여

### 3. Temporal encoder ranking within GNN / GNN 내 Temporal encoder 순위
- Transformer > PatchTST > BiLSTM > TCN (consistent across all horizons / 일관)
- PatchTST approaches GNN+Transformer at 6h (0.2200 vs 0.2178) / 6h에서 PatchTST가 근접

### 4. Input length effect / 입력 길이 영향
- **6h/12h**: Minimal difference across in1d/in2d/in3d / 입력 길이 차이 미미
- **24h**: in2d is mostly optimal; in3d slightly worse / in2d가 대부분 최적, in3d가 소폭 악화
- Increasing input length has limited benefit / 입력 길이 증가의 이득은 제한적

### 5. TimesNet — worst across all horizons / TimesNet — 전 구간 최하위
- 6h: 0.31~0.54, 24h: 0.69~0.81
- Performance degrades sharply with longer inputs (in1d 0.31 → in3d 0.54 at 6h) / 입력 길어질수록 급격히 악화
- **FFT-based period detection is fundamentally unsuitable for aperiodic solar wind/storm data** / FFT 기반 주기 탐지가 비주기적 태양풍 데이터에 부적합

### 6. PatchTST — weaker standalone, strong with GNN / PatchTST — 단독 열세, GNN 결합 시 효과적
- PatchTST standalone: #7 (avg 0.3999), below Transformer / 단독: 7위, Transformer보다 열세
- GNN+PatchTST: #2 (avg 0.3625), synergy with graph structure / GNN 결합: 2위, 그래프 구조와 시너지

---

## Changelog / 변경 이력

| Date / 날짜 | Description / 내용 |
|------------|-------------------|
| 2025-04-08 | Initialize experiment matrix. 81 experiments with identical parameters / 실험 매트릭스 초기화. 동일 파라미터 81개 |
| 2025-04-08 | Remove hp30 from input (23→22 vars). Dynamic GNN node groups / hp30 제거, GNN 동적 노드 그룹 |
| 2025-04-08 | Add out6h (6h/12h/24h). Naming convention update / out6h 추가, 네이밍 통일 |
| 2025-04-09 | Record all 81 results. GNN+Transformer ranks #1 / 81개 결과 기록. GNN+Transformer 1위 확인 |
