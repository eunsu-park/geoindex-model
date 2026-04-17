# Experiment Log / 실험 기록

## Overview / 개요

Comparison of 9 model architectures × 6 input lengths × 4 output lengths = **216 experiments** under identical conditions.
9가지 모델 아키텍처 × 6 입력 길이 × 4 출력 길이 = **216개 실험**의 동일 조건 비교.

CSV-based 30-min solar wind time series → ap30 regression prediction.
CSV 기반 30분 간격 태양풍 시계열 → ap30 회귀 예측.

Dataset details / 데이터셋 상세: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## Common Settings / 공통 설정

Shared settings across all 216 experiments (base.yaml + local.yaml).
모든 216개 실험에서 공유하는 설정.

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
| Total Samples | 23,514 |

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

All 24 IO combinations. Each cell = `{input}_{output}` config name.
전체 24개 IO 조합. 각 셀 = `{입력}_{출력}` config 이름.

| Input \ Output / 입력 \ 출력 | out6h (12ts) | out12h (24ts) | out18h (36ts) | out24h (48ts) |
|---------------------------|-------------|-------------|-------------|-------------|
| in6h (12ts) | `in6h_out6h` | `in6h_out12h` | `in6h_out18h` | `in6h_out24h` |
| in12h (24ts) | `in12h_out6h` | `in12h_out12h` | `in12h_out18h` | `in12h_out24h` |
| in18h (36ts) | `in18h_out6h` | `in18h_out12h` | `in18h_out18h` | `in18h_out24h` |
| in1d (48ts) | `in1d_out6h` | `in1d_out12h` | `in1d_out18h` | `in1d_out24h` |
| in2d (96ts) | `in2d_out6h` | `in2d_out12h` | `in2d_out18h` | `in2d_out24h` |
| in3d (144ts) | `in3d_out6h` | `in3d_out12h` | `in3d_out18h` | `in3d_out24h` |

---

## Metric Heatmaps (best model per IO) / 지표 히트맵 (IO별 최적 모델)

Each cell shows the best-performing model's score for that IO combination.
각 셀은 해당 IO 조합에서 최고 성능 모델의 점수를 나타냅니다.

### R2 (higher is better / 높을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 0.6788 | 0.6619 | 0.5466 | 0.4668 |
| **12h** | 0.7439 | 0.6994 | 0.5612 | 0.4680 |
| **18h** | 0.7449 | 0.7011 | 0.5704 | 0.4520 |
| **1d** | **0.7455** | 0.6954 | 0.5542 | 0.4495 |
| **2d** | 0.7414 | 0.6930 | 0.5602 | 0.4510 |
| **3d** | 0.7383 | 0.6875 | 0.5546 | 0.4313 |

### MAE (lower is better / 낮을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 0.3932 | 0.4024 | 0.4604 | 0.4982 |
| **12h** | 0.3509 | 0.3781 | 0.4498 | 0.4966 |
| **18h** | **0.3501** | **0.3765** | **0.4450** | 0.5050 |
| **1d** | 0.3504 | 0.3825 | 0.4535 | 0.5068 |
| **2d** | 0.3532 | 0.3835 | 0.4530 | 0.5069 |
| **3d** | 0.3557 | 0.3873 | 0.4565 | 0.5166 |

### RMSE (lower is better / 낮을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 0.5114 | 0.5255 | 0.6085 | 0.6596 |
| **12h** | 0.4566 | 0.4956 | 0.5986 | 0.6588 |
| **18h** | 0.4557 | **0.4942** | **0.5924** | 0.6687 |
| **1d** | **0.4552** | 0.4988 | 0.6034 | 0.6701 |
| **2d** | 0.4588 | 0.5008 | 0.5993 | 0.6692 |
| **3d** | 0.4616 | 0.5053 | 0.6031 | 0.6811 |

### Median AE (lower is better / 낮을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 0.3138 | 0.3184 | 0.3600 | 0.3865 |
| **12h** | **0.2790** | 0.2987 | 0.3468 | 0.3850 |
| **18h** | 0.2808 | **0.2973** | **0.3430** | 0.3901 |
| **1d** | 0.2809 | 0.3049 | 0.3489 | 0.3932 |
| **2d** | 0.2807 | 0.3033 | 0.3541 | 0.3959 |
| **3d** | 0.2855 | 0.3072 | 0.3580 | 0.4015 |

### Max Error (lower is better / 낮을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 2.5362 | 3.0212 | 3.5005 | 3.9637 |
| **12h** | 2.3914 | 2.9458 | 3.4852 | 3.9388 |
| **18h** | 2.3449 | 2.9600 | 3.5179 | 4.0077 |
| **1d** | **2.2693** | **2.9083** | 3.4252 | **3.8870** |
| **2d** | 2.2883 | 2.9852 | 3.4831 | 4.0289 |
| **3d** | 2.3180 | 3.0690 | **3.3635** | 3.8597 |

### Bias (closest to 0 is best / 0에 가까울수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | -0.0010 | +0.0064 | +0.0404 | +0.0108 |
| **12h** | +0.0049 | -0.0014 | +0.0141 | +0.0249 |
| **18h** | +0.0029 | +0.0095 | +0.0044 | +0.0257 |
| **1d** | +0.0012 | +0.0059 | +0.0185 | -0.0117 |
| **2d** | **+0.0002** | +0.0232 | -0.0188 | -0.0024 |
| **3d** | +0.0008 | +0.0097 | +0.0114 | +0.0454 |

### MAPE (%) (lower is better / 낮을수록 좋음)

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **6h** | 156.43 | 162.50 | 178.86 | 180.33 |
| **12h** | 146.04 | 150.53 | 177.02 | 183.48 |
| **18h** | **142.93** | 152.42 | **169.53** | 192.27 |
| **1d** | 142.28 | 157.21 | 176.24 | 189.86 |
| **2d** | 148.31 | 158.39 | 173.36 | 191.82 |
| **3d** | 147.83 | 160.62 | 175.66 | 188.19 |

> Note: MAPE values are high (>100%) because ap30 frequently has near-zero values, causing extreme percentage errors. Use MAE/RMSE/R2 as primary metrics.
> 참고: MAPE가 높은 이유는 ap30이 0에 가까운 값이 빈번하여 백분율 오차가 극단적으로 커지기 때문입니다. MAE/RMSE/R2를 주 지표로 사용하세요.

---

## Best Model per IO Combination / IO 조합별 최적 모델

| Input \ Output | out6h | out12h | out18h | out24h |
|---|---|---|---|---|
| **in6h** | gnn_patchtst (R2=0.679) | gnn_transformer (R2=0.662) | gnn_patchtst (R2=0.547) | gnn_transformer (R2=0.467) |
| **in12h** | gnn_transformer (R2=0.744) | gnn_patchtst (R2=0.699) | linear (R2=0.561) | gnn_transformer (R2=0.468) |
| **in18h** | gnn_patchtst (R2=0.745) | gnn_patchtst (R2=0.701) | gnn_patchtst (R2=0.570) | gnn_transformer (R2=0.452) |
| **in1d** | gnn_patchtst (R2=0.746) | gnn_transformer (R2=0.695) | linear (R2=0.554) | gnn_transformer (R2=0.450) |
| **in2d** | gnn_transformer (R2=0.741) | gnn_transformer (R2=0.693) | gnn_patchtst (R2=0.560) | gnn_transformer (R2=0.451) |
| **in3d** | gnn_transformer (R2=0.738) | gnn_patchtst (R2=0.688) | gnn_patchtst (R2=0.555) | gnn_transformer (R2=0.431) |

---

## Top 5 Rankings / 상위 5개 순위

### Top 5 by R2 / R2 기준 상위 5개

| Rank | Experiment | R2 | MAE | RMSE |
|---|---|---|---|---|
| 1 | in1d_out6h_gnn_patchtst | **0.7455** | 0.3504 | 0.4552 |
| 2 | in18h_out6h_gnn_patchtst | 0.7449 | 0.3501 | 0.4557 |
| 3 | in12h_out6h_gnn_transformer | 0.7439 | 0.3509 | 0.4566 |
| 4 | in2d_out6h_gnn_transformer | 0.7414 | 0.3532 | 0.4588 |
| 5 | in12h_out6h_gnn_patchtst | 0.7410 | 0.3529 | 0.4592 |

### Top 5 by MAE / MAE 기준 상위 5개

| Rank | Experiment | MAE | R2 | RMSE |
|---|---|---|---|---|
| 1 | in18h_out6h_gnn_patchtst | **0.3501** | 0.7449 | 0.4557 |
| 2 | in1d_out6h_gnn_patchtst | 0.3504 | 0.7455 | 0.4552 |
| 3 | in12h_out6h_gnn_transformer | 0.3509 | 0.7439 | 0.4566 |
| 4 | in12h_out6h_gnn_patchtst | 0.3529 | 0.7410 | 0.4592 |
| 5 | in2d_out6h_gnn_transformer | 0.3532 | 0.7414 | 0.4588 |

### Top 5 by RMSE / RMSE 기준 상위 5개

| Rank | Experiment | RMSE | R2 | MAE |
|---|---|---|---|---|
| 1 | in1d_out6h_gnn_patchtst | **0.4552** | 0.7455 | 0.3504 |
| 2 | in18h_out6h_gnn_patchtst | 0.4557 | 0.7449 | 0.3501 |
| 3 | in12h_out6h_gnn_transformer | 0.4566 | 0.7439 | 0.3509 |
| 4 | in2d_out6h_gnn_transformer | 0.4588 | 0.7414 | 0.3532 |
| 5 | in12h_out6h_gnn_patchtst | 0.4592 | 0.7410 | 0.3529 |

### Top 5 by Median AE / Median AE 기준 상위 5개

| Rank | Experiment | Median AE | R2 | MAE |
|---|---|---|---|---|
| 1 | in12h_out6h_gnn_transformer | **0.2790** | 0.7439 | 0.3509 |
| 2 | in2d_out6h_gnn_transformer | 0.2807 | 0.7414 | 0.3532 |
| 3 | in18h_out6h_gnn_patchtst | 0.2808 | 0.7449 | 0.3501 |
| 4 | in1d_out6h_gnn_patchtst | 0.2809 | 0.7455 | 0.3504 |
| 5 | in12h_out6h_gnn_patchtst | 0.2841 | 0.7410 | 0.3529 |

### Top 5 by |Bias| / |Bias| 기준 상위 5개

| Rank | Experiment | Bias | R2 | MAE |
|---|---|---|---|---|
| 1 | in2d_out6h_timesnet | +0.0002 | 0.5245 | 0.4701 |
| 2 | in3d_out6h_linear | +0.0008 | 0.7126 | 0.3706 |
| 3 | in6h_out6h_transformer | -0.0010 | 0.6698 | 0.3997 |
| 4 | in1d_out6h_gnn_patchtst | +0.0012 | 0.7455 | 0.3504 |
| 5 | in12h_out12h_transformer | -0.0014 | 0.6829 | 0.3913 |

### Top 5 by Max Error / Max Error 기준 상위 5개

| Rank | Experiment | Max Error | R2 | MAE |
|---|---|---|---|---|
| 1 | in1d_out6h_linear | **2.2693** | 0.7260 | 0.3637 |
| 2 | in2d_out6h_gnn_patchtst | 2.2883 | 0.7379 | 0.3570 |
| 3 | in3d_out6h_patchtst | 2.3180 | 0.7190 | 0.3705 |
| 4 | in3d_out6h_gnn_tcn | 2.3362 | 0.6957 | 0.3848 |
| 5 | in2d_out6h_gnn_transformer | 2.3434 | 0.7414 | 0.3532 |

---

## Composite Top 5 (Overall) / 종합 상위 5개

Ranking method: average rank across R2, MAE, RMSE, Median AE, Max Error, and |Bias| (6 metrics). Lower average rank = better overall performance.
순위 산정 방식: R2, MAE, RMSE, Median AE, Max Error, |Bias| 6개 지표의 평균 순위. 평균 순위가 낮을수록 종합적으로 우수.

| Rank | Experiment | Avg Rank | MAE | RMSE | R2 | Bias | Median AE | Max Error |
|---|---|---|---|---|---|---|---|---|
| **1** | **in1d_out6h_gnn_patchtst** | **6.2** | 0.3504 | 0.4552 | 0.7455 | +0.0012 | 0.2809 | 2.4414 |
| 2 | in2d_out6h_gnn_transformer | 8.3 | 0.3532 | 0.4588 | 0.7414 | +0.0134 | 0.2807 | 2.3434 |
| 3 | in12h_out6h_gnn_transformer | 8.8 | 0.3509 | 0.4566 | 0.7439 | -0.0180 | 0.2790 | 2.3914 |
| 4 | in12h_out6h_patchtst | 9.0 | 0.3552 | 0.4619 | 0.7379 | +0.0049 | 0.2842 | 2.4134 |
| 5 | in18h_out6h_gnn_patchtst | 9.2 | 0.3501 | 0.4557 | 0.7449 | +0.0187 | 0.2808 | 2.3449 |

### Composite Top 3 per Output Horizon / 출력 구간별 종합 상위 3개

#### out6h

| Rank | Experiment | Avg Rank | MAE | R2 | Bias |
|---|---|---|---|---|---|
| 1 | in18h_out6h_gnn_patchtst | 5.7 | 0.3501 | 0.7449 | +0.0187 |
| 2 | in2d_out6h_gnn_transformer | 5.8 | 0.3532 | 0.7414 | +0.0134 |
| 3 | in12h_out6h_gnn_transformer | 6.0 | 0.3509 | 0.7439 | -0.0180 |

#### out12h

| Rank | Experiment | Avg Rank | MAE | R2 | Bias |
|---|---|---|---|---|---|
| 1 | in12h_out12h_gnn_patchtst | 5.2 | 0.3781 | 0.6994 | +0.0186 |
| 2 | in18h_out12h_gnn_patchtst | 6.3 | 0.3765 | 0.7011 | +0.0114 |
| 3 | in12h_out12h_gnn_transformer | 7.3 | 0.3836 | 0.6921 | +0.0146 |

#### out18h

| Rank | Experiment | Avg Rank | MAE | R2 | Bias |
|---|---|---|---|---|---|
| 1 | in1d_out18h_linear | 4.7 | 0.4535 | 0.5542 | +0.0185 |
| 2 | in18h_out18h_gnn_patchtst | 6.0 | 0.4450 | 0.5704 | +0.0044 |
| 3 | in12h_out18h_linear | 6.2 | 0.4498 | 0.5612 | +0.0391 |

#### out24h

| Rank | Experiment | Avg Rank | MAE | R2 | Bias |
|---|---|---|---|---|---|
| 1 | in6h_out24h_patchtst | 5.7 | 0.5057 | 0.4503 | +0.0108 |
| 2 | in12h_out24h_gnn_transformer | 6.2 | 0.4966 | 0.4680 | +0.0724 |
| 3 | in6h_out24h_gnn_transformer | 6.3 | 0.4982 | 0.4668 | +0.0581 |

---

## Aggregated Comparisons / 집계 비교

### Model Comparison (averaged across all 24 IO combinations) / 모델 비교

| Model | Avg MAE | Avg RMSE | Avg R2 | Avg |Bias| | R2 Wins | MAE Wins |
|---|---|---|---|---|---|---|
| **gnn_patchtst** | 0.4290 | 0.5637 | **0.6021** | 0.0436 | 10/24 | 11/24 |
| **gnn_transformer** | 0.4300 | 0.5641 | 0.6020 | 0.0590 | 12/24 | 10/24 |
| linear | 0.4339 | 0.5707 | 0.5934 | **0.0339** | 2/24 | 2/24 |
| transformer | 0.4389 | 0.5741 | 0.5881 | 0.0582 | 0/24 | 0/24 |
| gnn_bilstm | 0.4404 | 0.5763 | 0.5845 | 0.0577 | 0/24 | 0/24 |
| patchtst | 0.4447 | 0.5826 | 0.5737 | 0.0362 | 0/24 | 0/24 |
| gnn_tcn | 0.4507 | 0.5883 | 0.5669 | 0.0664 | 0/24 | 1/24 |
| tcn | 0.4574 | 0.5969 | 0.5542 | 0.0585 | 0/24 | 0/24 |
| timesnet | 0.4937 | 0.6502 | 0.4706 | 0.0706 | 0/24 | 0/24 |

### Input Window Comparison / 입력 윈도우 비교

| Input | Avg MAE | Avg RMSE | Avg R2 | Avg |Bias| |
|---|---|---|---|---|
| **in12h** | **0.4324** | **0.5674** | **0.5973** | **0.0431** |
| in18h | 0.4375 | 0.5742 | 0.5872 | 0.0432 |
| in1d | 0.4432 | 0.5810 | 0.5768 | 0.0502 |
| in6h | 0.4462 | 0.5850 | 0.5755 | 0.0476 |
| in2d | 0.4552 | 0.5960 | 0.5531 | 0.0597 |
| in3d | 0.4644 | 0.6077 | 0.5339 | 0.0789 |

### Output Horizon Comparison / 출력 구간 비교

| Output | Avg MAE | Avg RMSE | Avg R2 | Avg |Bias| |
|---|---|---|---|---|
| **out6h** | **0.3784** | **0.4909** | **0.7021** | **0.0302** |
| out12h | 0.4052 | 0.5285 | 0.6554 | 0.0378 |
| out18h | 0.4758 | 0.6272 | 0.5170 | 0.0693 |
| out24h | 0.5267 | 0.6943 | 0.4079 | 0.0778 |

### Linear vs Best DL Model (R2 gain) / Linear 대비 최고 DL 모델 R2 이득

| Input \ Output | 6h | 12h | 18h | 24h |
|---|---|---|---|---|
| **in6h** | +0.0083 | +0.0032 | +0.0060 | +0.0096 |
| **in12h** | +0.0133 | +0.0125 | -0.0043 \* | +0.0200 |
| **in18h** | +0.0153 | +0.0166 | +0.0171 | +0.0054 |
| **in1d** | +0.0195 | +0.0108 | -0.0036 \* | +0.0058 |
| **in2d** | +0.0269 | +0.0250 | +0.0291 | +0.0214 |
| **in3d** | +0.0257 | +0.0210 | +0.0143 | +0.0274 |

\* Linear outperforms all DL models / Linear이 모든 DL 모델보다 우수

---

## Key Findings / 핵심 발견

### 1. Output horizon is the dominant factor / 출력 예측 구간이 성능의 지배적 요인

The prediction horizon has the strongest impact on performance by far. R2 drops from 0.70 (6h) → 0.66 (12h) → 0.52 (18h) → 0.41 (24h). This is consistent across all models and input windows, confirming that ap30 predictability degrades rapidly beyond 12 hours. The 18h horizon was newly added in this round: its R2 (0.52-0.57) falls squarely between 12h and 24h, confirming a smooth degradation curve rather than a cliff.

출력 예측 구간이 성능에 가장 큰 영향을 미칩니다. R2가 0.70(6h) → 0.66(12h) → 0.52(18h) → 0.41(24h)로 하락합니다. 이 패턴은 모든 모델과 입력 윈도우에서 일관되며, ap30 예측 가능성이 12시간 이후 급격히 떨어진다는 것을 확인합니다. 이번에 새로 추가된 18h 구간의 R2(0.52-0.57)는 12h와 24h 사이에 정확히 위치하여 점진적 하락 곡선을 확인할 수 있습니다.

**Practical usability / 실용성 판단:**
- **out6h** (R2 > 0.70): Operationally useful / 운영 활용 가능
- **out12h** (R2 ~ 0.66-0.70): Useful with caveats / 주의하여 활용 가능
- **out18h** (R2 ~ 0.52-0.57): Limited practical value / 실용성 제한적
- **out24h** (R2 ~ 0.41-0.47): Trend indication only / 추세 참고 수준

### 2. Optimal input window is 12h-18h / 최적 입력 윈도우는 12h-18h

The relationship between input length and performance follows an inverted-U pattern. 6h input is clearly insufficient (avg R2 = 0.576), but performance peaks at 12h-18h (R2 = 0.587-0.597) and then declines for longer inputs (in2d = 0.553, in3d = 0.534). The newly added 18h input performs strongly, achieving the best MAE for out6h (0.3501) and out12h (0.3765), and the best R2 for out18h (0.5704).

입력 길이와 성능의 관계는 역U자형입니다. 6h 입력은 확실히 부족하고(평균 R2 = 0.576), 12h-18h에서 정점(R2 = 0.587-0.597)을 찍은 후 더 긴 입력에서는 하락합니다(in2d = 0.553, in3d = 0.534). 새로 추가된 18h 입력은 out6h MAE(0.3501)와 out12h MAE(0.3765)에서 최저, out18h R2(0.5704)에서 최고를 기록하며 강력한 성능을 보입니다.

This suggests ~12-24 hours of solar wind history is the information-theoretic sweet spot. Longer inputs introduce noise from older, less relevant observations that dilute the signal.

이는 약 12-24시간의 태양풍 이력이 정보 이론적으로 최적 구간임을 시사합니다. 더 긴 입력은 관련성이 낮은 과거 관측값의 노이즈를 도입하여 신호를 희석시킵니다.

### 3. GNN+PatchTST and GNN+Transformer are co-champions / GNN+PatchTST와 GNN+Transformer가 공동 1위

These two models are virtually tied in average performance (R2: 0.6021 vs 0.6020) but show complementary strengths:
이 두 모델은 평균 성능이 거의 동일하지만(R2: 0.6021 vs 0.6020) 상호 보완적 강점을 보입니다:

- **gnn_patchtst**: Wins 10/24 by R2, 11/24 by MAE. Dominates short-to-mid horizons (out6h, out12h, out18h). Lower bias on average (0.044 vs 0.059).
- **gnn_patchtst**: R2 기준 10/24, MAE 기준 11/24에서 1위. 단-중기 예측(out6h, out12h, out18h)에서 우세. 평균 bias도 낮음(0.044 vs 0.059).

- **gnn_transformer**: Wins 12/24 by R2, 10/24 by MAE. Dominates out24h (wins all 6 input combos). Slightly better at long-range prediction.
- **gnn_transformer**: R2 기준 12/24, MAE 기준 10/24에서 1위. out24h 전 구간 1위. 장기 예측에서 소폭 우위.

Together, they win **22 out of 24** IO combinations, leaving only 2 for linear.
두 모델이 합쳐서 24개 IO 조합 중 **22개에서 1위**를 차지하며, 나머지 2개만 linear에게 양보합니다.

### 4. Linear baseline is surprisingly competitive / Linear baseline이 놀랍게 강력

The linear model ranks 3rd overall (avg R2 = 0.5934) and has the lowest average |Bias| (0.034). It even outperforms all DL models in 2 IO combinations (in12h_out18h, in1d_out18h). The margin between linear and the best DL model is typically only R2 +0.01~0.03. This suggests the prediction task has a strong linear component, and DL models provide only marginal non-linear improvement.

Linear 모델이 전체 3위(평균 R2 = 0.5934)를 기록하며 평균 |Bias|도 가장 낮습니다(0.034). 2개 IO 조합(in12h_out18h, in1d_out18h)에서는 모든 DL 모델보다 우수합니다. Linear과 최고 DL 모델의 차이는 보통 R2 +0.01~0.03에 불과합니다. 이는 예측 과제에 강한 선형 성분이 있으며, DL 모델의 비선형적 이득이 제한적임을 시사합니다.

However, the DL advantage grows with longer inputs (in2d/in3d: gain +0.02~0.03) where linear models struggle to extract relevant patterns from noisy long sequences.

다만 DL 이득은 입력이 길수록 커지는데(in2d/in3d: +0.02~0.03), 이는 linear 모델이 긴 시퀀스에서 관련 패턴을 추출하기 어려운 반면 GNN이 변수 간 관계를 효과적으로 모델링하기 때문입니다.

### 5. GNN consistently improves all temporal encoders / GNN이 모든 temporal encoder를 일관되게 개선

Comparing GNN vs non-GNN variants of the same temporal backbone:
동일 temporal backbone의 GNN/non-GNN 변형을 비교하면:

| Temporal encoder | Without GNN (R2) | With GNN (R2) | Improvement |
|---|---|---|---|
| Transformer | 0.5881 | 0.6020 | +0.0139 |
| PatchTST | 0.5737 | 0.6021 | +0.0284 |
| TCN | 0.5542 | 0.5669 | +0.0127 |

GNN benefits PatchTST the most (+0.028), likely because patch-based tokenization aligns well with per-node GNN embeddings, creating natural variable-group representations.

GNN은 PatchTST에 가장 큰 이득을 줍니다(+0.028). 이는 패치 기반 토크나이징이 GNN 노드별 임베딩과 자연스럽게 정렬되어 변수 그룹 표현을 효과적으로 생성하기 때문으로 보입니다.

### 6. TimesNet is fundamentally unsuitable / TimesNet은 근본적으로 부적합

TimesNet ranks last across all metrics (avg R2 = 0.471), far behind TCN (0.554). Its FFT-based period detection assumes periodic patterns, which do not exist in aperiodic solar wind/geomagnetic storm data. Performance degrades dramatically with longer inputs, confirming the architectural mismatch.

TimesNet이 모든 지표에서 최하위(평균 R2 = 0.471)로, TCN(0.554)보다 크게 뒤처집니다. FFT 기반 주기 탐지가 비주기적 태양풍/자기폭풍 데이터에는 근본적으로 부적합합니다. 입력이 길어질수록 성능이 급격히 악화되어 아키텍처적 부적합을 확인합니다.

### 7. Bias increases with prediction horizon / 예측 구간 증가에 따른 Bias 증가

Average |Bias| grows from 0.030 (out6h) → 0.038 (out12h) → 0.069 (out18h) → 0.078 (out24h). This indicates models tend toward mean-reversion at longer horizons, under-predicting extreme events. This is a known challenge for geomagnetic storm prediction and an area for potential improvement through loss function tuning or post-processing calibration.

평균 |Bias|가 0.030(out6h) → 0.038(out12h) → 0.069(out18h) → 0.078(out24h)로 증가합니다. 이는 모델이 장기 예측에서 평균 회귀 경향을 보이며 극한 이벤트를 과소 예측하는 것을 나타냅니다. 이는 자기폭풍 예측의 알려진 과제로, loss function 조정이나 후처리 보정을 통해 개선할 수 있는 영역입니다.

### 8. Why longer inputs degrade performance / 긴 입력이 성능을 저하시키는 이유

Performance peaks at in12h-in18h and degrades for in2d/in3d. This is caused by a combination of physical and architectural factors.

성능이 in12h-in18h에서 정점을 찍고 in2d/in3d에서 하락합니다. 이는 물리적 원인과 아키텍처적 원인이 결합된 결과입니다.

#### Physical cause: limited causal time scale of solar wind–magnetosphere coupling / 물리적 원인: 태양풍-자기권 결합의 제한된 인과적 시간 스케일

The ap30 index responds to solar wind conditions through a fast causal chain: IMF Bz southward turning → dayside magnetopause reconnection → substorm/storm development → enhanced ring current and auroral electrojet → ap increase. This entire process operates on timescales of **minutes to hours**. Solar wind conditions from 2–3 days ago have essentially no causal connection to the current magnetospheric state, because the solar wind is turbulent and non-stationary — past conditions do not determine future conditions.

ap30 지수는 빠른 인과 사슬을 통해 태양풍에 반응합니다: IMF Bz 남향 전환 → 주간측 자기권계면 재결합 → 서브스톰/자기폭풍 발달 → 환전류 및 오로라 전기제트 강화 → ap 상승. 이 전체 과정은 **수분~수시간** 시간 스케일에서 작동합니다. 2~3일 전 태양풍 상태는 현재 자기권 상태와 인과적 연결이 본질적으로 없습니다. 태양풍은 난류성이 강하고 비정상적(non-stationary)이어서 과거 조건이 미래 조건을 결정하지 않기 때문입니다.

Therefore, inputs beyond ~12–24 hours add noise without adding predictive signal. The 12–18 hour sweet spot corresponds to roughly 1–1.5× the Dst recovery time, which aligns with the physical memory of the magnetospheric system.

따라서 약 12~24시간을 넘어서는 입력은 예측 신호 없이 노이즈만 추가합니다. 12~18시간의 최적 구간은 Dst 회복 시간의 약 1~1.5배에 해당하며, 이는 자기권 시스템의 물리적 기억(memory)과 일치합니다.

#### Architectural cause: attention dilution and capacity saturation / 아키텍처적 원인: 주의력 희석과 용량 포화

The per-model breakdown reveals how different architectures handle long inputs:

모델별 분해를 통해 각 아키텍처가 긴 입력을 어떻게 처리하는지 확인할 수 있습니다:

| Model | R2 peak input | R2 at in6h | R2 at in3d | Degradation (peak→in3d) |
|---|---|---|---|---|
| linear | in12h | 0.6705 | 0.7126 | -0.0180 |
| transformer | in12h | 0.6698 | 0.7183 | -0.0131 |
| gnn_patchtst | in1d | 0.6788 | 0.7312 | -0.0143 |
| gnn_transformer | in12h | 0.6752 | 0.7383 | -0.0056 |
| timesnet | in12h | 0.6603 | 0.4239 | -0.2709 |

(out6h results shown / out6h 결과 기준)

Key observations / 주요 관찰:

- **Attention dilution**: Transformer-based models peak at in12h. Longer sequences spread self-attention across irrelevant past timesteps, making it harder to focus on the causally relevant recent window. The model's d_model=128 is fixed regardless of input length, so the representational capacity per timestep decreases.
- **주의력 희석**: Transformer 기반 모델은 in12h에서 정점. 긴 시퀀스는 self-attention을 인과적으로 무관한 과거 timestep에 분산시켜, 관련 있는 최근 구간에 집중하기 어렵게 만듭니다. d_model=128이 입력 길이와 무관하게 고정되어 timestep당 표현 용량이 줄어듭니다.

- **GNN models are more robust**: gnn_transformer degrades only -0.006 from peak to in3d. The GNN's per-variable-group encoding acts as a noise filter — by first learning inter-variable relationships at the graph level, it can better separate signal from noise in long sequences.
- **GNN 모델의 강건성**: gnn_transformer는 peak에서 in3d까지 -0.006만 하락합니다. GNN의 변수 그룹별 인코딩이 노이즈 필터 역할을 하여, 그래프 수준에서 변수 간 관계를 먼저 학습함으로써 긴 시퀀스에서 신호와 노이즈를 더 잘 분리합니다.

- **TimesNet catastrophic failure**: R2 drops from 0.695 (in12h) to 0.424 (in3d) — a 0.27 collapse. FFT interprets noise in long aperiodic sequences as spurious periodic components, actively corrupting the learned representation.
- **TimesNet의 파국적 실패**: R2가 0.695(in12h)에서 0.424(in3d)로 0.27이나 폭락합니다. FFT가 긴 비주기 시퀀스의 노이즈를 가짜 주기 성분으로 해석하여 학습된 표현을 적극적으로 손상시킵니다.

- **Exception — extreme events benefit from longer inputs**: Max Error actually improves with longer inputs (in6h: 2.54 → in1d: 2.27 for out6h). Large geomagnetic storms are often preceded by multi-day precursor patterns (high-speed solar wind streams, CIR structures), which only longer input windows can capture.
- **예외 — 극단 이벤트는 긴 입력에서 개선**: Max Error는 긴 입력에서 오히려 개선됩니다(in6h: 2.54 → in1d: 2.27, out6h 기준). 큰 자기폭풍은 종종 수일간의 전조 패턴(고속 태양풍 스트림, CIR 구조)이 선행하며, 이는 긴 입력 윈도우에서만 포착됩니다.

---

## Future Directions / 향후 개선 방향

### A. Improving long-input and long-output performance / 긴 입력·출력 성능 개선

The current architecture uses uniform attention across all timesteps, treating recent and distant past equally. Several strategies could improve handling of longer sequences:

현재 아키텍처는 모든 timestep에 균일한 attention을 적용하여 최근과 먼 과거를 동등하게 취급합니다. 긴 시퀀스 처리를 개선하기 위한 전략들:

#### A1. Hierarchical temporal encoding / 계층적 시간 인코딩

Instead of processing all timesteps at the same resolution, use a multi-resolution approach: fine-grained (30-min) for recent data (e.g., last 12h), coarse-grained (e.g., 2h or 6h averages) for older data. This preserves the detail where it matters most while compressing noise in distant past.

모든 timestep을 동일 해상도로 처리하는 대신, 다중 해상도 접근법을 사용합니다: 최근 데이터(예: 최근 12h)는 세밀하게(30분), 과거 데이터는 거칠게(예: 2h 또는 6h 평균). 이는 중요한 부분의 세부 사항을 보존하면서 먼 과거의 노이즈를 압축합니다.

```
[--- 3d ago: 6h avg ---][--- 1d ago: 2h avg ---][--- recent 12h: 30min ---]
     12 tokens              12 tokens              24 tokens = 48 total
vs. current in3d: 144 tokens at uniform 30-min resolution
```

#### A2. Learnable temporal attention gate / 학습 가능한 시간 주의력 게이트

Add a lightweight temporal importance module before the main encoder that learns to weight or mask timesteps based on their relevance. This lets the model explicitly learn that "2 days ago doesn't matter" rather than relying on attention to implicitly ignore it.

주 인코더 이전에 경량 시간 중요도 모듈을 추가하여, 관련성에 따라 timestep을 가중하거나 마스킹하도록 학습합니다. 이를 통해 모델이 attention에 암묵적으로 의존하는 대신 "2일 전은 중요하지 않다"는 것을 명시적으로 학습할 수 있습니다.

#### A3. Output-horizon-aware loss weighting / 출력 구간별 손실 가중치

For longer output horizons, apply higher loss weights to near-future timesteps and lower weights to distant-future timesteps (or vice versa for storm onset detection). Current uniform weighting forces the model to average performance across all output timesteps, which may compromise near-future accuracy.

긴 출력 구간에서는 가까운 미래 timestep에 높은 손실 가중치를, 먼 미래에 낮은 가중치를 적용합니다(또는 자기폭풍 발생 감지를 위해 반대로). 현재의 균일 가중치는 모델이 모든 출력 timestep에 걸쳐 성능을 평균화하도록 강제하여, 근미래 정확도를 타협할 수 있습니다.

#### A4. Autoregressive or iterative refinement / 자기회귀 또는 반복 정제

Instead of predicting the entire output horizon in one shot, use a rolling approach: predict 6h → feed prediction back → predict next 6h. This was shown to work well in weather forecasting (e.g., Pangu-Weather). The current direct multi-step approach may struggle with long horizons because small errors compound.

전체 출력 구간을 한 번에 예측하는 대신 롤링 방식을 사용합니다: 6h 예측 → 예측값 피드백 → 다음 6h 예측. 이는 기상 예보(예: Pangu-Weather)에서 효과적으로 입증되었습니다. 현재의 직접 다단계 방식은 작은 오차가 누적되어 긴 구간에서 어려움을 겪을 수 있습니다.

#### A5. Sparse or local attention patterns / 희소 또는 지역 주의력 패턴

Replace full self-attention with sparse patterns that enforce locality bias: each timestep attends primarily to nearby timesteps, with sparse long-range connections. This directly addresses the attention dilution problem.

전체 self-attention을 지역성 편향을 강제하는 희소 패턴으로 대체합니다: 각 timestep이 주로 인접 timestep에 주의를 기울이되, 희소한 장거리 연결을 유지합니다. 이는 주의력 희석 문제를 직접적으로 해결합니다.

#### A6. Extreme event specialization / 극단 이벤트 특화

Since longer inputs help specifically with extreme events (lower Max Error), consider a two-stage approach: a short-input model for general prediction + a long-input model specialized for storm detection/intensity. An ensemble or routing mechanism could select the appropriate model based on current solar wind conditions.

긴 입력이 극단 이벤트에 특히 도움이 되므로(낮은 Max Error), 2단계 접근법을 고려합니다: 일반 예측을 위한 단기 입력 모델 + 자기폭풍 감지/강도에 특화된 장기 입력 모델. 앙상블 또는 라우팅 메커니즘이 현재 태양풍 조건에 따라 적절한 모델을 선택할 수 있습니다.

### B. Incorporating solar observation images / 태양 관측 영상 도입

#### B1. Motivation / 동기

The current model relies exclusively on in-situ solar wind measurements at L1. It can follow trends in solar wind parameters well, but it has a fundamental limitation: **it cannot know about upcoming events until the solar wind carrying them arrives at L1**, which is only ~30–60 minutes before Earth impact. The model has no way to anticipate CMEs, high-speed streams, or other transient events before they appear in the in-situ data.

현재 모델은 L1 지점의 현장 태양풍 관측에만 의존합니다. 태양풍 매개변수의 추세는 잘 따라가지만, 근본적인 한계가 있습니다: **태양풍이 L1에 도달하기 전까지는 다가오는 이벤트를 알 수 없으며**, 이는 지구 영향 불과 ~30-60분 전입니다. 모델은 CME, 고속 스트림 또는 기타 과도 이벤트가 현장 데이터에 나타나기 전에는 이를 예측할 방법이 없습니다.

Solar observation images provide a solution: they show events (CMEs, flares, coronal hole evolution) happening **at the Sun**, 1–4 days before their effects reach Earth. By incorporating these images, the model can learn that "a source event has occurred on the Sun, and after a certain transit time, it will affect Earth" — transforming the prediction from pure extrapolation to physically informed forecasting.

태양 관측 영상이 해결책을 제공합니다: 태양에서 발생하는 이벤트(CME, 플레어, 코로나 홀 진화)를 그 영향이 지구에 도달하기 **1~4일 전에** 보여줍니다. 이 영상을 도입함으로써 모델은 "태양에서 소스 이벤트가 발생했고, 일정한 이동 시간 후 지구에 영향을 줄 것이다"라는 것을 학습할 수 있습니다 — 예측을 순수 외삽에서 물리적으로 근거 있는 예보로 전환합니다.

**The two modalities play asymmetric, complementary roles / 두 모달리티는 비대칭적·상호 보완적 역할**:

```
Solar images (D-7 ~ D-1)  →  "What event is coming?"     (probabilistic precursor)
                               "어떤 이벤트가 오고 있는가?"    (확률적 전조 정보)

In-situ data (H-18 ~ H-0) →  "What is happening now?"    (deterministic current state)
                               "지금 무엇이 일어나고 있는가?"  (확정적 현재 상태)

Combined → "Event X is arriving, and current conditions confirm it will cause a storm"
결합 → "이벤트 X가 도달 중이며, 현재 조건이 자기폭풍 발생을 확인"
```

#### B2. Physical caveats — what the model must learn, not assume / 물리적 주의사항

Several common assumptions about solar events and geomagnetic storms are oversimplified. The model architecture and training must account for these nuances rather than hardcoding naive causal relationships:

태양 이벤트와 자기폭풍에 대한 여러 일반적 가정은 지나치게 단순화되어 있습니다. 모델 아키텍처와 학습은 단순한 인과 관계를 하드코딩하는 대신 이러한 미묘함을 고려해야 합니다:

**Flare ≠ geomagnetic storm / 플레어 ≠ 자기폭풍**: Solar flares emit electromagnetic radiation (arrives in 8 minutes) and sometimes energetic particles (SEPs), but **neither directly drives ap30**. It is the CME — often but not always associated with a flare — whose magnetic structure interacts with Earth's magnetosphere. A flare without an Earth-directed CME will not raise ap30. The model must learn this distinction from data rather than treating all flares as storm precursors.

태양 플레어는 전자기 복사(8분 만에 도달)와 때때로 고에너지 입자(SEP)를 방출하지만, **둘 다 ap30을 직접 올리지 않습니다**. 자기권과 상호작용하는 것은 CME이며, CME는 플레어와 종종 동반되지만 항상 그런 것은 아닙니다. 지구 방향 CME가 없는 플레어는 ap30을 올리지 않습니다. 모델은 모든 플레어를 자기폭풍 전조로 취급하는 대신 데이터로부터 이 구분을 학습해야 합니다.

**The Bz problem — the biggest uncertainty / Bz 문제 — 가장 큰 불확실성**: Even when a CME hits Earth, the resulting storm intensity depends critically on the CME's internal magnetic field orientation, specifically the **Bz (north-south) component**. Southward Bz → major storm; northward Bz → minimal impact. This Bz orientation **cannot be determined from solar images** — it is only measurable when the CME arrives at L1. This is the single largest unsolved problem in space weather forecasting. The model should learn to output probabilistic predictions (wider uncertainty bands) when images show an incoming CME but in-situ Bz is not yet available.

CME가 지구에 도달하더라도 자기폭풍의 강도는 CME 내부 자기장 방향, 특히 **Bz(남북) 성분**에 결정적으로 의존합니다. Bz 남향 → 대형 자기폭풍; Bz 북향 → 최소 영향. 이 Bz 방향은 **태양 영상에서 결정할 수 없고** CME가 L1에 도달해야만 측정됩니다. 이것이 우주기상 예보에서 가장 큰 미해결 문제입니다. 모델은 영상에서 CME 접근을 감지하지만 현장 Bz가 아직 없는 경우 확률적 예측(넓은 불확실성 범위)을 출력하도록 학습해야 합니다.

**Coronal holes > CMEs for ap30 predictability / ap30 예측에서 코로나 홀 > CME**: Coronal hole-driven high-speed streams (HSS) are actually more predictable than CMEs for ap30:

ap30 예측에서 코로나 홀 기반 고속 스트림(HSS)이 CME보다 실제로 더 예측 가능합니다:

| Factor | Coronal Hole / HSS | CME |
|---|---|---|
| Persistence / 지속성 | Weeks–months (stable) | One-time event |
| Predictability / 예측 가능성 | High — CH area/position correlates with HSS arrival and intensity | Low — Bz unknown until arrival |
| Recurrence / 반복성 | 27-day solar rotation period | Sporadic |
| Contribution to ap30 / ap30 기여 | ~60–70% of moderate activity | ~dominant for major storms (G3+) |
| EUV detectability / EUV 감지 | Excellent (dark regions in AIA 193) | Indirect (post-eruption dimming) |

Therefore, the first image integration should prioritize **EUV (AIA 193/211) for coronal hole detection**, which has both the highest signal-to-noise ratio and the most direct causal pathway to ap30.

따라서 첫 번째 영상 통합은 가장 높은 신호 대 잡음비와 ap30에 대한 가장 직접적인 인과 경로를 모두 가진 **코로나 홀 감지를 위한 EUV(AIA 193/211)**를 우선시해야 합니다.

**Variable transit time / 가변적 이동 시간**: The time between a solar event and its Earth impact is not fixed — it depends on solar wind speed (~300–800 km/s for ambient wind, ~500–3000 km/s for CMEs). The model must learn this variable lag rather than using a fixed offset. This is a key advantage of cross-attention over simple temporal concatenation.

태양 이벤트와 지구 영향 사이의 시간은 고정되지 않으며, 태양풍 속도에 따라 달라집니다(배경풍 ~300-800 km/s, CME ~500-3000 km/s). 모델은 고정 오프셋을 사용하는 대신 이 가변적 지연을 학습해야 합니다. 이것이 단순 시간 결합 대비 교차 어텐션의 핵심 이점입니다.

#### B3. Image types and their roles / 영상 유형 및 역할

| Image Type | Source | Key Information | Lead Time | Priority |
|---|---|---|---|---|
| **EUV (AIA 193, 211)** | SDO/AIA | Coronal holes (CH), active regions (AR) → high-speed stream (HSS) and flare sources | 2–4 days (CH→HSS) | **High** — CHs are the primary driver of recurrent geomagnetic activity |
| **Magnetogram (HMI)** | SDO/HMI | Photospheric magnetic field topology → flux emergence, AR complexity | 1–3 days | **High** — magnetic complexity predicts eruptive potential |
| **Coronagraph (LASCO C2/C3)** | SOHO | CME detection, speed, direction, mass | 1–3 days (CME transit) | **Medium** — critical for major storm prediction, but sparse events |
| **EUV (AIA 304, 171)** | SDO/AIA | Chromospheric/transition region dynamics → flare precursors | Hours–1 day | **Lower** — more relevant for flare prediction than ap30 |
| **STEREO (EUVI, COR)** | STEREO-A | Far-side/limb observations → Earth-directed CME confirmation | 1–4 days | **Medium** — valuable for CME directionality |

| 영상 유형 | 소스 | 핵심 정보 | 선행 시간 | 우선순위 |
|---|---|---|---|---|
| **EUV (AIA 193, 211)** | SDO/AIA | 코로나 홀(CH), 활동 영역(AR) → 고속 태양풍 스트림(HSS) 및 플레어 소스 | 2~4일 (CH→HSS) | **높음** — CH가 반복적 자기권 활동의 주요 원인 |
| **자기장 영상 (HMI)** | SDO/HMI | 광구 자기장 토폴로지 → 자속 출현, AR 복잡도 | 1~3일 | **높음** — 자기 복잡도가 분출 가능성 예측 |
| **코로나그래프 (LASCO C2/C3)** | SOHO | CME 감지, 속도, 방향, 질량 | 1~3일 (CME 이동) | **중간** — 대형 자기폭풍 예측에 중요하나 희소 이벤트 |
| **EUV (AIA 304, 171)** | SDO/AIA | 채층/천이 영역 역학 → 플레어 전조 | 수시간~1일 | **낮음** — ap30보다 플레어 예측에 더 관련 |
| **STEREO (EUVI, COR)** | STEREO-A | 반대편/림 관측 → 지구 방향 CME 확인 | 1~4일 | **중간** — CME 방향성 판단에 유용 |

#### B4. Recommended architecture / 권장 아키텍처

The existing fusion module (`src/networks/fusion.py`) uses ConvLSTM + cross-modal attention as a prototype. A phased approach is recommended:

기존 fusion 모듈(`src/networks/fusion.py`)은 프로토타입으로 ConvLSTM + 교차 모달 어텐션을 사용합니다. 단계적 접근을 권장합니다:

**Phase 1: Minimum viable validation with existing architecture / 기존 아키텍처로 최소 검증**

Use the current ConvLSTM + fusion structure with AIA 193 + HMI (2 channels), 64×64 resolution, 6h cadence, 7 days of imagery. Baseline: in12h_out24h_gnn_transformer (current best for out24h, R2=0.468). Goal: confirm whether images improve out24h R2.

현재 ConvLSTM + fusion 구조를 AIA 193 + HMI(2채널), 64×64 해상도, 6시간 간격, 7일 영상으로 사용합니다. 기준선: in12h_out24h_gnn_transformer(현재 out24h 최고, R2=0.468). 목표: 영상이 out24h R2를 개선하는지 확인합니다.

**Phase 2: Vision encoder upgrade / 비전 인코더 업그레이드**

Replace ConvLSTM with a pretrained Vision Transformer (ViT) or domain-specific solar image encoder. Increase resolution to 128–256. Add per-wavelength independent encoding → cross-channel attention.

ConvLSTM을 사전 학습된 ViT 또는 태양 영상 특화 인코더로 교체합니다. 해상도 128–256으로 확대. 파장별 독립 인코딩 → 교차 채널 어텐션 추가.

```
Current:  SDO images (3×28×64×64) → ConvLSTM → (128,)
Proposed: SDO images (C×T×256×256) → ViT encoder → temporal sequence of patch tokens
```

**Phase 3: Cross-modal fusion with learned transit time / 이동 시간 학습을 포함한 교차 모달 융합**

Replace simple temporal alignment with cross-attention that includes **physical time offset encoding**: each image-timeseries attention pair encodes the absolute time difference (e.g., "this image was taken 72 hours before this in-situ measurement"), allowing the model to learn velocity-dependent transit times.

단순 시간 정렬을 **물리적 시간 오프셋 인코딩**을 포함한 교차 어텐션으로 교체합니다: 각 영상-시계열 어텐션 쌍이 절대 시간 차이(예: "이 영상은 이 현장 측정보다 72시간 전에 촬영됨")를 인코딩하여, 모델이 속도 의존적 이동 시간을 학습할 수 있게 합니다.

**Phase 4: GNN integration for multi-modal graph / 다중 모달 그래프를 위한 GNN 통합**

Extend the current 8-node GNN to include image-derived nodes:

현재 8노드 GNN을 영상 유래 노드를 포함하도록 확장합니다:

```
Current GNN nodes (8):    [v, np, t, bx, by, bz, bt, ap30]
Extended GNN nodes (11+): [v, np, t, bx, by, bz, bt, ap30, CH_area, AR_flux, CME_flag, ...]
```

Image features become additional graph nodes with learned edges to solar wind parameters, naturally modeling physical relationships: coronal hole area → solar wind speed, active region flux → CME probability → Bz disturbance.

영상 특징이 태양풍 매개변수에 대한 학습된 엣지를 가진 추가 그래프 노드가 됩니다. 물리적 관계를 자연스럽게 모델링합니다: 코로나 홀 면적 → 태양풍 속도, 활동 영역 자속 → CME 확률 → Bz 교란.

#### B5. Training strategy / 학습 전략

**Auxiliary tasks for image encoder pretraining / 영상 인코더 사전 학습을 위한 보조 태스크**: Pretrain the image encoder with physically meaningful auxiliary tasks (CH area regression, CME binary detection, AR classification) before end-to-end training. This guides the encoder to extract physically relevant features rather than arbitrary image statistics.

영상 인코더를 end-to-end 학습 전에 물리적으로 의미 있는 보조 태스크(CH 면적 회귀, CME 이진 감지, AR 분류)로 사전 학습합니다. 이는 인코더가 임의의 영상 통계가 아닌 물리적으로 관련된 특징을 추출하도록 유도합니다.

**Staged training / 단계적 학습**:
1. Pretrain image encoder on auxiliary tasks / 보조 태스크로 영상 인코더 사전 학습
2. Initialize in-situ branch from current best model weights / 현재 최고 모델 가중치로 현장 브랜치 초기화
3. End-to-end fine-tuning with both modalities / 양 모달리티로 end-to-end 미세 조정

#### B6. Key considerations / 핵심 고려 사항

**Resolution vs. compute tradeoff / 해상도 대 연산 트레이드오프**: Current config uses 64×64, but CH/AR identification may require 128–256. ViT with patch size 16 on 256×256 yields 256 tokens — manageable with efficient attention.

**해상도 대 연산 트레이드오프**: 현재 64×64이지만 CH/AR 식별에 128-256이 필요할 수 있습니다. 256×256에 패치 16의 ViT는 256 토큰 — 효율적 어텐션으로 관리 가능.

**Image cadence / 영상 간격**: 6h cadence (4/day) is adequate for CH evolution. For CME detection, 1–2h cadence is better. Consider event-triggered sampling for coronagraph data.

**영상 간격**: 6시간 간격(일 4장)은 CH 진화에 적합. CME 감지에는 1-2시간이 더 좋음. 코로나그래프 데이터는 이벤트 기반 샘플링 고려.

**Why images solve the "long input" problem / 영상이 "긴 입력" 문제를 해결하는 이유**: Our analysis shows extending in-situ inputs beyond 18h hurts performance because old solar wind data is noise. But images provide fundamentally different information — what is happening **on the Sun** 1–4 days before Earth impact. A coronal hole visible in AIA 193 today will produce a high-speed stream in 2–4 days. This is not noise — it is genuinely predictive information that in-situ data cannot provide. **The path to better long-range prediction is not "more hours of L1 data" but "adding solar disk imagery."**

분석 결과 현장 입력을 18h 이상 늘리면 성능이 저하됩니다 — 오래된 태양풍 데이터는 노이즈이기 때문입니다. 하지만 영상은 근본적으로 다른 정보를 제공합니다 — 지구 영향 1~4일 전 **태양에서** 무슨 일이 일어나고 있는지 보여줍니다. 오늘 AIA 193에서 보이는 코로나 홀은 2~4일 후 고속 스트림을 생성합니다. 이는 노이즈가 아니라 현장 데이터가 제공할 수 없는 진정한 예측 정보입니다. **장기 예측 개선의 경로는 "더 많은 L1 데이터"가 아니라 "태양 원면 영상 추가"입니다.**

#### B7. Current status and prerequisites / 현재 상태 및 선행 요건

| Item | Status | Next Step |
|---|---|---|
| SDO FITS download pipeline (setup-sw-db) | Exists but **SDO FITS files not yet registered in DB** | Register FITS files, build image table |
| Image preprocessing (resize, normalize) | HDF5 reader exists in pipeline | Adapt for FITS-based workflow |
| ConvLSTM encoder | Implemented | Usable for Phase 1 |
| Cross-modal fusion | Prototype exists | Usable for Phase 1 |
| ViT encoder | Not implemented | Phase 2 |
| Auxiliary task training | Not implemented | Phase 2–3 |
| GNN multi-modal extension | Not implemented | Phase 4 |

| 항목 | 상태 | 다음 단계 |
|---|---|---|
| SDO FITS 다운로드 파이프라인 (setup-sw-db) | 존재하지만 **SDO FITS 파일이 아직 DB에 등록되지 않음** | FITS 파일 등록, 영상 테이블 구축 |
| 영상 전처리 (리사이즈, 정규화) | HDF5 reader가 pipeline에 존재 | FITS 기반 워크플로에 맞게 적응 |
| ConvLSTM 인코더 | 구현됨 | Phase 1에 사용 가능 |
| 교차 모달 융합 | 프로토타입 존재 | Phase 1에 사용 가능 |
| ViT 인코더 | 미구현 | Phase 2 |
| 보조 태스크 학습 | 미구현 | Phase 2–3 |
| GNN 다중 모달 확장 | 미구현 | Phase 4 |

**Immediate blocker: SDO FITS files must be registered in the database before any image-based training can begin. This is a setup-sw-db task.**

**즉시 해결 필요 사항: 영상 기반 학습을 시작하려면 SDO FITS 파일을 데이터베이스에 등록해야 합니다. 이것은 setup-sw-db 작업입니다.**

---

## Changelog / 변경 이력

| Date / 날짜 | Description / 내용 |
|------------|-------------------|
| 2025-04-08 | Initialize experiment matrix. 81 experiments with identical parameters / 실험 매트릭스 초기화. 동일 파라미터 81개 |
| 2025-04-08 | Remove hp30 from input (23->22 vars). Dynamic GNN node groups / hp30 제거, GNN 동적 노드 그룹 |
| 2025-04-08 | Add out6h (6h/12h/24h). Naming convention update / out6h 추가, 네이밍 통일 |
| 2025-04-09 | Record all 81 results. GNN+Transformer ranks #1 / 81개 결과 기록. GNN+Transformer 1위 확인 |
| 2026-04-15 | Expand to 216 experiments (6 inputs x 4 outputs x 9 models). Add in6h, in12h, in18h inputs and out18h output. Full re-analysis with composite ranking / 216개 실험으로 확장. in6h, in12h, in18h 입력과 out18h 출력 추가. 종합 순위 기반 전체 재분석 |
