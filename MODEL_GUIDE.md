# Model Architecture Guide / 모델 아키텍처 가이드

This document describes the architecture, design principles, and hyperparameters of the 9 time series prediction models available in the regression-sw project.

이 문서는 regression-sw 프로젝트에서 사용 가능한 9가지 시계열 예측 모델의 아키텍처, 설계 원리, 하이퍼파라미터를 설명합니다.

---

## 1. Overview / 개요

All models follow the same interface:

모든 모델은 동일한 인터페이스를 따릅니다:

```
입력: (batch, input_seq_len, num_input_vars)  — e.g., (128, 96, 22)
출력: (batch, target_seq_len, num_target_vars) — e.g., (128, 24, 1)
```

The internal structure follows the **Encoder → Global Pooling → Regression Head** pattern:

내부 구조는 **Encoder → Global Pooling → Regression Head** 패턴입니다:

```
입력 시계열 → [Encoder] → (batch, d_model) → [Regression Head] → (batch, target_len × target_vars)
```

The Regression Head is identical across all models:

Regression Head는 모든 모델에서 동일합니다:
```
Linear(d_model, d_model//2) → ReLU → Dropout → Linear(d_model//2, target_len × target_vars) → Reshape
```

---

## 2. Model Classification / 모델 분류

### 2.1 Standalone Encoder Models (5 types) / 단독 Encoder 모델 (5종)

Directly encodes the input time series to produce a `(batch, d_model)` representation.

입력 시계열을 직접 인코딩하여 `(batch, d_model)` 표현을 생성.

### 2.2 GNN Composite Models (4 types) / GNN 복합 모델 (4종)

Maps input variables to graph nodes → learns inter-variable relationships with GCN → learns temporal patterns with a Temporal Encoder.

입력 변수를 그래프 노드로 매핑 → GCN으로 변수 간 관계 학습 → Temporal Encoder로 시간 패턴 학습.

---

## 3. Standalone Encoder Models / 단독 Encoder 모델

### 3.1 Linear (`model_type: "linear"`)

**Role / 역할**: MLP Baseline. The simplest model, serving as a lower bound for other models.

MLP Baseline. 가장 단순한 모델로 다른 모델의 하한선 역할.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, vars)
  → Flatten: (batch, seq_len × vars)
  → Linear(seq_len × vars, d_model) → ReLU → Dropout
  → Linear(d_model, d_model) → ReLU → Dropout
  → (batch, d_model)
```

**Characteristics / 특징**:
- Does not consider temporal structure (treats the entire input as a single vector)
- 시간적 구조를 고려하지 않음 (전체를 하나의 벡터로 취급)
- Parameter count grows proportionally with input length
- 파라미터 수가 입력 길이에 비례하여 증가
- Prone to overfitting, but training is stable due to simplicity
- 과적합 경향이 있으나 단순함으로 인해 학습이 안정적

**Config / 설정**: `configs/base.yaml` → `model.baseline_dropout`

---

### 3.2 Transformer (`model_type: "transformer"`)

**Role / 역할**: Self-attention based time series encoding. Captures global temporal dependencies.

Self-attention 기반 시계열 인코딩. 전역적 시간 의존성 포착.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, vars)
  → Linear(vars, d_model)                    [Input Projection]
  → + Sinusoidal Positional Encoding         [Positional info injection / 위치 정보 주입]
  → TransformerEncoderLayer × num_layers     [Multi-Head Self-Attention + FFN]
  → AdaptiveAvgPool1d(1)                     [Global Pooling]
  → Linear(d_model, d_model)                 [Output Projection]
  → (batch, d_model)
```

**Characteristics / 특징**:
- Applies attention to all timestep pairs → directly captures long-range dependencies
- 모든 timestep 쌍에 attention 적용 → 장거리 의존성 직접 포착
- Computational complexity O(seq_len²) — cost increases for long sequences
- 연산 복잡도 O(seq_len²) — 긴 시퀀스에서 비용 증가
- Model interpretability through attention maps
- Attention map으로 모델 해석 가능

**Key Config / 주요 설정**:
| Parameter / 파라미터 | Default / 기본값 | Description / 설명 |
|---------|-------|------|
| `transformer_nhead` | 4 | Number of attention heads / Attention head 수 |
| `transformer_num_layers` | 2 | Number of encoder layers / Encoder layer 수 |
| `transformer_dim_feedforward` | 256 | FFN hidden dimension / FFN 히든 차원 |
| `transformer_dropout` | 0.1 | Dropout rate / Dropout 비율 |

---

### 3.3 TCN (`model_type: "tcn"`)

**Role / 역할**: Dilated causal convolution based time series encoding. Gradual expansion of local patterns.

Dilated causal convolution 기반 시계열 인코딩. 로컬 패턴의 점진적 확장.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, vars)
  → Linear(vars, channels[0])               [Input Projection]
  → Transpose → (batch, channels[0], seq_len)
  → TemporalBlock × num_layers              [Dilated Causal Conv + Residual]
      dilation: 1, 2, 4, ...  (2^i)
  → AdaptiveAvgPool1d(1)                    [Global Pooling]
  → Linear(channels[-1], d_model)           [Output Projection]
  → (batch, d_model)
```

**TemporalBlock Structure / TemporalBlock 구조**:
```
x → Conv1d(padding=p) → trim(-p) → ReLU → Dropout
  → Conv1d(padding=p) → trim(-p) → ReLU → Dropout
  → + residual(x)
  → ReLU
```

**Characteristics / 특징**:
- Causal padding (`Conv1d(padding=p)` + `trim(-p)`) blocks future information
- Causal padding (`Conv1d(padding=p)` + `trim(-p)`)으로 미래 정보 차단
- Receptive field = 1 + 2 × (kernel-1) × Σ(2^i) — default 29 timesteps (14.5h)
- Receptive field = 1 + 2 × (kernel-1) × Σ(2^i) — 기본값 29 timesteps (14.5h)
- Computational complexity O(seq_len) — more efficient than Transformer
- 연산 복잡도 O(seq_len) — Transformer보다 효율적
- Weight normalization applied
- Weight normalization 적용

**Key Config / 주요 설정**:
| Parameter / 파라미터 | Default / 기본값 | Description / 설명 |
|---------|-------|------|
| `tcn_channels` | [64, 128, 256] | Channel count per layer / 각 layer의 채널 수 |
| `tcn_kernel_size` | 3 | Convolution kernel size / Convolution 커널 크기 |
| `tcn_dropout` | 0.1 | Dropout rate / Dropout 비율 |

---

### 3.4 PatchTST (`model_type: "patchtst"`)

**Role / 역할**: Transformer that tokenizes time series into patch units. Computational efficiency and local pattern preservation.

시계열을 patch 단위로 토큰화한 Transformer. 연산 효율과 로컬 패턴 보존.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, vars)
  → PatchEmbedding:
      Sliding window (patch_len, stride) → patch segmentation / 패치 분할
      Linear(patch_len × vars, d_model)     [Patch → Token / 패치 → 토큰]
  → + Learnable Positional Embedding
  → TransformerEncoder × num_layers
  → AdaptiveAvgPool1d(1)                    [Global Pooling]
  → Linear(d_model, d_model)               [Output Projection]
  → (batch, d_model)
```

**Patch Example / 패치 예시** (in2d, seq_len=96, patch=16, stride=8):
```
96 timesteps → 11 patch tokens (vs 96 tokens for Transformer)
96 timesteps → 11개 패치 토큰 (vs Transformer의 96개 토큰)
Attention computation: O(11²) vs O(96²) ≈ 76x reduction / 76배 감소
```

**Characteristics / 특징**:
- Groups adjacent timesteps so a single token directly represents local patterns
- 인접 timestep들을 묶어 로컬 패턴을 하나의 토큰이 직접 표현
- Learnable positional embedding (instead of sinusoidal)
- Learnable positional embedding (sinusoidal 대신)
- More efficient than Transformer for long input sequences
- 긴 입력 시퀀스에서 Transformer보다 효율적

**Key Config / 주요 설정**:
| Parameter / 파라미터 | Default / 기본값 | Description / 설명 |
|---------|-------|------|
| `patch_len` | 16 | Patch length (timesteps) / 패치 길이 (timesteps) |
| `patch_stride` | 8 | Stride between patches (overlap = 8) / 패치 간 stride (overlap = 8) |
| `patchtst_dropout` | 0.1 | Dropout rate / Dropout 비율 |

**Reference / 참고 논문**: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)

---

### 3.5 TimesNet (`model_type: "timesnet"`)

**Role / 역할**: Detects periods via FFT, converts 1D time series to 2D, then applies Inception Conv.

FFT로 주기를 탐지하여 1D 시계열을 2D로 변환 후 Inception Conv 적용.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, vars)
  → Linear(vars, d_model)                   [Input Projection]
  → TimesBlock × num_blocks:
      FFT → top-k period detection / top-k 주기 탐지
      For each period / 각 주기별:
        1D→2D reshape: (seq_len,) → (period, seq_len/period)
        InceptionBlock1 (multi-scale 2D Conv) → GELU
        InceptionBlock2 (multi-scale 2D Conv)
        2D→1D reshape back
      Adaptive aggregation (FFT amplitude-based weighted sum / FFT 진폭 기반 가중합)
      + Residual
  → Cross-variable Self-Attention            [Channel mixing / 채널 혼합]
  → AdaptiveAvgPool1d(1)                    [Global Pooling]
  → Linear(d_model, output_dim)             [Output Projection]
  → (batch, output_dim)
```

**Characteristics / 특징**:
- The original TimesNet is channel-independent — this implementation adds cross-variable self-attention
- 원본 TimesNet은 채널 독립 — 본 구현에서는 cross-variable self-attention 추가
- Strong with periodic patterns (e.g., 27-day solar rotation)
- 주기적 패턴에 강점 (태양 자전 27일 등)
- **Weak with aperiodic sudden events (storms)** — ranked last across all experimental ranges
- **비주기적 돌발 이벤트(폭풍)에 약점** — 실험에서 전 구간 최하위

**Key Config / 주요 설정**:
| Parameter / 파라미터 | Default / 기본값 | Description / 설명 |
|---------|-------|------|
| `timesnet_d_model` | 64 | Internal feature dimension / 내부 feature 차원 |
| `timesnet_d_ff` | 128 | Inception hidden dimension / Inception 히든 차원 |
| `timesnet_num_blocks` | 2 | TimesBlock stack count / TimesBlock 스택 수 |
| `timesnet_top_k` | 3 | Number of top FFT periods / FFT 상위 주기 수 |
| `timesnet_num_kernels` | 3 | Inception Conv parallel branches / Inception Conv 병렬 브랜치 수 |
| `timesnet_cross_variable` | true | Enable cross-variable attention / Cross-variable attention 활성화 |

**Reference / 참고 논문**: Wu et al., "TimesNet: Temporal 2D-Variation Modeling" (ICLR 2023)

---

## 4. GNN Composite Models / GNN 복합 모델

### 4.0 GNN Common Structure / GNN 공통 구조 (`model_type: "gnn"`)

All GNN models share the same **GNNEncoder** structure; only the temporal encoder differs.

모든 GNN 모델은 동일한 **GNNEncoder** 구조를 공유하며, temporal encoder만 다릅니다.

**Architecture / 아키텍처**:
```
입력 (batch, seq_len, 22)
  → Variable grouping: 22 variables → 8 graph nodes / 변수 그룹화: 22개 변수 → 8개 그래프 노드
      v: [v_avg, v_min, v_max]     → Node 0 / 노드 0 (3dim / 3차원)
      np: [np_avg, np_min, np_max]  → Node 1 / 노드 1 (3dim / 3차원)
      t: [t_avg, t_min, t_max]     → Node 2 / 노드 2 (3dim / 3차원)
      bx: [bx_avg, bx_min, bx_max] → Node 3 / 노드 3 (3dim / 3차원)
      by: [by_avg, by_min, by_max]  → Node 4 / 노드 4 (3dim / 3차원)
      bz: [bz_avg, bz_min, bz_max] → Node 5 / 노드 5 (3dim / 3차원)
      bt: [bt_avg, bt_min, bt_max]  → Node 6 / 노드 6 (3dim / 3차원)
      ap30: [ap30]                  → Node 7 / 노드 7 (1dim / 1차원)
  → Per-node Linear projection → (batch, seq_len, 8, node_feat_dim)
  → Per-timestep GCN × num_layers:
      Adaptive Adjacency: A = softmax(relu(E1 · E2ᵀ))
      Message passing: X' = A @ X @ W
  → Flatten nodes: (batch, seq_len, 8 × gcn_hidden_dim)
  → [Temporal Encoder] → (batch, d_model)
```

**Adaptive Adjacency Matrix / 적응적 인접 행렬**:
- Automatically learns adjacency matrix from learnable node embeddings `E1`, `E2` (8 × embed_dim)
- 학습 가능한 노드 임베딩 `E1`, `E2` (8 × embed_dim)로 인접 행렬 자동 학습
- Learned graph can be visualized via `adjacency_matrix` property
- `adjacency_matrix` property로 학습된 그래프 시각화 가능
- Physical meaning: automatically captures strong Bz↔ap30 connections, V↔T correlations, etc.
- 물리적 의미: Bz↔ap30 강한 연결, V↔T 상관 등을 자동 포착

**Node Grouping / 노드 그룹화**:
- Defined in `configs/base.yaml` under `gnn_variable_groups` (dynamic, config-based)
- `configs/base.yaml`의 `gnn_variable_groups`에서 정의 (동적, config 기반)
- 3-step validation at model creation: variable existence, coverage, order consistency
- 모델 생성 시 3단계 검증: 변수 존재, 커버리지, 순서 일치
- Only config modification needed when input variables change (no code changes required)
- 입력 변수가 변경되면 config만 수정하면 됨 (코드 수정 불필요)

**GNN Common Config / GNN 공통 설정**:
| Parameter / 파라미터 | Default / 기본값 | Description / 설명 |
|---------|-------|------|
| `gnn_node_feature_dim` | 32 | Node feature dimension / 노드 feature 차원 |
| `gnn_gcn_hidden_dim` | 64 | GCN hidden dimension / GCN 히든 차원 |
| `gnn_num_gcn_layers` | 2 | Number of GCN layers / GCN layer 수 |
| `gnn_dropout` | 0.1 | Dropout rate / Dropout 비율 |
| `gnn_node_embed_dim` | 16 | Embedding dim for adaptive adjacency / 적응적 인접 행렬용 임베딩 차원 |

---

### 4.1 GNN+Transformer (`gnn_temporal_type: "transformer"`)

Encodes GCN output temporally with Transformer Encoder. **Ranked 1st across all experiments.**

GCN 출력을 Transformer Encoder로 시간 인코딩. **전체 실험에서 1위**.

**Temporal part / Temporal 부분**: Same as §3.2 Transformer (Positional Encoding + TransformerEncoder + Global Pool)

§3.2 Transformer와 동일 (Positional Encoding + TransformerEncoder + Global Pool)

**Why best performance / 왜 최고 성능인가**:
- GCN explicitly learns **physical inter-variable relationships** (Bz→ap30, etc.)
- GCN이 **변수 간 물리적 관계**를 명시적으로 학습 (Bz→ap30 등)
- Transformer captures **global temporal dependencies**
- Transformer가 **전역 시간 의존성**을 포착
- Separation of the two roles makes each learning process more efficient
- 두 역할의 분리가 각각의 학습을 효율화
- 8-node graph structure acts as a structural regularizer to suppress overfitting
- 8-node 그래프 구조가 structural regularizer로 과적합 억제

---

### 4.2 GNN+TCN (`gnn_temporal_type: "tcn"`)

Encodes GCN output temporally with TCN.

GCN 출력을 TCN으로 시간 인코딩.

**Temporal part / Temporal 부분**: Same structure as §3.3 TCN.

§3.3 TCN과 동일 구조.

---

### 4.3 GNN+BiLSTM (`gnn_temporal_type: "bilstm"`)

Encodes GCN output temporally with Bidirectional LSTM.

GCN 출력을 Bidirectional LSTM으로 시간 인코딩.

**Temporal part / Temporal 부분**:
```
GCN output (batch, seq_len, d_model)
  → LSTM(forward) + LSTM(backward)
  → (batch, seq_len, hidden_size × 2)
  → AdaptiveAvgPool1d(1)
  → Linear(hidden_size × 2, d_model)
```

**Characteristics / 특징**:
- Bidirectional processing utilizes both past and future context
- 양방향 처리로 과거/미래 컨텍스트 모두 활용
- Sequential processing makes training slower than Transformer/TCN
- 순차 처리로 Transformer/TCN보다 훈련 느림
- No causality constraint since input spans are all past data
- 입력 구간이 모두 과거 데이터이므로 인과성 제약 없음

**Config / 설정**: `bilstm_hidden_size: 128`, `bilstm_num_layers: 2`

**Reference / 참고 논문**: Abduallah et al., "Prediction of the SYM-H Index" (Space Weather, 2024) — Achieved SOTA for solar wind → SYM-H prediction using GNN+BiLSTM.

GNN+BiLSTM으로 태양풍→SYM-H 예측에서 SOTA.

---

### 4.4 GNN+PatchTransformer (`gnn_temporal_type: "patch_transformer"`)

Encodes GCN output temporally using the PatchTST approach. **Ranked 2nd overall.**

GCN 출력을 PatchTST 방식으로 시간 인코딩. **전체 2위**.

**Temporal part / Temporal 부분**: GCN output → PatchEmbedding → Transformer (same principle as §3.4)

GCN output → PatchEmbedding → Transformer (§3.4와 동일 원리)

**Characteristics / 특징**:
- Applies patch-based efficient attention to GCN's graph features
- GCN의 그래프 features에 patch 기반 효율적 attention 적용
- Approaches GNN+Transformer for 6h prediction (0.2200 vs 0.2178)
- 6h 예측에서 GNN+Transformer에 근접 (0.2200 vs 0.2178)

---

## 5. Model Selection Guide / 모델 선택 가이드

| Purpose / 목적 | Recommended Model / 권장 모델 | Reason / 이유 |
|------|---------|------|
| **Best performance / 최고 성능** | GNN+Transformer | Ranked 1st across all ranges, separate learning of variable relationships + temporal patterns / 전 구간 1위, 변수 관계 + 시간 패턴 분리 학습 |
| **Fast training / 빠른 훈련** | TCN or Linear / TCN 또는 Linear | O(n) computation, fewer parameters / O(n) 연산, 파라미터 적음 |
| **Interpretability / 해석 가능성** | GNN+Transformer | Learned graph + Attention map / 학습된 그래프 + Attention map |
| **Long input sequences / 긴 입력 시퀀스** | GNN+PatchTST | Reduces token count via patches, efficient attention / Patch로 토큰 수 감소, 효율적 attention |
| **Baseline comparison / Baseline 비교** | Linear | Simplest, serves as lower bound / 가장 단순, 하한선 역할 |

---

## 6. How to Add a New Model / 모델 추가 방법

To add a new model, follow these steps:

새 모델을 추가하려면:

1. Implement `NewEncoder` class in `src/networks.py`
   - `forward(x) → (batch, d_model)` interface

   `src/networks.py`에 `NewEncoder` 클래스 구현
   - `forward(x) → (batch, d_model)` 인터페이스

2. Implement `NewOnlyModel` class
   - `forward(solar_wind_input, image_input=None, return_features=False)`
   - Returns: `(batch, target_seq_len, num_target_vars)` or `(output, features, None)`

   `NewOnlyModel` 클래스 구현
   - `forward(solar_wind_input, image_input=None, return_features=False)`
   - 반환: `(batch, target_seq_len, num_target_vars)` 또는 `(output, features, None)`

3. Add `elif model_type == "new":` branch to `create_model()`

   `create_model()`에 `elif model_type == "new":` 분기 추가

4. Add `"new"` to the model_type dispatch tuple in `src/trainers.py`

   `src/trainers.py`의 model_type dispatch 튜플에 `"new"` 추가

5. Add hyperparameter defaults to `configs/base.yaml`

   `configs/base.yaml`에 하이퍼파라미터 기본값 추가

6. Create experiment configs: `configs/in{1,2,3}d_out{6,12,24}h_new.yaml`

   실험 config 생성: `configs/in{1,2,3}d_out{6,12,24}h_new.yaml`

To add as a GNN temporal encoder:

GNN temporal encoder로 추가하려면:

1. Add `elif temporal_type == "new":` branch to `GNNEncoder.__init__`

   `GNNEncoder.__init__`에 `elif temporal_type == "new":` 분기 추가

2. Add the same branch to `GNNEncoder.forward`

   `GNNEncoder.forward`에 동일 분기 추가

3. Pass required parameters to `GNNOnlyModel`

   `GNNOnlyModel`에 필요한 파라미터 전달
