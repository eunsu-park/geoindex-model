# Analysis Guide / 분석 가이드

This document describes the analysis tools available in the regression-sw project — their purpose, methodology, and usage.

이 문서는 regression-sw 프로젝트에서 사용 가능한 분석 도구의 목적, 방법론, 사용법을 설명합니다.

---

## 1. Overview / 개요

Analysis tools are divided into three categories:

분석 도구는 세 가지 범주로 나뉩니다:

| Category / 범주 | Scripts | Description / 설명 |
|----------------|---------|-------------------|
| **Primary Analysis / 주요 분석** | `run_attention.py`, `run_mcd.py`, `run_saliency.py` | Model inference-based analysis requiring checkpoint + dataloader / 체크포인트 + 데이터로더 필요 |
| **Post-hoc Evaluation / 사후 평가** | `evaluate_storm_performance.py`, `compare_predictions.py`, `evaluate_mcd.py`, `visualize_gnn_graph.py` | Process existing results (NPZ/checkpoint) without re-running inference / 기존 결과를 처리, 재추론 불필요 |
| **Utility Modules / 유틸리티** | `attention_analysis.py`, `saliency_maps.py` | Core extraction classes used by primary scripts / 주요 스크립트가 사용하는 핵심 추출 클래스 |

### Shell Script Runners / 셸 스크립트 실행기

All primary analysis scripts can be run in parallel via shell scripts:

모든 주요 분석 스크립트는 셸 스크립트로 병렬 실행 가능:

```bash
./attention.sh --config-file configs.txt --epoch best --max-jobs 8
./mcd.sh --filter gnn_transformer --epoch best
./saliency.sh --config-file configs.txt --epoch 10
```

Common options (shared by all runners / 모든 실행기 공통 옵션):
- `--config-file FILE`: Read config names from file / 파일에서 config 목록 읽기
- `--filter PATTERN`: Glob-based filtering / 패턴 필터링
- `--max-jobs N`: Max parallel processes (default 8) / 최대 병렬 수
- `--epoch EPOCH`: Checkpoint epoch (`best`, `final`, or number) / 체크포인트 에폭
- `--dry-run`: Print config list without running / 실행 없이 목록만 출력

---

## 2. Primary Analysis / 주요 분석

### 2.1 Attention Analysis / Attention 분석

**Script**: `analysis/run_attention.py`
**Shell**: `./attention.sh`

**What it does / 기능**:
Extracts attention weights from Transformer encoder layers. Shows which timesteps the model attends to when making predictions.

Transformer encoder layer에서 attention 가중치를 추출합니다. 모델이 예측 시 어떤 timestep에 주목하는지 보여줍니다.

**Methodology / 방법론**:
1. Load trained model in eval mode / 학습된 모델을 eval 모드로 로드
2. Manual forward pass through TransformerEncoder layers with `need_weights=True`
3. Extract per-layer attention: `(batch, num_heads, seq_len, seq_len)`
4. Compute temporal importance: incoming attention per timestep / timestep별 수신 attention 계산
5. Save NPZ per sample with attention weights and temporal importance

**Output / 출력**:
- `attention/best.zip → npz/{timestamp}.npz`: attention_weights, temporal_importance, predictions, targets
- Optional: heatmap, matrix, layer comparison plots

**Supported models / 지원 모델**: `transformer`, `gnn` (with transformer temporal)

**Usage / 사용법**:
```bash
# Single experiment / 단일 실험
python analysis/run_attention.py --config-name=in2d_out12h_gnn_transformer attention.epoch=best

# Parallel batch / 병렬 일괄
./attention.sh --filter gnn_transformer --epoch best
```

**Interpretation / 해석**:
- High attention at specific timesteps → model considers those time periods important for prediction
- 특정 timestep에 높은 attention → 모델이 해당 시간대를 예측에 중요하게 판단
- Can reveal if model focuses on recent history or captures long-range patterns
- 모델이 최근 이력에 집중하는지, 장거리 패턴을 포착하는지 확인 가능

---

### 2.2 Monte Carlo Dropout (MCD) / MCD 불확실성 추정

**Script**: `analysis/run_mcd.py` (alias: `analysis/monte_carlo_dropout.py`)
**Shell**: `./mcd.sh`

**What it does / 기능**:
Estimates prediction uncertainty by running multiple forward passes with dropout enabled. Produces mean prediction and standard deviation (uncertainty band).

Dropout을 활성화한 상태에서 여러 번 forward pass를 수행하여 예측 불확실성을 추정합니다. 평균 예측과 표준편차(불확실성 대역)를 생성합니다.

**Methodology / 방법론**:
1. Load trained model, enable dropout layers (`.train()` mode for Dropout only)
2. Run N forward passes per sample (default N=100) with different dropout masks
3. Compute mean and std across N predictions (denormalized to original scale)
4. Calculate coverage: fraction of ground truth within ±2σ band (~95.4% expected)

**Output / 출력**:
- `mcd/best.zip → npz/{timestamp}.npz`: mean, std, target, n_samples, coverage
- Optional: uncertainty band plots (input + prediction ± 2σ + ground truth)

**Supported models / 지원 모델**: All model types (any model with dropout layers)
**모든 모델 지원** (dropout layer가 있는 모든 모델)

**Usage / 사용법**:
```bash
python analysis/run_mcd.py --config-name=in2d_out12h_gnn_transformer mcd.epoch=best

./mcd.sh --config-file configs.txt --epoch best
```

**Interpretation / 해석**:
- Large std → high uncertainty (model is unsure) / 큰 std → 높은 불확실성
- Std increases during storm events → physically meaningful (storms are harder to predict)
- 폭풍 시 std 증가 → 물리적으로 타당 (폭풍은 예측이 더 어려움)
- Coverage ≈ 95% → well-calibrated uncertainty / 커버리지 ≈ 95% → 잘 보정된 불확실성

---

### 2.3 Saliency Analysis / Saliency 분석

**Script**: `analysis/run_saliency.py`
**Shell**: `./saliency.sh`
**Core module**: `analysis/saliency_maps.py`

**What it does / 기능**:
Computes gradient-based attribution maps (Grad-CAM, Integrated Gradients) for SDO image inputs. Identifies which image regions drive predictions.

SDO 이미지 입력에 대한 gradient 기반 속성 맵(Grad-CAM, Integrated Gradients)을 계산합니다. 예측에 기여하는 이미지 영역을 식별합니다.

**Methodology / 방법론**:
1. Forward pass + backward pass through ConvLSTM layers
2. Grad-CAM: gradient of output w.r.t. ConvLSTM feature maps
3. Integrated Gradients: interpolate from baseline to input, accumulate gradients
4. Per-channel importance ranking (AIA 193, AIA 211, HMI magnetogram)

**Output / 출력**:
- Per-sample directories with channel-level heatmaps, temporal importance plots
- `saliency/best.zip`

**Supported models / 지원 모델**: `convlstm`, `fusion` (requires SDO image modality)

> **Note / 참고**: Currently inactive for time series-only models (modalities.sdo=false). Will be used when SDO image modality is activated.
> 현재 시계열 전용 모델에서는 비활성. SDO 이미지 모달리티 활성화 시 사용.

---

## 3. Post-hoc Evaluation / 사후 평가

These scripts process existing results without re-running model inference.

이 스크립트들은 모델 재추론 없이 기존 결과를 처리합니다.

### 3.1 Storm Performance Evaluation / 폭풍 성능 평가

**Script**: `analysis/evaluate_storm_performance.py`

**What it does / 기능**:
Computes MAE, RMSE, bias for each NOAA G-Scale storm tier. Evaluates model performance on operationally critical storm periods.

NOAA G-Scale 폭풍 티어별 MAE, RMSE, bias를 계산합니다. 운영적으로 중요한 폭풍 구간의 모델 성능을 평가합니다.

**Storm tiers / 폭풍 티어**:
| Tier | ap30 Range | NOAA Scale |
|------|-----------|------------|
| none | 0–29 | Kp < 5 |
| G1 | 30–49 | Minor storm |
| G2 | 50–99 | Moderate storm |
| G3+ | 100+ | Strong–Extreme |

**Usage / 사용법**:
```bash
python analysis/evaluate_storm_performance.py \
    --results-dir /path/to/results \
    --output-dir ./storm_analysis \
    --filter out12h
```

**Output / 출력**: `storm_metrics.csv`, `storm_summary.txt`

---

### 3.2 Prediction Comparison Plots / 예측 비교 플롯

**Script**: `analysis/compare_predictions.py`

**What it does / 기능**:
Overlays predictions from multiple models on the same event. Automatically selects top-k storm events and quiet events for comparison.

동일 이벤트에 대해 여러 모델의 예측을 오버레이합니다. 상위 k개 폭풍 이벤트와 정상 이벤트를 자동 선별합니다.

**Usage / 사용법**:
```bash
python analysis/compare_predictions.py \
    --results-dir /path/to/results \
    --config-base in2d_out12h \
    --top-k 5
```

**Output / 출력**: Per-event PNG plots with multi-model overlay + MAE annotations

---

### 3.3 MCD Uncertainty Evaluation / MCD 불확실성 평가

**Script**: `analysis/evaluate_mcd.py`

**What it does / 기능**:
Aggregates MCD results across all samples. Computes overall and storm-period 95% CI coverage, std-target correlation, and calibration diagrams.

전체 샘플에 대해 MCD 결과를 집계합니다. 전체/폭풍 구간 95% CI 커버리지, std-target 상관관계, 보정 다이어그램을 계산합니다.

**Usage / 사용법**:
```bash
python analysis/evaluate_mcd.py \
    --results-dir /path/to/results \
    --output-dir ./mcd_analysis \
    --filter out12h
```

**Output / 출력**: `mcd_coverage.csv`, calibration plots, std vs target scatter plots

---

### 3.4 GNN Graph Visualization / GNN 그래프 시각화

**Script**: `analysis/visualize_gnn_graph.py`

**What it does / 기능**:
Extracts and visualizes the adaptive adjacency matrix learned by GNN models. Validates whether the model captures physically meaningful variable relationships (e.g., Bz→ap30).

GNN 모델이 학습한 적응적 인접 행렬을 추출/시각화합니다. Bz→ap30 같은 물리적으로 의미 있는 변수 관계를 모델이 포착했는지 검증합니다.

**Usage / 사용법**:
```bash
python analysis/visualize_gnn_graph.py \
    --results-dir /path/to/results \
    --configs in2d_out6h_gnn_transformer,in2d_out12h_gnn_transformer \
    --epoch best
```

**Output / 출력**: 8×8 adjacency heatmaps, comparison plots, physical validation stats

**Interpretation / 해석**:
- Strong Bz→ap30 edge → model learned that southward IMF drives storms
- Bz→ap30 강한 엣지 → 모델이 남향 IMF가 폭풍을 유발함을 학습
- ap30→Bt strong edge → model uses IMF magnitude as primary predictor for ap30
- ap30→Bt 강한 엣지 → IMF 크기를 ap30 예측의 주요 경로로 사용

---

## 4. Utility Modules / 유틸리티 모듈

These are not run directly — they are imported by primary analysis scripts.

직접 실행하지 않으며, 주요 분석 스크립트가 import하여 사용합니다.

| Module / 모듈 | Used by / 사용처 | Description / 설명 |
|--------------|-----------------|-------------------|
| `attention_analysis.py` | `run_attention.py` | `AttentionExtractor` class — hooks into Transformer layers to extract attention weights |
| `saliency_maps.py` | `run_saliency.py` | `SaliencyExtractor` class — Grad-CAM, Integrated Gradients, occlusion sensitivity for SDO images |

---

## 5. Multi-modal Analysis (Future) / 멀티모달 분석 (향후)

These scripts are for multi-modal models (SDO images + OMNI time series). Currently inactive as SDO modality is disabled.

SDO 이미지 + OMNI 시계열을 사용하는 멀티모달 모델용입니다. 현재 SDO 모달리티 비활성으로 미사용.

| Script | Description / 설명 |
|--------|-------------------|
| `ablation_analysis.py` | Zeros out SDO/OMNI encoders separately to measure modality contribution / 모달리티별 기여도 측정 |
| `cross_modal_analysis.py` | Analyzes fusion gate weights and feature norms / 퓨전 게이트 가중치, 특성 분석 |

---

## 6. Quick Reference / 빠른 참조

### Run all analyses for a set of models / 모델 세트에 대한 전체 분석 실행

```bash
# Create config list / config 목록 생성
cat > my_configs.txt << EOF
in2d_out12h_gnn_transformer
in2d_out12h_transformer
in2d_out12h
EOF

# Run all primary analyses / 주요 분석 전체 실행
./validation.sh --config-file my_configs.txt --epoch best
./mcd.sh --config-file my_configs.txt --epoch best
./attention.sh --config-file my_configs.txt --epoch best

# Run post-hoc evaluations / 사후 평가 실행
python analysis/evaluate_storm_performance.py --results-dir /path/to/results --filter out12h
python analysis/compare_predictions.py --results-dir /path/to/results --config-base in2d_out12h
python analysis/evaluate_mcd.py --results-dir /path/to/results --filter out12h
python analysis/visualize_gnn_graph.py --results-dir /path/to/results --configs in2d_out12h_gnn_transformer
```
