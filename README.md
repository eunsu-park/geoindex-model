# Solar Wind Prediction - ap30 Regression
# 태양풍 기반 ap30 지자기 지수 예측

Deep learning system for predicting the ap30 geomagnetic index using solar wind time series data. Supports 9 model architectures including GNN, PatchTST, and TimesNet.

태양풍 시계열 데이터를 이용한 ap30 지자기 지수 딥러닝 예측 시스템. GNN, PatchTST, TimesNet 등 9가지 모델 아키텍처를 지원합니다.

---

## Features / 주요 기능

- **Modular data pipeline / 모듈형 데이터 파이프라인**: CSV time series (active), HDF5 multi-modal (SDO+OMNI, future)
- **9 model architectures / 9가지 모델**: Linear, Transformer, TCN, PatchTST, TimesNet, GNN×4
- **SolarWindWeightedLoss**: NOAA G-Scale based weighted loss for geomagnetic storms / NOAA G-Scale 기반 지자기 폭풍 가중 손실함수
- **Hydra configuration / Hydra 설정**: Easy experiment management with config inheritance / 설정 상속으로 손쉬운 실험 관리
- **GNN with dynamic node groups / GNN 동적 노드 그룹**: Config-based variable grouping with validation / 설정 기반 변수 그룹화 및 검증

---

## Quick Start / 빠른 시작

```bash
# Using ap CLI (recommended) / ap CLI 사용 (권장)
../ap train --profile standard           # GNN+Transformer, 2-day input, 12h output
../ap train --profile quick              # Linear baseline, fast (~2 min)
../ap validate --profile standard --epoch best
../ap analyze attention --profile standard

# Using Hydra config groups directly / Hydra 설정 그룹 직접 사용
python scripts/train.py --config-name=local +io=in2d_out12h +model=gnn_transformer
python scripts/train.py --config-name=local +io=in1d_out6h +model=linear

# Train all 81 experiments / 81개 전체 실험 훈련
./train.sh

# Train specific subset / 특정 부분만 훈련
./train.sh --filter out12h          # 12h output only
./train.sh --model transformer      # Transformer only
./train.sh --filter "in(6|12)h"     # Regex supported / 정규식 지원
./train.sh --max-jobs 4             # Limit parallel jobs / 병렬 작업 제한

# Validate all / 전체 검증
./validation.sh --epoch best
```

---

## Project Structure / 프로젝트 구조

```
regression-sw/
├── configs/                # Hydra configuration files / Hydra 설정 파일
│   ├── base.yaml           # Shared defaults / 공유 기본 설정
│   ├── local.yaml          # Environment settings / 환경 설정
│   ├── io/                 # I/O window configs (9) / 입출력 윈도우 설정
│   │   ├── in1d_out6h.yaml ... in3d_out24h.yaml
│   ├── model/              # Model configs (9) / 모델 설정
│   │   ├── linear.yaml, transformer.yaml, gnn_transformer.yaml ...
│   └── experiments/        # Experiment overrides / 실험 오버라이드
├── src/                    # Core modules / 핵심 모듈
│   ├── networks/           # Model architectures (package) / 모델 아키텍처 (패키지)
│   │   ├── _registry.py    # @register_model decorator / 모델 레지스트리
│   │   ├── transformer.py, tcn.py, gnn.py, patchtst.py ...
│   ├── pipeline/           # Data loading (package) / 데이터 로딩 (패키지)
│   │   ├── normalizer.py, readers.py, datasets_csv.py, factory.py ...
│   ├── plotting.py         # Shared visualization / 공용 시각화
│   ├── trainers.py         # Training loop / 훈련 루프
│   ├── validators.py       # Validation loop / 검증 루프
│   ├── testers.py          # Inference / 추론
│   ├── losses.py           # Loss functions / 손실 함수
│   └── utils.py            # Utilities / 유틸리티
├── scripts/                # Entry points / 실행 스크립트
├── analysis/               # Interpretability / 해석 도구 (attention, MCD)
├── tests/                  # Unit tests (156 tests) / 단위 테스트
├── train.sh                # Parallel training runner / 병렬 훈련 실행기
├── validation.sh           # Parallel validation runner / 병렬 검증 실행기
└── docs/                   # Documentation / 문서
    ├── DATASET_GUIDE.md    # Data format docs / 데이터 형식 문서
    ├── MODEL_GUIDE.md      # Model architecture docs / 모델 아키텍처 문서
    ├── EXPERIMENTS.md      # Experiment log / 실험 기록
    ├── ANALYSIS.md         # Analysis tools guide / 분석 도구 가이드
    └── DATA_FORMAT_ANALYSIS.md  # Data format comparison / 데이터 형식 비교
```

---

## Data / 데이터

Current dataset: CSV-based 30-min solar wind time series from `setup-sw-db`.
현재 데이터셋: `setup-sw-db`에서 생성된 CSV 기반 30분 간격 태양풍 시계열.

See / 상세: [DATASET_GUIDE.md](docs/DATASET_GUIDE.md)

- **22 input variables / 22개 입력 변수**: solar wind (v, np, t, bx, by, bz, bt) × (avg/min/max) + ap30
- **Target / 타겟**: ap30 (30-min equivalent amplitude geomagnetic index / 30분 등가진폭 지자기 지수)
- **Output windows / 출력 윈도우**: 6h (12 timesteps), 12h (24 timesteps), 24h (48 timesteps)

---

## Models / 모델 (9종)

See / 상세: [MODEL_GUIDE.md](docs/MODEL_GUIDE.md)

| # | Config Suffix | Type | Description / 설명 |
|---|--------------|------|-------------------|
| 1 | (none) | `linear` | MLP baseline / MLP 기준선 |
| 2 | `_transformer` | `transformer` | Transformer encoder |
| 3 | `_tcn` | `tcn` | Temporal Convolutional Network |
| 4 | `_patchtst` | `patchtst` | Patch-based Transformer (ICLR 2023) |
| 5 | `_timesnet` | `timesnet` | FFT + 2D Inception Conv (ICLR 2023) |
| 6 | `_gnn_transformer` | `gnn` | GNN (8-node graph) + Transformer |
| 7 | `_gnn_tcn` | `gnn` | GNN + TCN |
| 8 | `_gnn_bilstm` | `gnn` | GNN + BiLSTM |
| 9 | `_gnn_patchtst` | `gnn` | GNN + PatchTransformer |

---

## Experiment Matrix / 실험 매트릭스

81 experiments = 9 models × 3 inputs (1d/2d/3d) × 3 outputs (6h/12h/24h)
81개 실험 = 9모델 × 3입력 × 3출력

See / 상세: [EXPERIMENTS.md](docs/EXPERIMENTS.md)

---

## Config System / 설정 시스템

Hydra config groups compose I/O windows and models independently.
Hydra 설정 그룹으로 I/O 윈도우와 모델을 독립적으로 조합합니다.

```bash
# Syntax / 문법:
python scripts/train.py --config-name=local +io={window} +model={model}

# Examples / 예시:
python scripts/train.py --config-name=local +io=in2d_out12h +model=transformer
python scripts/train.py --config-name=local +io=in1d_out6h +model=linear
python scripts/train.py --config-name=local +io=in3d_out24h +model=gnn_patchtst
```

| I/O Window | Input | Output | `+io=` |
|------------|-------|--------|--------|
| 1 day → 6h | 48 steps | 12 steps | `in1d_out6h` |
| 2 days → 12h | 96 steps | 24 steps | `in2d_out12h` |
| 3 days → 24h | 144 steps | 48 steps | `in3d_out24h` |

| Model | `+model=` |
|-------|-----------|
| Linear | `linear` |
| Transformer | `transformer` |
| TCN | `tcn` |
| PatchTST | `patchtst` |
| TimesNet | `timesnet` |
| GNN+Transformer | `gnn_transformer` |
| GNN+TCN | `gnn_tcn` |
| GNN+BiLSTM | `gnn_bilstm` |
| GNN+PatchTST | `gnn_patchtst` |

---

## Analysis / 분석

Post-training analysis tools for model interpretability and evaluation.
훈련 후 모델 해석 및 평가를 위한 분석 도구.

See / 상세: [ANALYSIS.md](docs/ANALYSIS.md)

### Primary Analysis (require checkpoint + data) / 주요 분석 (체크포인트 + 데이터 필요)

| Script | Shell | Description / 설명 |
|--------|-------|-------------------|
| `analysis/run_attention.py` | `./attention.sh` | Transformer attention weight extraction / Attention 가중치 추출 |
| `analysis/run_mcd.py` | `./mcd.sh` | Monte Carlo Dropout uncertainty / MCD 불확실성 추정 |
| `analysis/run_saliency.py` | `./saliency.sh` | Gradient-based saliency maps (SDO) / Saliency 맵 (SDO 전용) |

### Post-hoc Evaluation (process existing results) / 사후 평가 (기존 결과 처리)

| Script | Description / 설명 |
|--------|-------------------|
| `analysis/evaluate_storm_performance.py` | Storm-tier filtered MAE/RMSE / 폭풍 티어별 성능 |
| `analysis/compare_predictions.py` | Multi-model prediction overlay plots / 다중 모델 예측 비교 플롯 |
| `analysis/evaluate_mcd.py` | MCD coverage and calibration / MCD 커버리지 및 보정 |
| `analysis/visualize_gnn_graph.py` | GNN learned adjacency heatmap / GNN 인접 행렬 시각화 |

---

## Testing / 테스트

```bash
conda activate ap
pytest tests/ -v
```

---

## License / 라이선스

See / 참조: [LICENSE](LICENSE)
