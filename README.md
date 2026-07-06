# Solar Wind Prediction - Geomagnetic Index Regression
# 태양풍 기반 지자기 지수 예측

Deep learning system for predicting a geomagnetic index from solar wind time series data. The target index is a Hydra config choice (index-agnostic); ap30 is the default example, and hp30 is also defined. Supports 14 model architectures including GNN, PatchTST, and TimesNet.

태양풍 시계열 데이터로 지자기 지수를 예측하는 딥러닝 시스템. 예측 대상 지수는 Hydra 설정으로 선택하며(지수 무관), ap30이 기본 예시이고 hp30도 정의되어 있습니다. GNN, PatchTST, TimesNet 등 14가지 모델 아키텍처를 지원합니다.

---

## Features / 주요 기능

- **Modular data pipeline / 모듈형 데이터 파이프라인**: CSV time series (active), HDF5 multi-modal (SDO+OMNI, future)
- **14 model architectures / 14가지 모델**: Linear, LSTM, BiLSTM, Transformer, TCN, PatchTST, TimesNet, GNN×7
- **SolarWindWeightedLoss**: NOAA G-Scale based weighted loss for geomagnetic storms / NOAA G-Scale 기반 지자기 폭풍 가중 손실함수
- **Hydra configuration / Hydra 설정**: Easy experiment management with config inheritance / 설정 상속으로 손쉬운 실험 관리
- **GNN with dynamic node groups / GNN 동적 노드 그룹**: Config-based variable grouping with validation / 설정 기반 변수 그룹화 및 검증

---

## Quick Start / 빠른 시작

```bash
# Using ap CLI (recommended) / ap CLI 사용 (권장)
../geoindex/ap train --profile standard           # GNN+Transformer, 2-day input, 12h output
../geoindex/ap train --profile quick              # Linear baseline, fast (~2 min)
../geoindex/ap validate --profile standard --epoch best
../geoindex/ap analyze attention --profile standard

# Using Hydra config groups directly / Hydra 설정 그룹 직접 사용
python scripts/train.py --config-name=local +io=in2d_out12h +model=gnn_transformer
python scripts/train.py --config-name=local +io=in1d_out6h +model=linear

# Train all 336 experiments / 336개 전체 실험 훈련
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
geoindex-model/
├── configs/                # Hydra configuration files / Hydra 설정 파일
│   ├── base.yaml           # Shared defaults / 공유 기본 설정
│   ├── local.yaml          # Environment settings / 환경 설정
│   ├── io/                 # I/O window configs (24) / 입출력 윈도우 설정
│   │   ├── in6h_out6h.yaml ... in3d_out24h.yaml
│   ├── model/              # Model configs (14) / 모델 설정
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
└── validation.sh           # Parallel validation runner / 병렬 검증 실행기
```

Documentation lives in the sibling hub repo under
`../geoindex/docs/geoindex-model/`, not inside this repo.
문서는 형제 허브 저장소의 `../geoindex/docs/geoindex-model/` 아래에 있다.

---

## Data / 데이터

Current dataset: CSV-based 30-min solar wind time series from `geoindex-data`.
현재 데이터셋: `geoindex-data`에서 생성된 CSV 기반 30분 간격 태양풍 시계열.

See / 상세: [dataset-guide.md](../geoindex/docs/geoindex-model/dataset-guide.md)

- **22 input variables / 22개 입력 변수**: solar wind (v, np, t, bx, by, bz, bt) × (avg/min/max) + ap30
- **Target / 타겟**: configurable geomagnetic index (Hydra config choice) — ap30 by default (30-min equivalent amplitude geomagnetic index); hp30 also defined / 설정 가능한 지자기 지수(Hydra 설정 선택) — 기본값 ap30(30분 등가진폭 지자기 지수), hp30도 정의됨
- **Output windows / 출력 윈도우**: 6h (12 timesteps), 12h (24 timesteps), 24h (48 timesteps)

---

## Models / 모델 (14종)

See / 상세: [model-guide.md](../geoindex/docs/geoindex-model/model-guide.md)

| # | `+model=` | Type | Description / 설명 |
|---|-----------|------|-------------------|
| 1 | `linear` | `linear` | MLP baseline / MLP 기준선 |
| 2 | `lstm` | `lstm` | LSTM recurrent encoder |
| 3 | `bilstm` | `bilstm` | Bidirectional LSTM |
| 4 | `transformer` | `transformer` | Transformer encoder |
| 5 | `tcn` | `tcn` | Temporal Convolutional Network |
| 6 | `patchtst` | `patchtst` | Patch-based Transformer (ICLR 2023) |
| 7 | `timesnet` | `timesnet` | FFT + 2D Inception Conv (ICLR 2023) |
| 8 | `gnn_linear` | `gnn` | GNN + Linear |
| 9 | `gnn_lstm` | `gnn` | GNN + LSTM |
| 10 | `gnn_bilstm` | `gnn` | GNN + BiLSTM |
| 11 | `gnn_transformer` | `gnn` | GNN + Transformer |
| 12 | `gnn_tcn` | `gnn` | GNN + TCN |
| 13 | `gnn_patchtst` | `gnn` | GNN + PatchTST |
| 14 | `gnn_timesnet` | `gnn` | GNN + TimesNet |

GNN variants build a graph over config-driven node groups (variable node count; 8 groups is the default grouping).
GNN 계열은 설정 기반 노드 그룹(가변 노드 수, 기본 8개 그룹)으로 그래프를 구성합니다.

---

## Experiment Matrix / 실험 매트릭스

`train.sh` cross-products every `configs/io/*.yaml` with every `configs/model/*.yaml`:
336 experiments = 24 I/O windows × 14 models.
`train.sh`는 모든 `configs/io/*.yaml`와 모든 `configs/model/*.yaml`를 교차 조합합니다:
336개 실험 = 24 입출력 윈도우 × 14 모델.

I/O windows = 6 inputs (6h/12h/18h/1d/2d/3d) × 4 outputs (6h/12h/18h/24h).
입출력 윈도우 = 6 입력(6h/12h/18h/1d/2d/3d) × 4 출력(6h/12h/18h/24h).

See / 상세: [experiments.md](../geoindex/docs/geoindex-model/experiments.md)

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

Sample I/O windows (3 of 24; full list in `configs/io/`) / 예시 입출력 윈도우 (24개 중 3개; 전체는 `configs/io/`):

| I/O Window | Input | Output | `+io=` |
|------------|-------|--------|--------|
| 1 day → 6h | 48 steps | 12 steps | `in1d_out6h` |
| 2 days → 12h | 96 steps | 24 steps | `in2d_out12h` |
| 3 days → 24h | 144 steps | 48 steps | `in3d_out24h` |

All 14 models (full list in `configs/model/`) / 전체 14개 모델 (전체는 `configs/model/`):

| Model | `+model=` |
|-------|-----------|
| Linear | `linear` |
| LSTM | `lstm` |
| BiLSTM | `bilstm` |
| Transformer | `transformer` |
| TCN | `tcn` |
| PatchTST | `patchtst` |
| TimesNet | `timesnet` |
| GNN+Linear | `gnn_linear` |
| GNN+LSTM | `gnn_lstm` |
| GNN+BiLSTM | `gnn_bilstm` |
| GNN+Transformer | `gnn_transformer` |
| GNN+TCN | `gnn_tcn` |
| GNN+PatchTST | `gnn_patchtst` |
| GNN+TimesNet | `gnn_timesnet` |

---

## Analysis / 분석

Post-training analysis tools for model interpretability and evaluation.
훈련 후 모델 해석 및 평가를 위한 분석 도구.

See / 상세: [analysis.md](../geoindex/docs/geoindex-model/analysis.md)

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
