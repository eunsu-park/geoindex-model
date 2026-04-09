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
# Train all 81 experiments (9 models × 9 input/output combos)
# 81개 전체 실험 훈련 (9모델 × 9 입출력 조합)
./train.sh

# Train specific subset / 특정 부분만 훈련
./train.sh --filter out12h          # 12h output only
./train.sh --filter gnn_transformer # GNN+Transformer only
./train.sh --max-jobs 4             # Limit parallel jobs / 병렬 작업 제한

# Validate all / 전체 검증
./validation.sh --epoch best

# Train single model / 단일 모델 훈련
python scripts/train.py --config-name=in2d_out12h_gnn_transformer
```

---

## Project Structure / 프로젝트 구조

```
regression-sw/
├── configs/                # Hydra configuration files / Hydra 설정 파일
│   ├── base.yaml           # Shared defaults / 공유 기본 설정
│   ├── local.yaml          # Environment settings / 환경 설정
│   ├── in{1,2,3}d_out{6,12,24}h.yaml     # Base I/O configs (Linear baseline)
│   ├── in*_*_{model}.yaml  # Model-specific configs / 모델별 설정
│   ├── archive/            # Old experiment configs / 이전 실험 설정
│   └── experiments/        # Experiment overrides / 실험 오버라이드
├── src/                    # Core modules / 핵심 모듈
│   ├── pipeline.py         # Data loading, normalization / 데이터 로딩, 정규화
│   ├── networks.py         # Model architectures / 모델 아키텍처
│   ├── trainers.py         # Training loop / 훈련 루프
│   ├── validators.py       # Validation loop / 검증 루프
│   ├── testers.py          # Inference / 추론
│   └── losses.py           # Loss functions / 손실 함수
├── scripts/                # Entry points / 실행 스크립트
├── analysis/               # Interpretability / 해석 도구 (attention, MCD)
├── tests/                  # Unit tests / 단위 테스트
├── train.sh                # Parallel training runner / 병렬 훈련 실행기
├── validation.sh           # Parallel validation runner / 병렬 검증 실행기
├── DATASET_GUIDE.md        # Data format docs / 데이터 형식 문서
├── MODEL_GUIDE.md          # Model architecture docs / 모델 아키텍처 문서
└── EXPERIMENTS.md          # Experiment log / 실험 기록
```

---

## Data / 데이터

Current dataset: CSV-based 30-min solar wind time series from `setup-sw-db`.
현재 데이터셋: `setup-sw-db`에서 생성된 CSV 기반 30분 간격 태양풍 시계열.

See / 상세: [DATASET_GUIDE.md](DATASET_GUIDE.md)

- **22 input variables / 22개 입력 변수**: solar wind (v, np, t, bx, by, bz, bt) × (avg/min/max) + ap30
- **Target / 타겟**: ap30 (30-min equivalent amplitude geomagnetic index / 30분 등가진폭 지자기 지수)
- **Output windows / 출력 윈도우**: 6h (12 timesteps), 12h (24 timesteps), 24h (48 timesteps)

---

## Models / 모델 (9종)

See / 상세: [MODEL_GUIDE.md](MODEL_GUIDE.md)

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

See / 상세: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## Config Naming Convention / 설정 파일 명명 규칙

```
in{input_days}d_out{output_hours}h_{model_suffix}.yaml

Examples / 예시:
  in2d_out12h.yaml                  → Linear baseline (2-day input, 12h output)
  in2d_out12h_transformer.yaml      → Transformer
  in2d_out12h_gnn_transformer.yaml  → GNN + Transformer
```

---

## Testing / 테스트

```bash
conda activate ap
pytest tests/ -v
```

---

## License / 라이선스

See / 참조: [LICENSE](LICENSE)
