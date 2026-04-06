# Experiment Log

## Overview

CSV 기반 30분 간격 태양풍 시계열 데이터를 사용한 ap30 회귀 예측 실험 기록.

데이터셋 상세: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## Common Settings

모든 실험에서 공유하는 설정 (base.yaml + local.yaml 기반).

| 항목 | 값 |
|------|-----|
| Model | Transformer (d_model=128, nhead=4, layers=2, ff=256) |
| Batch Size | 128 |
| Optimizer | Adam (lr=2e-4, weight_decay=0.0) |
| LR Scheduler | ReduceOnPlateau (factor=0.5, patience=5) |
| Loss | SolarWindWeightedLoss (multi_tier, base=MSE) |
| Early Stopping | patience=10, min_delta=0.0 |
| Max Epochs | 30 |
| Dropout | 0.1 |
| Gradient Clipping | max_norm=1.0 |
| Input Variables | 23개 (태양풍 21 + ap30, hp30) |
| Target Variable | ap30 |
| Normalization | log_zscore (v, np, t, bt), zscore (bx, by, bz), log1p_zscore (ap30, hp30) |
| Dataset Mode | Table (Parquet + index CSV) |
| Device | CUDA |

---

## Experiment Matrix

입력 길이(1일/2일/3일) × 출력 예측 길이(12h/24h) = 6개 조합.

| 실험명 | 입력 범위 | 입력 timesteps | 출력 범위 | 출력 timesteps | Config |
|--------|----------|---------------|----------|---------------|--------|
| in1d_out12h | T-1d ~ T | 48 | T ~ T+12h | 24 | `in1d_out12h.yaml` |
| in1d_out24h | T-1d ~ T | 48 | T ~ T+24h | 48 | `in1d_out24h.yaml` |
| in2d_out12h | T-2d ~ T | 96 | T ~ T+12h | 24 | `in2d_out12h.yaml` |
| in2d_out24h | T-2d ~ T | 96 | T ~ T+24h | 48 | `in2d_out24h.yaml` |
| in3d_out12h | T-3d ~ T | 144 | T ~ T+12h | 24 | `in3d_out12h.yaml` |
| in3d_out24h | T-3d ~ T | 144 | T ~ T+24h | 48 | `in3d_out24h.yaml` |

---

## Results

### Best Epoch Performance Summary

| 실험 | Total Epochs | Best Epoch | Val Loss | Val MAE | Val RMSE | Train Loss | Train MAE | Train-Val Gap |
|------|-------------|------------|----------|---------|----------|------------|-----------|--------------|
| **in1d_out12h** | 22 | 12 | 0.2887 | 0.3979 | 0.5125 | 0.2640 | 0.3816 | +0.0247 |
| **in2d_out12h** | 22 | 12 | **0.2842** | **0.3932** | **0.5055** | 0.2607 | 0.3800 | +0.0235 |
| **in3d_out12h** | 22 | 12 | 0.2840 | 0.3953 | 0.5082 | 0.2601 | 0.3799 | +0.0239 |
| **in1d_out24h** | 12 | 2 | 0.5898 | 0.5154 | 0.6661 | 0.5886 | 0.5207 | +0.0012 |
| **in2d_out24h** | 12 | 2 | 0.5845 | 0.5174 | 0.6681 | 0.5812 | 0.5193 | +0.0033 |
| **in3d_out24h** | 12 | 2 | 0.5938 | 0.5231 | 0.6721 | 0.5713 | 0.5195 | +0.0224 |

> **Best model: in2d_out12h** (Val Loss=0.2842, Val MAE=0.3932)

### Cosine Similarity

전 실험에서 0.0 — Contrastive Loss가 비활성 상태 (SDO 이미지 모달리티 미사용).

---

## Analysis

### 1. 12시간 예측 (out12h) — 안정적 학습

- 3개 실험 모두 22 epoch 학습, best epoch = 12로 동일
- Val Loss 범위: 0.284 ~ 0.289 (편차 매우 작음)
- Train-Val gap: epoch 12 기준 ~0.024로 적정 수준
- Epoch 12 이후 train loss는 계속 감소하나 val loss는 정체/상승 → **mild overfitting 시작**
- 입력 길이 증가에 따른 성능 향상 미미:
  - 1d → 2d: Val MAE 0.3979 → 0.3932 (-0.0047, 약간 개선)
  - 2d → 3d: Val MAE 0.3932 → 0.3953 (+0.0021, 오히려 소폭 하락)
  - **결론: 2일 입력이 최적 (sweet spot)**

#### 12h 예측 Training Curves (in2d_out12h)

```
Epoch | Train Loss | Val Loss   | Gap       | Train MAE | Val MAE
    1 | 0.504613   | 0.327313   | -0.177300 | 0.494543  | 0.433293
    2 | 0.334727   | 0.307461   | -0.027267 | 0.427207  | 0.420913
    3 | 0.310128   | 0.293713   | -0.016415 | 0.412197  | 0.403871
    4 | 0.297899   | 0.296247   | -0.001651 | 0.403811  | 0.401699
    5 | 0.289297   | 0.287800   | -0.001497 | 0.398321  | 0.395488
   12 | 0.260681   | 0.284200   | +0.023518 | 0.380029  | 0.393169  ← Best
   22 | 0.239857   | 0.298876   | +0.059020 | 0.367734  | 0.404590
```

### 2. 24시간 예측 (out24h) — 심각한 과적합

- **Best epoch이 모두 2** — 학습 초기에만 유효하고 이후 계속 악화
- Epoch 12 기준 train-val gap: **0.27 ~ 0.34** (12h 실험 대비 10배 이상)
- Val loss가 epoch 2 이후 단조 증가 → **전형적인 과적합 (overfitting)**
- 입력 길이 증가 효과 없음 (3개 실험 간 성능 거의 동일)

#### 24h 예측 Training Curves (in2d_out24h)

```
Epoch | Train Loss | Val Loss   | Gap       | Train MAE | Val MAE
    1 | 0.736767   | 0.601972   | -0.134795 | 0.573262  | 0.526359
    2 | 0.581154   | 0.584454   | +0.003300 | 0.519306  | 0.517444  ← Best
    3 | 0.541636   | 0.600354   | +0.058718 | 0.506113  | 0.527771
    6 | 0.456655   | 0.677105   | +0.220451 | 0.481222  | 0.556823
   12 | 0.376707   | 0.717561   | +0.340853 | 0.450765  | 0.545409
```

### 3. 예측 품질 (플롯 분석)

- **12h 예측**: 전반적 추세를 따라가지만, ap30 급상승(지자기 폭풍) 시 피크 값을 과소 예측
  - 예시: in2d_out12h epoch 5, batch 200 — MAE 1.46, RMSE 1.92
- **24h 예측**: 예측선이 거의 평탄하게 나옴, ground truth의 급격한 변동을 캡처하지 못함
  - 예시: in2d_out24h epoch 5, batch 200 — MAE 3.18, RMSE 5.29
  - ap30 > 10인 폭풍 이벤트에서 괴리가 특히 심각

### 4. LR Scheduler 동작

- 전 실험에서 LR이 2e-4로 고정 (start = end)
- ReduceOnPlateau (patience=5)가 설정되어 있으나, val loss가 plateau 구간 없이 완만하게 변동하여 트리거되지 않은 것으로 추정

---

## Issues & Recommendations

### 우선순위 1: 24h 예측 과적합 해결

- [x] Dropout 증가 (현재 0.1 → 0.2~0.3 시도) → A1, A3, A4 config 생성
- [x] Weight decay 추가 (현재 0.0 → 1e-4~1e-3 시도) → A1, A3, A4, A5 config 생성
- [x] Data augmentation 적용 → B2 (Gaussian noise augmentation 구현 완료)
- [x] 모델 용량 축소 검토 (d_model=128 → 64, layers=2 → 1) → A4 config 생성

### 우선순위 2: 학습률 스케줄링 개선

- [x] Cosine annealing 또는 warmup + decay 전략으로 변경 → A2, A3, A5 config 생성
- [x] ReduceOnPlateau 버그 수정: train_loss → val_loss 기준으로 변경 (trainers.py)

### 우선순위 3: 폭풍 이벤트 예측 강화

- [ ] SolarWindWeightedLoss의 고 ap30 구간 가중치 조정
- [ ] 폭풍 이벤트 별도 평가 메트릭 추가 (ap30 ≥ 30 구간 MAE/RMSE)

### 우선순위 4: ap30 이산값 매핑

ap30은 연속 실수가 아닌 28개 이산값(준로그 스케일)만 존재하나, 모델은 실수를 출력한다.

```
유효 ap30 값: 0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94, 111, 132, 154, 179, 207, 236, 300, 400
```

현재 log1p_zscore 정규화 → 역변환 `exp(...) - 1`이 양수를 보장하므로 음수 문제는 없음. 핵심은 이산값 매핑.

**단계별 접근:**

1. **후처리: Snap to nearest valid ap30** (코드 수정 최소, 모델 재훈련 불필요)
   - [ ] validation/inference 후처리로 가장 가까운 유효 ap30 값에 매핑
   - 단순 반올림보다 정확 (예: 10.3 → 반올림=10(무효), snap=9 또는 12(유효))
   - 저활동 구간(0, 2, 3)에서 메트릭 소폭 개선 기대

2. **모델 구조: Ordinal classification + regression hybrid** (과적합 해결 후 검토)
   - [ ] 회귀 head 유지 + 28개 클래스 분류 head 추가
   - 최종 출력: 분류 확률 가중 평균 또는 분류 결과로 snap
   - ap30의 이산 구조를 모델에 직접 주입

3. **출력 활성화 함수** (우선순위 낮음)
   - [ ] Softplus 등으로 출력 하한 제한 — 현재 log1p_zscore가 이미 양수 보장하므로 실질적 이득 작음

### 기타

- [x] 입력 길이: 2일이 최적으로 확인됨, 향후 실험은 in2d 기반 진행
- [ ] Cosine similarity / contrastive loss: 시계열 전용 모드에서는 의미 없으므로 로그에서 제외 검토

---

## Planned Experiments: 24h 과적합 해결

in2d_out24h 기반. 성공 기준: best epoch ≥ 8, train-val gap < 0.10, val_loss < 0.45.

### Phase A: Config-Only (코드 수정 없음)

| 실험 | Config | 변경 내용 | 우선순위 |
|------|--------|----------|---------|
| A1 | `in2d_out24h_A1.yaml` | weight_decay=0.01, dropout=0.3 | HIGH |
| A2 | `in2d_out24h_A2.yaml` | CosineAnnealing + LR warmup(3ep) | HIGH |
| A3 | `in2d_out24h_A3.yaml` | A1 + A2 결합 | HIGH |
| A4 | `in2d_out24h_A4.yaml` | d_model=64, layers=1, ff=128, dropout=0.2, wd=0.01 | MEDIUM |
| A5 | `in2d_out24h_A5.yaml` | lr=5e-5, wd=0.01, cosine, epochs=60 | MEDIUM |

### Phase B: 코드 수정 필요

| 실험 | Config | 변경 내용 | 상태 |
|------|--------|----------|------|
| B1 | — | ReduceOnPlateau가 val_loss 기준으로 step하도록 수정 (trainers.py) | 완료 |
| B2 | `in2d_out24h_B2.yaml` | A3 + Gaussian noise (std=0.05) augmentation | 구현 완료 |

### 실행 계획

1. **Batch 1**: A1, A2, A3 병렬 실행 (config-only, 즉시 가능)
2. **Batch 2**: B2 실행 (Batch 1 결과 후, A3 대신 최적 config 상속으로 변경 가능)
3. **Batch 3**: A4 또는 A5 (Batch 1 결과 기반 선택)

---

## Changelog

| 날짜 | 내용 |
|------|------|
| 2025-04-05 | 초기 6개 실험 (입력 1d/2d/3d × 출력 12h/24h) 결과 기록 |
| 2025-04-05 | 24h 과적합 해결 실험 계획 수립 (A1~A5, B1~B2) |
| 2025-04-05 | B1: ReduceOnPlateau 스케줄러 버그 수정 (train_loss→val_loss) |
| 2025-04-05 | B2: Gaussian noise augmentation 구현 (pipeline.py + base.yaml) |
| 2025-04-05 | 우선순위 4 추가: ap30 이산값 매핑 (snap 후처리 → ordinal classification 단계별) |
