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

> **Best 12h model: in2d_out12h** (Val Loss=0.2842, Val MAE=0.3932)

### 24h 과적합 해결 실험 결과

in2d_out24h 기반. Baseline의 best epoch=2, val_loss=0.584에서 개선 시도.

| 실험 | Best Epoch | Val Loss | Val MAE | Val RMSE | Gap (best) | Gap (last) | LR (end) |
|------|-----------|----------|---------|----------|-----------|-----------|----------|
| **Baseline** | 2 | 0.5845 | 0.5174 | 0.6681 | +0.003 | +0.341 | 2.00e-4 |
| **A1** (wd+dropout) | 3 | 0.6087 | 0.5275 | 0.6777 | +0.014 | +0.325 | 1.00e-4 |
| **A2** (cosine+warmup) | 2 | 0.5841 | 0.5170 | 0.6676 | +0.003 | +0.228 | 1.99e-4 |
| **A3** (A1+A2) | 3 | 0.6086 | 0.5266 | 0.6769 | +0.014 | +0.228 | 1.95e-4 |
| **A4** (모델 축소) | 5 | 0.6142 | 0.5388 | 0.6932 | +0.003 | +0.096 | 1.00e-4 |
| **A5** (낮은 LR) | **11** | **0.6056** | **0.5225** | **0.6746** | +0.017 | +0.118 | 2.55e-5 |
| **B2** (A3+noise) | 3 | 0.6063 | 0.5257 | 0.6762 | +0.011 | +0.221 | 1.95e-4 |

> **Best 24h model: A5 (in2d_out24h_A5_low_lr)** — best epoch 11, 과적합 가장 효과적으로 억제

#### 성공 기준 달성 여부

| 기준 | 목표 | A5 (최고) | 달성 |
|------|------|----------|------|
| Best epoch ≥ 8 | 안정적 학습 | 11 | **달성** |
| Train-val gap < 0.10 (best epoch) | 과적합 억제 | 0.017 | **달성** |
| Val loss < 0.45 | 성능 개선 | 0.606 | 미달성 |

#### 실험별 분석

- **A5 (낮은 LR)**: 가장 성공적. lr=5e-5 + cosine annealing으로 best epoch이 11로 이동. Cosine 주기 재시작(epoch 11)이 2차 탐색 기회 제공. 다만 2번째 cycle에서 다시 과적합 시작.
- **A4 (모델 축소)**: 두 번째로 효과적. d_model=64, layers=1로 best epoch=5, 마지막 gap=0.096. 하지만 val_loss(0.614)가 높아 underfitting 경향.
- **A1/A3 (weight decay + dropout)**: 과적합 속도 감소했으나 강한 정규화가 학습 자체를 억제. ReduceOnPlateau가 정상 발동(A1: LR 2e-4→1e-4).
- **A2 (cosine+warmup)**: Baseline과 거의 동일. Warmup 3ep + cosine T_0=10이라 초반 동작이 유사.
- **B2 (A3+noise)**: A3 대비 소폭 개선(val_loss 0.606 vs 0.609). Augmentation 효과 제한적.

#### 핵심 시사점

1. **낮은 학습률이 가장 결정적** — 정규화(dropout, weight decay)보다 LR 자체를 낮추는 것이 과적합 억제에 효과적
2. **Val loss 하한 ~0.60** — 모든 실험에서 val_loss가 0.58~0.61 범위. 24h 예측의 본질적 난이도 한계 가능성
3. **ReduceOnPlateau 버그 수정 효과 확인** — A1에서 LR이 정상적으로 감소(2e-4→1e-4)

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

#### 배경

ap30은 연속 실수가 아닌 **Kp→ap 변환표**에서 파생되는 28개 이산값만 존재한다.

```
Kp:  0o  0+  1-  1o  1+  2-  2o  2+  3-  3o  3+  4-  4o  4+  5-  5o  5+  6-  6o  6+  7-  7o  7+  8-  8o  8+  9-  9o
ap:   0   2   3   4   5   6   7   9  12  15  18  22  27  32  39  48  56  67  80  94 111 132 154 179 207 236 300 400
```

현재 log1p_zscore 정규화 → 역변환 `exp(...) - 1`이 양수를 보장하므로 음수 문제는 없음. 핵심은 이산값 매핑.

#### 현재 파이프라인에서의 데이터 공간

| 단계 | 파일 | 데이터 공간 |
|------|------|-----------|
| 모델 출력 | networks.py | 정규화 공간 (활성화 없음, 연속 실수) |
| Loss 계산 | losses.py | 정규화 공간 (pred & target) |
| Loss 가중치 | losses.py | 역정규화 공간 (target만, AP_TIERS 매핑용) |
| 학습 메트릭 (MAE/RMSE) | trainers.py | 정규화 공간 |
| 검증 메트릭 (MAE/RMSE) | validators.py | 정규화 공간 |
| 출력 파일 (npz/plot) | validators.py, testers.py | 역정규화 공간 |

#### 방법 (1): 후처리 Snap — 즉시 적용 권장

모델 출력 → denormalize → **가장 가까운 유효 ap30 값에 snap**. 모델 재훈련 불필요.

```python
AP30_VALUES = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32,
                         39, 48, 56, 67, 80, 94, 111, 132, 154, 179, 207, 236, 300, 400])

def snap_to_ap30(predictions):
    idx = np.abs(AP30_VALUES[None, :] - predictions[:, None]).argmin(axis=1)
    return AP30_VALUES[idx]
```

- [ ] validators.py, testers.py의 denormalize 직후에 snap 적용
- 저활동 구간(0, 2, 3)에서 메트릭 소폭 개선 기대
- 학습에는 영향 없으므로 근본적 성능 향상은 아님

#### 방법 (2): 모델 구조 변경 — 현 시점 비권장

검토한 3가지 접근과 평가:

**(2A) Ordinal Classification**: regression head를 28개 클래스 분류로 교체
- 출력 차원 `seq_len × 1` → `seq_len × 28` (28배 증가)
- 24h 예측: 48 → 1,344 출력 뉴런
- **비권장**: 이미 과적합이 심각한 상황에서 파라미터 증가는 역효과. 클래스 불균형 극심 (하위 7~8개 클래스가 데이터 80% 이상)

**(2B) 듀얼 헤드 (회귀 + 분류)**: regression head 유지 + classification head 보조 추가
- Loss = regression_loss + λ × classification_loss
- **보류**: (2A)와 동일한 파라미터 증가 문제 + 하이퍼파라미터(λ) 추가

**(2C) Snap-aware Loss**: 모델 구조 변경 없이 loss에서 유효값 근접 패널티 추가
- 예: `loss = base_loss + λ × distance_to_nearest_valid_ap30`
- snap 연산이 non-differentiable → straight-through estimator 등 우회 필요
- **과적합 해결 후 검토 가능**

#### 종합 판단

| 방법 | 구현 난이도 | 성능 개선 기대 | 과적합 영향 | 판정 |
|------|-----------|-------------|-----------|------|
| (1) Snap 후처리 | 매우 낮음 | 소폭 (메트릭만) | 없음 | **즉시 적용** |
| (2A) Ordinal Classification | 높음 | 불확실 | 악화 (파라미터 28배↑) | 비권장 |
| (2B) 듀얼 헤드 | 중간 | 불확실 | 악화 | 보류 |
| (2C) Snap-aware Loss | 중간 | 소폭 | 중립 | 향후 검토 |

현 시점에서는 **(1) Snap 후처리를 먼저 적용**하여 평가 메트릭 변화를 확인하고, 과적합 해결 이후 (2C) snap-aware loss를 검토하는 것이 합리적이다.

### 기타

- [x] 입력 길이: 2일이 최적으로 확인됨, 향후 실험은 in2d 기반 진행
- [ ] Cosine similarity / contrastive loss: 시계열 전용 모드에서는 의미 없으므로 로그에서 제외 검토

---

## Planned Experiments: 24h 과적합 해결

in2d_out24h 기반. 성공 기준: best epoch ≥ 8, train-val gap < 0.10, val_loss < 0.45.

### Phase A: Config-Only (코드 수정 없음)

| 실험 | Config | 변경 내용 | 결과 |
|------|--------|----------|------|
| A1 | `in2d_out24h_A1.yaml` | weight_decay=0.01, dropout=0.3 | best=3, val_loss=0.609 (악화) |
| A2 | `in2d_out24h_A2.yaml` | CosineAnnealing + LR warmup(3ep) | best=2, val_loss=0.584 (동일) |
| A3 | `in2d_out24h_A3.yaml` | A1 + A2 결합 | best=3, val_loss=0.609 (악화) |
| A4 | `in2d_out24h_A4.yaml` | d_model=64, layers=1, ff=128, dropout=0.2, wd=0.01 | best=5, val_loss=0.614 (과적합↓ 성능↓) |
| **A5** | `in2d_out24h_A5.yaml` | lr=5e-5, wd=0.01, cosine, epochs=60 | **best=11, val_loss=0.606 (최고)** |

### Phase B: 코드 수정 필요

| 실험 | Config | 변경 내용 | 결과 |
|------|--------|----------|------|
| B1 | — | ReduceOnPlateau가 val_loss 기준으로 step하도록 수정 (trainers.py) | 완료 (A1에서 효과 확인) |
| B2 | `in2d_out24h_B2.yaml` | A3 + Gaussian noise (std=0.05) augmentation | best=3, val_loss=0.606 (A3 대비 소폭 개선) |

### Phase C: A5 기반 결합 실험

| 실험 | Config | 변경 내용 | 결과 |
|------|--------|----------|------|
| C1 | `in2d_out24h_C1.yaml` | A5 + A4 (d_model=64, layers=1) | best=40, val_loss=0.619 (안정적이나 underfitting) |
| **C2** | `in2d_out24h_C2.yaml` | A5 + noise (std=0.05) | **best=12, val_loss=0.603 (전체 최고)** |
| C3 | `in2d_out24h_C3.yaml` | A5 + A4 + noise (전부 결합) | best=31, val_loss=0.620 (underfitting) |

#### Phase C 분석

- **C2가 현재까지 최적 24h 모델** (val_loss=0.603, val_mae=0.521)
- Noise augmentation이 cosine 2번째 cycle에서 일반화를 도와 A5(0.606) 대비 추가 개선
- 모델 축소(C1, C3)는 과적합을 잘 억제하나(gap=0.04~0.06) 용량 부족으로 underfitting
- **d_model=128, layers=2를 유지하면서 lr=5e-5 + noise augmentation이 최적 조합**
- Val loss ~0.60이 현 Transformer 아키텍처의 24h 예측 한계선으로 확인

---

## Next Steps

### 1. 12h Cascade 추론 (24h = 12h × 2단)

24h을 한 번에 예측하는 대신, 검증된 12h 모델(val_loss=0.284)을 2단 연결하여 24h 예측.

#### 핵심 도전과제

```
Stage 1: 입력 [T-2d, T] (23변수) → 출력 ap30 [T, T+12h]      ← 문제 없음
Stage 2: 입력 [T-36h, T+12h] (23변수) → 출력 ap30 [T+12h, T+24h]
              ├─ [T-36h, T]: 실제 관측 데이터 23변수 ✓
              └─ [T, T+12h]: ap30은 Stage 1 출력, 나머지 22개 태양풍 변수는? ✗
```

모델 입력이 23변수를 요구하므로, Stage 2의 미래 구간 [T, T+12h]에 대해 22개 태양풍 변수를 채워야 함.

#### 구현 Phase

**Phase 1: Oracle Cascade** (상한선 측정)
- 미래 구간에 실제 태양풍 데이터 사용 (운영 불가, 이론적 상한 측정용)
- 새 스크립트: `scripts/validate_cascade.py`
- 새 config: `configs/cascade_oracle.yaml`
- 결과가 C2(val_loss=0.603)를 유의미하게 이기면 Phase 2 진행

**Phase 2: Persistence Cascade** (실용적 버전)
- 미래 구간의 태양풍 22변수를 T 시점 값으로 forward-fill
- ap30만 Stage 1 예측으로 교체
- `validate_cascade.py`에 `--mode oracle|persistence` 옵션 추가

**Phase 3: 결과 비교**

| 비교 | 의미 |
|------|------|
| Oracle vs C2(0.603) | Cascade 방식의 이론적 이점 존재 여부 |
| Persistence vs Oracle | 미래 태양풍 불확실성의 영향 크기 |
| Persistence vs C2 | 실용적으로 cascade가 우월한가 |

#### 구현 상세

```python
# Stage 2 입력 구성 (Persistence 모드):
input_stage2_past = real_data[T-36h : T]         # (72, 23) 실제 데이터
future_fill = real_data[T].repeat(24, axis=0)     # (24, 23) T 시점 값 복사
future_fill[:, ap30_idx] = stage1_pred_ap30       # ap30만 Stage 1 예측으로 교체
input_stage2 = concat(input_stage2_past, future_fill)  # (96, 23)
```

#### 수정 대상 파일

| 파일 | 변경 |
|------|------|
| `scripts/validate_cascade.py` | 새로 생성 — cascade 추론 로직 |
| `configs/cascade_oracle.yaml` | 새로 생성 — 두 모델 경로, 데이터 설정 |

### 2. TCN 아키텍처 실험

`model_type=tcn`이 이미 구현되어 있으며, 코드 검증 완료 (causal padding 정상, 입출력 shape Transformer와 호환).

**Receptive field 참고**: channels=[64,128,256], kernel=3 → RF=29 timesteps (14.5h). 4 layer 시 RF=57 (28.5h).

| 실험 | Config | 내용 | 상태 |
|------|--------|------|------|
| TCN 12h | `in2d_out12h_tcn.yaml` | TCN 기본, 12h 예측 (Transformer 비교용) | 대기 |
| TCN 24h | `in2d_out24h_tcn.yaml` | TCN + C2 설정 (lr=5e-5, cosine, noise) | 대기 |
| TCN 24h deep | `in2d_out24h_tcn_deep.yaml` | 4 layer (RF=57, 24h 커버), C2 설정 | 대기 |

### 3. GNN 모델 구현

#### 선행 연구

SYMHnet (Abduallah et al., 2024, Space Weather)이 GNN+BiLSTM으로 태양풍 → SYM-H 예측에서 SOTA 달성. 7개 태양풍/IMF 파라미터를 완전연결 그래프 노드로 구성, GCN 2 layer로 변수 간 관계 학습. Ablation에서 GNN 제거 시 R² 0.993→0.789로 급락 — 변수 간 관계 학습이 핵심 기여 확인.

#### 우리 과제에 대한 적용 방향

SYMHnet은 7개 변수 × 단일 시점 그래프 + BiLSTM(10 timestep)이지만, 우리 과제는 23개 변수 × 96 timestep 입력 → 24-48 timestep 시퀀스 출력. 따라서 SYMHnet 구조를 그대로 차용하지 않고, 기존 파이프라인에 맞게 설계.

**GNNEncoder 설계:**
- 노드 = 7개 물리 변수 그룹 (v, np, t, bx, by, bz, bt) + 2개 지수 (ap30, hp30) = 9 노드
  - 각 노드의 feature = avg/min/max triplet (3차원), ap30/hp30은 1차원
  - 전체: 7×3 + 2×1 = 23개 입력 변수를 9개 노드로 자연스럽게 매핑
- 엣지 = 적응적 학습 (MTGNN 방식: `A = softmax(relu(E1·E2ᵀ))`)
  - 물리적으로 의미 있는 관계 자동 학습 (예: Bz↔ap30 강한 연결)
  - 학습된 인접 행렬 시각화로 모델 해석 가능
- 시간축 = 기존 Transformer/TCN과 동일하게 처리 (GCN per timestep → temporal encoder)

**구현 상태: 완료** (순수 PyTorch, 외부 의존성 없음)
- `GraphConvLayer`: 단일 GCN layer (adaptive adj × node features × weight)
- `GNNEncoder`: 변수 그룹화 → per-timestep GCN → temporal encoder (플러그인)
- `GNNOnlyModel`: GNNEncoder + regression head (기존 인터페이스 호환)
- `create_model()`에 `model_type="gnn"` 분기 추가 완료
- Temporal encoder를 `gnn_temporal_type`으로 선택: `"transformer"`, `"tcn"`, `"bilstm"`

**실험 config:**

| Config | GNN temporal | 예측 | 훈련 설정 |
|--------|-------------|------|---------|
| `in2d_out12h_gnn_transformer.yaml` | Transformer | 12h | 기본 |
| `in2d_out12h_gnn_tcn.yaml` | TCN | 12h | 기본 |
| `in2d_out12h_gnn_bilstm.yaml` | BiLSTM | 12h | 기본 |
| `in2d_out24h_gnn_transformer.yaml` | Transformer | 24h | C2 (lr=5e-5, cosine, noise) |
| `in2d_out24h_gnn_tcn.yaml` | TCN | 24h | C2 |
| `in2d_out24h_gnn_bilstm.yaml` | BiLSTM | 24h | C2 |

### 4. TimesNet 모델 구현

#### 선행 연구

SINet (Wang et al., 2026, JGR Space Physics)이 TimesNet 기반으로 F10.7/F30 태양 지수 예측에서 TCN 대비 MAPE 개선 (10.2% vs 10.8%). 단, **단변량** 예측이며, 돌발 이벤트(AR 12673)에서 성능 저하를 논문 스스로 인정.

#### 우리 과제에 대한 적용 방향

원본 TimesNet은 채널 독립(channel-independent)이라 변수 간 관계를 학습하지 못함. 우리 과제에 맞게 **채널 혼합 확장**과 **다변량 입력** 지원 추가.

**TimesNetEncoder 설계:**
- FFT 기반 주기 탐지: 입력 시계열에서 top-k 지배 주기 추출 (k=3)
  - 태양풍 데이터의 준주기적 패턴 (태양 자전 27일 등) 포착 가능
- 1D→2D reshape: 각 주기별로 (period, seq_len/period) 2D 텐서 생성
- 2D Inception Conv: 다중 스케일 커널로 주기 내/간 패턴 추출
- Adaptive aggregation: FFT 진폭 기반 가중 합산
- **채널 혼합 확장**: 2D Conv 이후 cross-variable attention 또는 1×1 Conv로 변수 간 정보 교환

**구현 상태: 완료**
- `InceptionBlock`: 다중 스케일 2D Conv (kernel 1,3,5)
- `TimesBlock`: FFT 주기 탐지 → 1D→2D reshape → Inception → reshape back → adaptive aggregation
- `TimesNetEncoder`: TimesBlock 스택 + LayerNorm + **cross-variable self-attention** (채널 독립 문제 해결)
- `TimesNetOnlyModel`: TimesNetEncoder + regression head (기존 인터페이스 호환)
- `create_model()`에 `model_type="timesnet"` 분기 추가 완료

**실험 config:**

| Config | 예측 | 훈련 설정 |
|--------|------|---------|
| `in2d_out12h_timesnet.yaml` | 12h | 기본 |
| `in2d_out24h_timesnet.yaml` | 24h | C2 (lr=5e-5, cosine, noise) |

**주의:** 폭풍 이벤트가 비주기적이므로 TimesNet 단독 성능은 제한적일 수 있음. GNN과의 비교 실험이 핵심.

### 5. GNN vs TimesNet 비교 근거

| 항목 | GNN | TimesNet |
|------|-----|----------|
| **선행 사례** | SYMHnet: 태양풍→SYM-H (동일 도메인) | SINet: F10.7 예측 (다른 과제) |
| **변수 간 관계** | 명시적 그래프 학습 (핵심 강점) | 채널 독립 (확장 필요) |
| **주기성** | 약함 | 핵심 강점 (FFT 기반) |
| **폭풍 이벤트** | 양호 (SYMHnet 검증) | 약함 (SINet 논문에서 확인) |
| **해석 가능성** | 학습된 그래프 시각화 | 주기 분해 시각화 |
| **구현 난이도** | 중간 (GCN + adaptive adj) | 중간 (FFT + 2D Conv) |

### 6. Attention 분석

기존 12h/24h 모델의 attention 결과(`attention/best.zip`)를 비교하여 모델이 입력의 어느 시간대/변수에 집중하는지 분석. 24h 모델의 attention 분산 여부로 아키텍처 한계 진단.

### 7. Snap 후처리 적용

모델 재훈련 없이 denormalize 후 가장 가까운 유효 ap30 값에 매핑. validators.py, testers.py에 적용.

### 8. 18h 실험

12h(0.284)와 24h(0.603) 사이의 성능 저하 곡선 형태 확인. 예측 가능 시간 한계 특성화.

---

## Changelog

| 날짜 | 내용 |
|------|------|
| 2025-04-05 | 초기 6개 실험 (입력 1d/2d/3d × 출력 12h/24h) 결과 기록 |
| 2025-04-05 | 24h 과적합 해결 실험 계획 수립 (A1~A5, B1~B2) |
| 2025-04-05 | B1: ReduceOnPlateau 스케줄러 버그 수정 (train_loss→val_loss) |
| 2025-04-05 | B2: Gaussian noise augmentation 구현 (pipeline.py + base.yaml) |
| 2025-04-05 | 우선순위 4 추가: ap30 이산값 매핑 (snap 후처리 → ordinal classification 단계별) |
| 2025-04-06 | A1~A5, B2 실험 결과 기록. A5(lr=5e-5)가 최고 — best epoch=11, 과적합 효과적 억제 |
| 2025-04-06 | 우선순위 4 상세화: 이산값 매핑 방법 (1)후처리 snap, (2A~2C)모델 변경 분석. (1) 즉시 적용 권장, (2)는 현 시점 비권장 |
| 2025-04-06 | C1~C3 실험 결과 기록. C2(A5+noise)가 전체 최고 — val_loss=0.603. 모델 축소는 underfitting 확인 |
| 2025-04-06 | Next Steps 추가: 12h Cascade 구현 계획 (Oracle→Persistence 단계별), TCN 실험, Attention 분석 등 |
| 2025-04-06 | TCN 코드 검증 완료 (causal padding 정상). TCN config 3개 생성 (12h, 24h, 24h deep) |
| 2025-04-06 | 문헌 조사: SYMHnet(GNN, Abduallah 2024), SINet(TimesNet, Wang 2026), Billcliff(Hp30, 2026) |
| 2025-04-06 | GNN/TimesNet 구현 계획 수립 — 선행 연구 기반, 우리 데이터에 맞게 설계 |
| 2025-04-06 | GNN 구현 완료: GNNEncoder + GNNOnlyModel (순수 PyTorch). 3종 temporal encoder (Transformer/TCN/BiLSTM) 플러그인. config 6개 생성 |
| 2025-04-06 | TimesNet 구현 완료: TimesBlock + TimesNetEncoder + cross-variable attention 확장. config 2개 생성 |
