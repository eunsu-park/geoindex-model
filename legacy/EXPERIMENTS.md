# Experiment Log

Solar Wind Prediction Model (Ap Index Regression) 실험 기록

---

## Summary Table

| Version | Model | MAE | RMSE | R² | Cosine Sim | Best Epoch | Notes |
|---------|-------|-----|------|-----|------------|------------|-------|
| v2 | baseline | **0.673** | 0.881 | **+0.052** | 0.319 | 1 | Best baseline |
| v2 | fusion | **0.670** | **0.884** | +0.044 | 0.387 | 6 | Best fusion |
| v3 | baseline | 0.680 | 0.894 | +0.024 | 0.461 | 1 | Lower LR |
| v3 | fusion | 0.737 | 0.942 | -0.085 | 0.509 | 3 | Lower LR |
| v4 | baseline | 0.672 | 0.884 | +0.046 | 0.404 | 1 | Lightweight |
| v4 | fusion | 0.701 | 0.899 | +0.012 | 0.435 | 2 | Lightweight |
| v5 | baseline | 0.675 | 0.883 | +0.049 | 0.041 | 1 | Two-Stage |
| v5 | fusion | 0.665 | 0.906 | -0.003 | -0.005 | - | Two-Stage (failed) |
| v6 | baseline | 0.674 | 0.886 | +0.041 | 0.379 | 1 | Contrastive Warmup |
| v6 | fusion | 0.696 | 0.895 | +0.022 | 0.393 | 5 | Contrastive Warmup |
| v7 | baseline | **0.672** | 0.884 | +0.046 | **0.404** | 1 | Solar Wind Weighted |
| v7 | fusion | 0.701 | 0.899 | +0.012 | **0.435** | 2 | Solar Wind Weighted |
| v8 | baseline | 0.680 | 0.888 | +0.038 | 0.049 | - | Strong Reg + No Contrastive (실패) |
| v8 | fusion | 0.696 | 0.901 | +0.008 | 0.161 | - | Strong Reg + No Contrastive (실패) |
| v9 | transformer | 0.692 | 0.905 | -0.000 | N/A | 1 | OMNI-only Transformer (실패) |
| v10 | linear | 0.698 | 0.903 | +0.004 | N/A | 1 | OMNI-only Linear (실패) |
| v11b | baseline | **0.672** | 0.884 | +0.046 | **0.404** | 1 | Phase 1: LR Warmup + Cosine |
| v11c | baseline | **0.672** | 0.884 | +0.046 | **0.404** | 1 | Phase 1: Cosine only |
| v13 | tcn | 0.801 | 0.962 | -1.159 | N/A | 1 | TCN default (k=3, 3L, RF=29) |
| v13b | tcn | 0.742 | 0.908 | -0.787 | N/A | 1 | TCN deeper (k=3, 4L, RF=61) |
| v13c | tcn | **0.707** | 0.870 | -0.600 | N/A | 1 | TCN larger kernel (k=5, RF=57) |

---

## v2: Baseline Configuration (2026-02-06)

### Changes from v1
- AP_TIERS weights reduced: 4/8/16 → 2/4/8
- Dropout: 0.2 → 0.1
- lambda_contrastive: 0.3 → 0.5
- weight_decay: 0.0001 → 0.0
- Undersampling: disabled

### Config
```yaml
model:
  d_model: 256
  transformer_num_layers: 3
  convlstm_hidden_channels: 64
training:
  learning_rate: 0.0002
  early_stopping_patience: 10
  lambda_contrastive: 0.5
sampling:
  enable_undersampling: false
```

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v2 | 0.673 | 0.881 | +0.052 | 0.319 | 1 |
| fusion_v2 | 0.670 | 0.884 | +0.044 | 0.387 | 6 |

### Analysis
- R²가 처음으로 양수로 전환 (모델이 평균보다 나은 예측)
- Bias 문제 해결 (-0.03)
- 여전히 빠른 오버피팅 (Best epoch 1-6)
- Contrastive loss collapse 지속

---

## v3: Lower Learning Rate (2026-02-06)

### Changes from v2
- learning_rate: 0.0002 → 0.0001
- early_stopping_patience: 10 → 5

### Config
```yaml
training:
  learning_rate: 0.0001
  early_stopping_patience: 5
```

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v3 | 0.680 | 0.894 | +0.024 | 0.461 | 1 |
| fusion_v3 | 0.737 | 0.942 | -0.085 | 0.509 | 3 |

### Analysis
- Cosine similarity 개선 (+32~44%)
- MAE/R² 성능 저하
- 결론: v2가 더 좋은 균형

---

## v4: Lightweight Model (2026-02-07)

### Changes from v2
- d_model: 256 → 128
- transformer_nhead: 8 → 4
- transformer_num_layers: 3 → 2
- transformer_dim_feedforward: 512 → 256
- convlstm_hidden_channels: 64 → 32

### Config
```yaml
model:
  d_model: 128
  transformer_nhead: 4
  transformer_num_layers: 2
  transformer_dim_feedforward: 256
  convlstm_hidden_channels: 32
training:
  learning_rate: 0.0002
  early_stopping_patience: 10
```

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v4 | 0.672 | 0.884 | +0.046 | 0.404 | 1 |
| fusion_v4 | 0.701 | 0.899 | +0.012 | 0.435 | 2 |

### Analysis
- Baseline 성능 유지 (v2와 동등)
- Contrastive collapse 지연 (cont_loss 0.02 수준 유지)
- Cosine similarity 개선 (0.32 → 0.40)
- Fusion 성능 약간 저하 (MAE +5%)
- 파라미터 수 약 50% 감소

---

## v5: Two-Stage Training (2026-02-07)

### Approach
Stage 1과 Stage 2를 분리하여 학습:
- **Stage 1**: Contrastive pre-training (regression 비활성)
- **Stage 2**: Regression fine-tuning (contrastive 비활성, Stage 1 weights 로드)

### Config
```yaml
# Stage 1
training:
  regression_loss_type: none
  lambda_contrastive: 1.0
  epochs: 15

# Stage 2
training:
  regression_loss_type: solar_wind_weighted
  lambda_contrastive: 0.0
  epochs: 30
  pretrained_checkpoint: "*_twostage_s1/checkpoint/model_best.pth"
```

### Code Changes
- `losses.py`: `regression_loss_type="none"` 지원 추가
- `train.py`: `pretrained_checkpoint` 로딩 추가
- `trainers.py`: criterion=None 처리 추가
- `base.yaml`: `pretrained_checkpoint` 설정 추가

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_twostage | 0.675 | 0.883 | +0.049 | 0.041 | 1 |
| fusion_twostage | 0.665 | 0.906 | -0.003 | -0.005 | - |

### Analysis
- **Baseline Two-Stage**: MAE/R² 성능 v2와 유사, 하지만 Cosine Sim 크게 저하 (0.319 → 0.041)
- **Fusion Two-Stage**: **실패** - Stage 1에서 contrastive 학습 안됨 (cosine_sim = 0.0)
  - train_loss 변화 없음 (1.27 유지)
  - cross-modal attention이 contrastive-only로는 학습되지 않음

### Conclusion
- Two-Stage 접근법은 이 문제에 비효과적
- 단일 학습 (v2)이 더 나은 결과
- Contrastive와 Regression을 함께 학습하는 것이 representation 품질 유지에 중요

---

## v6: Contrastive Warmup (2026-02-08)

### Approach
Contrastive loss 가중치를 점진적으로 감소:
- **Warmup Phase (1-10 epochs)**: λ = 1.0 → 0.2 (선형 감소)
- **After Warmup**: λ = 0.2 고정

### Config
```yaml
training:
  contrastive_warmup:
    enable: true
    warmup_epochs: 10
    lambda_start: 1.0
    lambda_end: 0.2
```

### Code Changes
- `trainers.py`: `_update_lambda_contrastive()` 메서드 추가
- `base.yaml`: `contrastive_warmup` 설정 추가

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v6 | 0.674 | 0.886 | +0.041 | 0.379 | 1 |
| fusion_v6 | 0.696 | 0.895 | +0.022 | 0.393 | 5 |

### Comparison with v2 (Best)
| Metric | baseline (v2→v6) | fusion (v2→v6) |
|--------|------------------|----------------|
| MAE | 0.673 → 0.674 (동등) | 0.670 → 0.696 (↓악화) |
| R² | +0.052 → +0.041 (↓악화) | +0.044 → +0.022 (↓악화) |
| Cosine Sim | 0.319 → 0.379 (↑개선) | 0.387 → 0.393 (↑개선) |

### Analysis
- **Cosine Similarity 개선**: baseline +18.8%, fusion +1.6%
- **회귀 성능 저하**: MAE/R² 모두 v2 대비 악화
- 높은 초기 λ (1.0)가 회귀 학습 초기에 방해
- Contrastive loss가 초기에 과도하게 dominant

### Conclusion
- Contrastive Warmup은 representation 품질은 개선하나 회귀 성능 저하
- v2의 고정 λ=0.5가 여전히 최적
- **v2 유지**

---

## v7: Solar Wind Weighted Loss (2026-02-08)

### Approach
NOAA 지자기 활동 기준(G-Scale)에 따라 Ap 값에 가중치를 부여하는 Loss 함수 적용.
희귀한 고활동 이벤트(지자기 폭풍)에 더 높은 가중치를 부여하여 예측 정확도 향상 목표.

**AP_TIERS (NOAA G-Scale based, 4 levels):**
| Tier | Ap Range | Weight | Description |
|------|----------|--------|-------------|
| none | 0-29 | 1.0 | No storm (Kp < 5) |
| G1 | 30-49 | 2.0 | Minor Storm (Kp 5) |
| G2 | 50-99 | 4.0 | Moderate Storm (Kp 6) |
| G3+ | 100+ | 8.0 | Strong-Extreme (Kp 7-9) |

### Config
```yaml
training:
  regression_loss_type: "solar_wind_weighted"
  solar_wind_weighted:
    base_loss: "mse"
    weighting_mode: "multi_tier"
    combine_temporal: true
    temporal_weight_range: [0.5, 1.0]
```

### Code
- `src/losses.py`: `SolarWindWeightedLoss` 클래스 (이미 구현됨)
- Denormalization 지원: normalized target을 raw Ap로 변환 후 tier 할당

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v7 | **0.672** | 0.884 | +0.046 | 0.404 | 1 |
| fusion_v7 | 0.701 | 0.899 | +0.012 | 0.435 | 2 |

### Comparison with v2 (Best)
| Metric | baseline (v2→v7) | fusion (v2→v7) |
|--------|------------------|----------------|
| MAE | 0.673 → **0.672** (↑개선) | 0.670 → 0.701 (↓악화) |
| RMSE | 0.881 → 0.884 (동등) | 0.884 → 0.899 (↓악화) |
| R² | +0.052 → +0.046 (↓악화) | +0.044 → +0.012 (↓악화) |
| Cosine Sim | 0.319 → 0.404 (**+26.6%**) | 0.387 → 0.435 (**+12.4%**) |

### Analysis
- **Baseline MAE 개선**: baseline_v7이 v2 baseline MAE를 개선 (0.673 → 0.672)
- **Cosine Similarity 대폭 개선**: baseline +26.6%, fusion +12.4%
- **Fusion 회귀 성능 저하**: fusion 모델은 MAE/R² 모두 악화
- **v4와 동일 결과**: lightweight 모델 + solar_wind_weighted 조합이 v4와 동일
- **빠른 오버피팅 지속**: Best epoch 1-2

### Conclusion
- **baseline_v7 = baseline_v4 = 최고 baseline MAE** (0.672)
- Cosine Similarity 측면에서 큰 개선 (+26.6%)
- Fusion 모델은 여전히 v2 대비 성능 저하
- Solar Wind Weighted Loss + Lightweight 모델 조합이 baseline에는 효과적

---

## v8: Strong Regularization + No Contrastive (2026-02-09)

### Approach
오버피팅 해결 및 Fusion 성능 개선을 위한 두 가지 변경:
1. **Regularization 강화**: weight_decay, dropout 증가
2. **Contrastive Loss 제거**: regression 학습에 집중

**가설:**
- 빠른 오버피팅(Best epoch 1-2)은 regularization 부족 때문
- Contrastive loss가 fusion 모델의 regression 학습을 방해

### Config
```yaml
training:
  weight_decay: 0.01        # 0.0 → 0.01
  lambda_contrastive: 0.0   # 0.5 → 0.0 (제거)
model:
  transformer_dropout: 0.2  # 0.1 → 0.2
  fusion_dropout: 0.2       # 0.1 → 0.2
  baseline_dropout: 0.2     # 0.1 → 0.2
```

### Code
- 코드 변경 없음 (config만 변경)

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| baseline_v8 | 0.680 | 0.888 | +0.038 | 0.049 | - |
| fusion_v8 | 0.696 | 0.901 | +0.008 | 0.161 | - |

### Comparison with v7
| Metric | baseline (v7→v8) | fusion (v7→v8) |
|--------|------------------|----------------|
| MAE | 0.672 → 0.680 (↓악화) | 0.701 → 0.696 (↑소폭 개선) |
| RMSE | 0.884 → 0.888 (↓악화) | 0.899 → 0.901 (동등) |
| R² | +0.046 → +0.038 (↓악화) | +0.012 → +0.008 (↓악화) |
| Cosine Sim | 0.404 → **0.049** (↓↓-88%) | 0.435 → **0.161** (↓↓-63%) |

### Analysis
- **Cosine Similarity 대폭 하락**: Contrastive loss 제거가 representation 품질에 치명적
  - baseline: 0.404 → 0.049 (-88%)
  - fusion: 0.435 → 0.161 (-63%)
- **회귀 성능 개선 없음**: Strong regularization이 오버피팅 해결에 효과 없음
  - baseline MAE 오히려 악화
  - fusion MAE 소폭 개선되었으나 v2 대비 여전히 열등
- **가설 기각**: "Contrastive loss가 fusion regression을 방해한다"는 가설이 틀림
  - Contrastive loss 없이도 fusion 성능 개선 없음
  - 오히려 representation 품질만 크게 저하

### Conclusion
- **v8 실패**: Contrastive loss 제거는 부정적 영향만 미침
- Contrastive loss는 representation 품질 유지에 필수 (λ=0.5 유지)
- **v2 유지**

---

## v9: OMNI-only Transformer (2026-02-09)

### Approach
Ablation 분석 결과(SDO가 성능 저하 유발)를 바탕으로 **OMNI만 사용하는 Transformer 모델** 훈련.

**가설:**
- SDO 제거 시 baseline ablation에서 MAE -1.2%, R² +41% 개선
- 처음부터 OMNI만으로 훈련하면 더 좋은 성능 가능

### Config
```yaml
model:
  model_type: transformer  # TransformerOnlyModel
  d_model: 256
  transformer_nhead: 8
  transformer_num_layers: 3
  transformer_dim_feedforward: 512
  transformer_dropout: 0.1
training:
  lambda_contrastive: 0.0  # 단일 모달, contrastive 불필요
sampling:
  enable_undersampling: false
```

### Code
- 코드 변경 없음 (기존 TransformerOnlyModel 사용)

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| transformer_v9 | 0.692 | 0.905 | -0.000 | N/A | 1 |

### Comparison
| Model | MAE | R² | 비고 |
|-------|-----|-----|-----|
| baseline_v2 (OMNI only ablation) | **0.6655** | **+0.0736** | 목표 |
| baseline_v2 (Full) | 0.6733 | +0.0522 | 비교군 |
| transformer_v9 | 0.6916 | -0.0003 | **목표 미달** |

### Analysis
1. **목표 미달**: transformer_v9가 baseline_v2 OMNI ablation보다 MAE +3.9% 나쁨
2. **R² 거의 0**: 평균 예측 수준의 성능, representation 학습 실패
3. **Best epoch = 1**: 첫 epoch 이후 오버피팅, 학습 불안정

### Conclusion
- **v9 실패**: OMNI-only Transformer가 ablation 성능을 재현하지 못함
- Ablation은 이미 훈련된 OMNI encoder 사용, v9는 처음부터 OMNI만으로 훈련
- **Multi-task 효과**: SDO와 함께 훈련 시 OMNI encoder가 더 좋은 representation 학습

---

## v10: OMNI-only Linear (2026-02-09)

### Approach
Transformer 대신 간단한 Linear encoder로 OMNI-only 모델 훈련.
Ablation에서 baseline의 LinearEncoder가 좋은 성능을 보였으므로 재현 시도.

### Config
```yaml
model:
  model_type: linear  # LinearOnlyModel
  d_model: 256
training:
  lambda_contrastive: 0.0
sampling:
  enable_undersampling: false
```

### Results
| Model | MAE | RMSE | R² | Cosine Sim | Best Epoch |
|-------|-----|------|-----|------------|------------|
| linear_v10 | 0.698 | 0.903 | +0.004 | N/A | 1 |

### Comparison
| Model | MAE | R² | 비고 |
|-------|-----|-----|-----|
| baseline_v2 (OMNI only ablation) | **0.6655** | **+0.0736** | 목표 |
| linear_v10 | 0.6977 | +0.0035 | **목표 미달** |
| transformer_v9 | 0.6916 | -0.0003 | 비교군 |

### Analysis
1. **목표 미달**: linear_v10이 baseline_v2 OMNI ablation보다 MAE +4.8% 나쁨
2. **transformer_v9보다 약간 나쁨**: MAE 0.698 vs 0.692
3. **Best epoch = 1**: 동일한 오버피팅 패턴

### Conclusion
- **v10 실패**: Linear-only도 ablation 성능 재현 실패
- v9 (Transformer)와 v10 (Linear) 모두 처음부터 OMNI만으로 훈련 시 성능 저하
- **핵심 발견**: Multi-modal 훈련이 OMNI encoder에 유익한 regularization 효과 제공

---

## Cross-Modal Fusion Analysis (2026-02-09)

### 목적
Fusion 모델이 Baseline을 유의미하게 능가하지 못하는 원인 분석.
SDO 이미지가 Ap 예측에 실제로 기여하는지 확인.

### 방법
`analysis/cross_modal_analysis.py` 스크립트를 통해 fusion_v2 모델 분석:
1. **Gate Weight 분석**: CrossModalFusion의 feature_gate sigmoid 출력 추출
   - gate > 0.5: OMNI (Transformer) 우세
   - gate < 0.5: SDO (ConvLSTM) 우세
2. **Feature Norm 분석**: 각 modality에서 추출된 feature의 크기 비교

### Results

| 지표 | 값 | 해석 |
|-----|-----|-----|
| Mean Gate Weight | 0.537 | 53.7% OMNI, 46.3% SDO |
| OMNI Dominant Samples | **99.8%** | 거의 모든 샘플에서 OMNI 선호 |
| OMNI Feature Norm | 1.028 (±0.588) | |
| SDO Feature Norm | 0.264 (±0.050) | |
| OMNI/SDO Norm Ratio | **3.89** | OMNI 피처가 ~4배 강함 |

### Analysis

1. **Feature Magnitude 불균형**
   - SDO 피처가 OMNI 대비 **4배 약함** (0.26 vs 1.03)
   - SDO 정보가 fusion에서 희석됨
   - ConvLSTM이 SDO에서 유용한 피처 추출에 실패

2. **Gate의 일관된 OMNI 선호**
   - 99.8%의 샘플에서 gate > 0.5
   - 모델이 학습을 통해 OMNI를 더 신뢰하도록 수렴
   - SDO 정보를 거의 활용하지 않음

3. **Fusion 효과 부재와 일치**
   - Baseline vs Fusion MAE 차이: 0.003 (0.673 vs 0.670)
   - 분석 결과가 성능 차이 미미함을 설명

### Visualization

결과 저장 위치: `/opt/projects/10_Harim/01_AP/04_Result/fusion_v2/analysis/`
- `gate_distribution.png`: Gate weight 분포
- `feature_norms.png`: Feature magnitude 비교
- `cross_modal_results.npz`: Raw 데이터

### Conclusion

**SDO 이미지가 Ap 예측에 유용한 정보를 제공하지 못함.**

가능한 원인:
- SDO → Ap 관계가 OMNI보다 약하거나 간접적
- ConvLSTM이 SDO에서 유용한 피처 추출 실패
- 7일 과거 SDO 이미지가 1-3일 후 Ap 예측에 충분한 정보 부재

**권장 사항**: Fusion 모델 개선 중단, v2 baseline/fusion 최종 모델로 선정

---

## Baseline Ablation Analysis (2026-02-09)

### 목적
Baseline 모델(Conv3D + Linear)에서 각 modality의 기여도 정량적 측정.
SDO 이미지가 실제로 예측에 도움이 되는지 확인.

### 방법
`analysis/ablation_analysis.py` 스크립트를 통해 baseline_v2 모델 분석:
1. **Full**: SDO + OMNI 둘 다 사용 (기본)
2. **OMNI only**: SDO encoder 출력을 0으로 설정
3. **SDO only**: OMNI encoder 출력을 0으로 설정

### Results

| Mode | MAE | RMSE | R² |
|------|-----|------|-----|
| Full (SDO+OMNI) | 0.6733 | 0.8808 | +0.0522 |
| **OMNI only** | **0.6655** | **0.8708** | **+0.0736** |
| SDO only | 0.7099 | 0.9638 | -0.1347 |

### Analysis

1. **OMNI only가 Full model보다 성능이 좋음 (!!)**
   - MAE: 0.6733 → 0.6655 (**-1.2% 개선**)
   - R²: +0.0522 → +0.0736 (**+41% 개선**)
   - SDO를 제거하면 오히려 성능 향상

2. **SDO only는 성능이 매우 나쁨**
   - R²: -0.1347 (음수, 평균 예측보다 못함)
   - SDO 단독으로는 Ap 예측 불가

3. **SDO 기여도: 음수 (해로움)**
   - SDO 제거 시 MAE -1.2% 감소 (성능 향상)
   - SDO가 OMNI 정보에 노이즈를 추가하고 있음

### Visualization

결과 저장 위치: `/opt/projects/10_Harim/01_AP/04_Result/baseline_v2/analysis/`
- `ablation_results.npz`: Raw 데이터
- `ablation_report.txt`: 텍스트 리포트

### Conclusion

**SDO 이미지가 Ap 예측에 해로운 영향을 미침.**

- SDO를 사용하면 오히려 성능 저하
- OMNI만 사용하는 것이 최적
- Multi-modal fusion의 가치 부재 확인

**핵심 결론:**
1. **7일 과거 SDO 이미지 → 1-3일 후 Ap 예측** 관계는 존재하지 않거나 매우 약함
2. OMNI in-situ 데이터가 Ap 예측에 압도적으로 중요
3. SDO 기반 다중 모달 접근법은 이 문제에 부적합

---

## Key Findings

1. **Undersampling 비활성이 효과적**: 전체 데이터 사용이 더 좋은 성능
2. **AP_TIERS 가중치 감소 필요**: 너무 높은 가중치는 positive bias 유발
3. **R² 양수 달성**: v2 이후 모델이 평균보다 나은 예측 가능
4. **오버피팅 문제**: Best epoch이 1-6으로 매우 빠름
5. **Contrastive collapse**: 두 loss가 경쟁하여 contrastive가 빠르게 감소
6. **Two-Stage 학습 실패**: Cross-modal attention은 contrastive-only로 학습 불가
7. **Contrastive Warmup 비효과적**: 초기 λ가 너무 높으면 회귀 학습 방해
8. **Solar Wind Weighted + Lightweight**: Baseline MAE 최고 (0.672), Cosine Sim +26.6%
9. **Contrastive Loss 필수**: 제거 시 representation 품질 88% 하락, 회귀 성능 개선 없음
10. **SDO 기여도 부재 (Fusion)**: Cross-modal 분석 결과 SDO 피처가 OMNI 대비 4배 약함, 99.8% 샘플에서 OMNI 우세
11. **SDO가 성능 저하 유발 (Baseline)**: Ablation 분석 결과 SDO 제거 시 MAE -1.2%, R² +41% 개선. OMNI only가 최적
12. **Multi-modal 훈련의 Regularization 효과**: v9/v10 실패로 발견. OMNI-only로 처음부터 훈련하면 ablation 성능 재현 불가. SDO와 함께 훈련 시 OMNI encoder가 더 좋은 representation 학습

---

## Next Steps

- [x] v5 (Two-Stage) 결과 확인 → 실패, v2 유지
- [x] Contrastive warmup → 비효과적, v2 유지
- [x] Data augmentation → SDO 이미지 특성상 기하학적 변환 부적합, 미적용
- [x] v7: Solar Wind Weighted Loss → baseline MAE 개선 (0.672), Cosine Sim +26.6%
- [x] v8: Strong Regularization + No Contrastive → 실패, Cosine Sim -88% 하락
- [x] Cross-modal fusion 분석 → SDO 기여도 부재 확인 (OMNI/SDO ratio 3.89)
- [x] Baseline ablation 분석 → **SDO가 성능 저하 유발** (OMNI only MAE 0.6655 vs Full 0.6733)
- [x] v9: OMNI-only Transformer → **실패** (MAE 0.692, ablation 0.666보다 나쁨)
- [x] v10: OMNI-only Linear → **실패** (MAE 0.698, ablation보다 나쁨)
- [ ] ~~Attention mechanism 개선~~ (SDO 기여도 부재로 중단)
- [ ] ~~Fusion 모델 개선 방안 탐색~~ (SDO 기여도 부재로 중단)
- [x] v11: Phase 1 - 오버피팅 해결 시도 → **미해결** (MAE 유지, Cosine Sim +26.7% 개선)
- [ ] ~~v12: Phase 2 - Data Augmentation~~ (진행 안함)
- [x] v13: Phase 3 - TCN Encoder → **실패** (v13c MAE 0.707, baseline 0.672보다 나쁨)
- [ ] ~~v14: Phase 4 - Probabilistic Output~~ (진행 안함)

---

## v11: Phase 1 - Overfitting Fix (2026-02-09)

### Approach
오버피팅 문제(Best epoch = 1) 해결을 위한 4가지 개선 적용:

1. **LR Warmup**: 초기 학습률을 낮게 시작 → 점진적 증가
   - 효과: 학습 초기 불안정성 해소, 더 나은 수렴점 도달
   - 설정: 5 epochs 동안 10% → 100% 선형 증가

2. **Cosine Annealing Scheduler**: 주기적 LR 재시작
   - 효과: local minima 탈출, 더 나은 generalization
   - 설정: T_0=10, T_mult=2 (10 → 20 → 40 epochs)

3. **Gradient Accumulation**: 여러 batch의 gradient 누적
   - 효과: 효과적 batch size 증가, 학습 안정성
   - 설정: 4 steps (effective batch = 16)

4. **Early Stopping Patience 증가**: 15 epochs
   - 효과: 충분한 학습 기회 제공

### Config
```yaml
experiment:
  name: "baseline_v11"

training:
  # LR Warmup (신규)
  lr_warmup:
    enable: true
    warmup_epochs: 5
    warmup_start_factor: 0.1  # 10%에서 시작

  # Cosine Annealing (기존 ReduceLROnPlateau 대체)
  scheduler_type: "cosine_annealing"
  cosine_annealing:
    T_0: 10
    T_mult: 2
    eta_min: 1.0e-6

  # Gradient Accumulation (신규)
  gradient_accumulation_steps: 4  # effective batch = 4 × 4 = 16

  # Early Stopping (증가)
  early_stopping_patience: 15
```

### Code Changes
- `trainers.py`:
  - `_apply_lr_warmup()` 메서드 추가
  - `train_step()`: Gradient accumulation 로직 추가
  - `train_epoch()`: Epoch 끝 gradient 처리, scheduler_type별 step 호출

- `scripts/train.py`:
  - `create_scheduler()`: CosineAnnealingWarmRestarts 지원

- `configs/base.yaml`:
  - `lr_warmup`, `cosine_annealing`, `gradient_accumulation_steps`, `scheduler_type` 설정 추가

### Results

| 실험 | 설정 | MAE | R² | Cosine Sim | Best Epoch |
|------|------|-----|-----|------------|------------|
| v11 | All (fusion 실수) | 0.717 | -0.051 | 0.114 | 18 |
| **v11b** | LR Warmup + Cosine | **0.672** | +0.046 | **0.404** | 1 |
| **v11c** | Cosine only | **0.672** | +0.046 | **0.404** | 1 |

### Analysis

1. **오버피팅 미해결**: Best Epoch = 1 유지
2. **성능 유지**: MAE 0.672 (v2와 동등)
3. **Cosine Similarity 개선**: 0.319 → 0.404 (+26.7%)

### Conclusion

Phase 1 개선사항(LR Warmup, Cosine Annealing)은 오버피팅을 해결하지 못함.
그러나 Cosine Similarity가 크게 개선되어 cross-modal representation 품질은 향상됨.

---

## v13: Phase 3 - TCN Encoder (2026-02-13)

### Approach
OMNI-only 모델의 대안으로 TCN (Temporal Convolutional Network) 인코더 도입:
- Dilated causal convolution으로 장거리 의존성 포착
- Transformer 대비 경량 (~480K vs ~1.7M params)
- Causal 구조로 temporal leakage 방지

### Variants
| 실험 | Channels | Kernel | Receptive Field | Params |
|------|----------|--------|-----------------|--------|
| v13 | [64,128,256] | 3 | 29 | ~480K |
| v13b | [64,128,256,512] | 3 | 61 | ~2.1M |
| v13c | [64,128,256] | 5 | 57 | ~870K |

### Results

| 실험 | MAE | RMSE | R² | Best Epoch |
|------|-----|------|-----|------------|
| tcn_v13 | 0.801 | 0.962 | -1.159 | 1 |
| tcn_v13b | 0.742 | 0.908 | -0.787 | 1 |
| **tcn_v13c** | **0.707** | **0.870** | **-0.600** | 1 |

### Analysis

1. **v13c (larger kernel)가 TCN 중 최고**: kernel size=5가 효과적
2. **여전히 baseline보다 낮음**: MAE 0.707 vs baseline_v7 0.672 (+5.2%)
3. **OMNI-only 모델 한계 재확인**:
   - transformer_v9: 0.692
   - linear_v10: 0.698
   - tcn_v13c: 0.707
4. **RF 크기 ≠ 성능**: v13c(RF=57) > v13b(RF=61) → kernel이 depth보다 중요

### Conclusion

TCN은 Transformer/Linear보다 나쁜 결과. OMNI-only 접근의 근본적 한계 확인.
Multi-modal 학습 (SDO+OMNI)이 필요.

---

## Final Model Selection

**실험 종료.**

### 전체 비교

| Model | MAE | RMSE | R² | 비고 |
|-------|-----|------|-----|-----|
| baseline_v2 (Full) | 0.673 | 0.881 | +0.052 | SDO+OMNI |
| **baseline_v2 (OMNI ablation)** | **0.666** | **0.871** | **+0.074** | **최고 성능** |
| baseline_v7 | 0.672 | 0.884 | +0.046 | Solar Wind Weighted |
| fusion_v2 | 0.670 | 0.884 | +0.044 | SDO+OMNI |
| transformer_v9 | 0.692 | 0.905 | -0.000 | OMNI-only (실패) |
| linear_v10 | 0.698 | 0.903 | +0.004 | OMNI-only (실패) |
| tcn_v13c | 0.707 | 0.870 | -0.600 | OMNI-only TCN (실패) |

### 결론

**baseline_v2의 OMNI encoder (ablation)가 최고 성능** (MAE 0.666, R² +0.074)

**핵심 발견:**
1. SDO 이미지가 Ap 예측에 해로운 영향 (inference 시 제거 권장)
2. 그러나 **훈련 시에는 SDO 필요** (Multi-modal 훈련의 regularization 효과)
3. OMNI-only로 처음부터 훈련하면 성능 저하 (Transformer, Linear, TCN 모두 실패)

### 권장 사항

1. **Production용**: baseline_v2를 SDO+OMNI로 훈련 후, **inference 시 OMNI encoder만 사용**
2. **논문용**:
   - Ablation 분석 결과 포함
   - SDO는 inference에 불필요하나, 훈련에는 유익 (regularization)
   - Multi-modal 훈련 → Single-modal inference 전략 제안
3. **향후 연구**:
   - 다른 예측 타겟(Dst, Kp 등)에서 SDO 유용성 검증
   - Multi-task learning의 regularization 효과 분석
