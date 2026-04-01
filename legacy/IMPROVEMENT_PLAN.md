# Project Improvement Plan

Solar Wind Prediction (Ap Index Regression) 프로젝트 개선 계획

---

## 현재 상태 요약

| 지표 | 현재 최고 성능 | 목표 |
|-----|---------------|------|
| MAE | 0.666 (ablation) | < 0.60 |
| R² | +0.074 | > 0.20 |
| Best Epoch | 1 (심한 오버피팅) | 10+ |

**핵심 발견:**
1. SDO 이미지는 inference에 해롭지만, 훈련 시 regularization 효과
2. OMNI 데이터가 Ap 예측에 압도적으로 중요
3. 심한 오버피팅 (Best epoch = 1)
4. R²가 낮음 (설명력 ~7%)

---

## 1. 데이터 준비 측면

### 1.1 OMNI 데이터 개선

#### A. 입력 변수 확장
현재 12개 변수 → 추가 변수 고려:

| 추가 변수 | 설명 | 예상 효과 |
|----------|------|----------|
| `F10.7` | Solar radio flux | 태양 활동 직접 지표 |
| `Dst_index` | Disturbance storm time | 지자기 폭풍 관련 |
| `Kp_index` | Planetary K index | Ap와 직접 연관 |
| `proton_flux` | Solar proton flux | 태양 입자 이벤트 |
| `AE_index` | Auroral electrojet | 오로라 활동 지표 |

```python
# configs/base.yaml 수정
data:
  input_variables:
    - Bx, By, Bz, V, Np, Tp  # 기존
    - AE, AL, AU, PC_N, SYM_D, SYM_H  # 기존
    - F10_7, Dst, proton_flux  # 추가
```

#### B. 시간 해상도 증가
현재: 6시간 간격 (56 timesteps = 14일)
개선: 1시간 간격으로 더 세밀한 패턴 포착

```python
# 기존
input_sequence_length: 56  # 14일 × 4 (6시간 간격)

# 개선
input_sequence_length: 168  # 7일 × 24 (1시간 간격)
```

#### C. 입력 윈도우 확장
현재: 14일 과거 → 21일 또는 27일 (태양 자전 주기)로 확장

### 1.2 타겟 변수 개선

#### A. 다중 타겟 예측 (Multi-target)
Ap만 예측 → Ap, Kp, Dst 동시 예측으로 representation 향상

```python
data:
  target_variables:
    - Ap
    - Kp
    - Dst  # 추가
```

#### B. 타겟 변환
Ap 분포가 heavily skewed → Log 변환 또는 Box-Cox 변환

```python
# Log 변환
target = np.log1p(ap_index)

# 예측 후 역변환
prediction = np.expm1(model_output)
```

### 1.3 데이터 증강

#### A. OMNI 시계열 증강
- **Jittering**: 작은 노이즈 추가
- **Scaling**: 특정 변수 스케일 변형
- **Time warping**: 시간축 비선형 변환 (DTW 기반)
- **Window slicing**: 부분 윈도우 샘플링

```python
class OMNIAugmentation:
    def jitter(self, x, sigma=0.01):
        return x + np.random.normal(0, sigma, x.shape)

    def scale(self, x, sigma=0.1):
        factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
        return x * factor
```

#### B. Mixup / CutMix
두 샘플을 선형 조합하여 데이터 증강

```python
def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```

---

## 2. 네트워크 구조 측면

### 2.1 Temporal Modeling 개선

#### A. Temporal Convolutional Network (TCN)
Transformer 대신 TCN 사용 - dilated causal convolution

```python
class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(
                in_channels=num_inputs if i == 0 else num_channels[i-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
```

**장점:**
- Transformer보다 가벼움
- Long-range dependency 포착
- Causal structure로 temporal leakage 방지

#### B. Informer / Autoformer
Long sequence prediction에 특화된 Transformer 변형

```python
class InformerEncoder(nn.Module):
    """ProbSparse self-attention for O(L log L) complexity."""
    pass
```

#### C. LSTM + Attention Hybrid
LSTM의 sequential modeling + Attention의 global context

```python
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
```

### 2.2 Output Head 개선

#### A. Probabilistic Output
Point prediction → Distribution prediction (Uncertainty quantification)

```python
class ProbabilisticHead(nn.Module):
    """Predict mean and variance for Gaussian output."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.var_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mean = self.mean_head(x)
        var = F.softplus(self.var_head(x))  # Ensure positive
        return mean, var
```

**Loss: Negative Log-Likelihood**
```python
def nll_loss(mean, var, target):
    return 0.5 * (torch.log(var) + (target - mean)**2 / var)
```

#### B. Quantile Regression
특정 분위수 예측으로 불확실성 표현

```python
class QuantileHead(nn.Module):
    """Predict multiple quantiles (0.1, 0.5, 0.9)."""
    def __init__(self, input_dim, output_dim, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in quantiles
        ])
```

### 2.3 Regularization 강화

#### A. Dropout Scheduling
훈련 초기에 높은 dropout → 점진적 감소

```python
def get_dropout_rate(epoch, max_epochs, initial=0.5, final=0.1):
    return initial - (initial - final) * (epoch / max_epochs)
```

#### B. Weight Decay Scheduling
Epoch에 따라 weight decay 조절

#### C. Spectral Normalization
Lipschitz constraint로 모델 안정화

```python
from torch.nn.utils import spectral_norm

self.fc = spectral_norm(nn.Linear(256, 128))
```

---

## 3. 훈련 전략 측면

### 3.1 오버피팅 해결 (최우선)

#### A. Early Stopping with Patience
현재: Best epoch = 1 → 학습 불안정

```python
# 개선된 early stopping
early_stopping:
  patience: 10
  min_delta: 0.001
  restore_best_weights: true
```

#### B. Learning Rate Warmup
초기 학습률을 낮게 시작 → 점진적 증가

```python
def warmup_lr(epoch, warmup_epochs=5, base_lr=1e-3):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr
```

#### C. Gradient Accumulation
Batch size 증가 효과로 학습 안정화

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3.2 Loss Function 개선

#### A. Huber Loss (Smooth L1)
Outlier에 robust한 loss

```python
criterion = nn.HuberLoss(delta=1.0)
```

#### B. Asymmetric Loss
과대예측 vs 과소예측에 다른 페널티

```python
def asymmetric_loss(pred, target, alpha=0.7):
    """Penalize under-prediction more (alpha > 0.5)."""
    error = target - pred
    loss = torch.where(error > 0, alpha * error**2, (1-alpha) * error**2)
    return loss.mean()
```

#### C. Focal Loss for Regression
어려운 샘플에 집중

```python
def focal_mse_loss(pred, target, gamma=2.0):
    mse = (pred - target) ** 2
    weight = (1 + mse) ** gamma
    return (weight * mse).mean()
```

### 3.3 Learning Rate Schedule

#### A. Cosine Annealing with Warm Restarts

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

#### B. OneCycleLR

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=len(train_loader)
)
```

### 3.4 Multi-modal 훈련 전략 개선

#### A. Auxiliary Loss
SDO encoder에 별도 task 부여 (e.g., solar activity classification)

```python
# Main task: Ap regression
# Auxiliary task: Solar flare classification from SDO
total_loss = regression_loss + 0.1 * auxiliary_classification_loss
```

#### B. Gradient Reversal Layer
SDO encoder가 task-irrelevant feature를 학습하지 않도록

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

---

## 4. 그 외 개선 사항

### 4.1 Ensemble Methods

#### A. Model Ensemble
여러 모델의 예측 평균

```python
models = [baseline_v2, baseline_v7, fusion_v2]
predictions = [m(x) for m in models]
ensemble_pred = torch.stack(predictions).mean(dim=0)
```

#### B. Snapshot Ensemble
단일 훈련에서 여러 모델 저장

```python
# Cosine annealing으로 여러 local minima 도달
# 각 cycle 끝에서 모델 저장
```

### 4.2 Post-processing

#### A. Temporal Smoothing
예측 결과에 moving average 적용

```python
def smooth_predictions(pred, window=3):
    return np.convolve(pred, np.ones(window)/window, mode='same')
```

#### B. Calibration
예측값 보정으로 bias 제거

```python
# Isotonic regression for calibration
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrated = calibrator.fit_transform(val_pred, val_target)
```

### 4.3 Evaluation 개선

#### A. Storm-specific Evaluation
전체 MAE 뿐 아니라 지자기 폭풍 시기 성능 별도 평가

```python
def evaluate_by_activity(pred, target, ap_threshold=30):
    storm_mask = target > ap_threshold
    quiet_mask = ~storm_mask

    storm_mae = np.abs(pred[storm_mask] - target[storm_mask]).mean()
    quiet_mae = np.abs(pred[quiet_mask] - target[quiet_mask]).mean()

    return {'storm_mae': storm_mae, 'quiet_mae': quiet_mae}
```

#### B. Lead-time Analysis
예측 시간대별 성능 분석 (Day 1, Day 2, Day 3)

```python
def evaluate_by_lead_time(pred, target):
    # pred, target: (batch, 24, 1) = 3일 × 8 timesteps
    day1 = slice(0, 8)
    day2 = slice(8, 16)
    day3 = slice(16, 24)

    return {
        'day1_mae': mae(pred[:, day1], target[:, day1]),
        'day2_mae': mae(pred[:, day2], target[:, day2]),
        'day3_mae': mae(pred[:, day3], target[:, day3]),
    }
```

### 4.4 Interpretability

#### A. SHAP Values
각 입력 변수의 기여도 분석

```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)
```

#### B. Attention Visualization
Transformer attention weight 분석 (이미 구현됨)

---

## 5. 구현 우선순위

### Phase 1: 오버피팅 해결 (1주) ✅ COMPLETED (효과 없음)
1. [x] Learning rate warmup 구현 - `trainers.py`: `_apply_lr_warmup()` 메서드 추가
2. [x] Cosine annealing scheduler 적용 - `train.py`: `CosineAnnealingWarmRestarts` 지원
3. [x] Gradient accumulation 구현 - `trainers.py`: `train_step()` 수정
4. [x] Early stopping patience 증가 - `base.yaml`: 10으로 설정됨

**결과**: 오버피팅 미해결 (Best Epoch = 1 유지), MAE 유지, Cosine Sim +26.7% 개선

### Phase 2: 데이터 개선 ❌ SKIPPED
- Data Augmentation (Jittering, Scaling, Mixup) 진행 안함
- Target Log transformation: 이미 `log1p_zscore` 정규화로 구현됨

### Phase 3: 모델/Loss 개선 (진행 중)
1. [x] **Weighted Loss 구현** ✅ COMPLETED
   - `GeneralWeightedMSELoss`: 일반적인 threshold 기반 가중치 MSE
   - `SolarWindWeightedLoss`: NOAA G-Scale 기반 다중 티어 가중치
     - 4-tier (none/G1/G2/G3+): weight = 1/2/4/8
     - threshold, continuous, multi_tier 모드 지원
     - temporal weight 조합 지원 (미래 시점 강조)
     - denormalization 지원 (정규화된 target을 raw Ap로 변환)
   - Config: `training.regression_loss_type: "solar_wind_weighted"`
   - Unit tests: 28개 테스트 통과
2. [x] **TCN Encoder 구현** ✅ COMPLETED
   - `TemporalBlock`: Dilated causal convolution block
   - `TCNEncoder`: 다층 TCN encoder with exponential dilation
   - `TCNOnlyModel`: OMNI-only 모델 (Transformer 대안)
   - 특징:
     - Causal convolution (temporal leakage 방지)
     - Exponential dilation (2^i for layer i)
     - Weight normalization for stability
     - Configurable receptive field
   - Default config: 3 layers [64, 128, 256], kernel=3 → RF=29
   - Config: `model.model_type: "tcn"`
   - Unit tests: 15개 테스트 통과
   - Parameters: ~480K (Transformer ~1.7M 대비 경량)
   - Trainer integration: `trainers.py` train_step/validate_step 수정 완료
3. [ ] Probabilistic output head 구현
4. [x] Huber loss: 이미 지원됨 (`training.regression_loss_type: "huber"`)

### Phase 4: 평가 개선 (1주)
1. [ ] Storm-specific evaluation 구현
2. [ ] Lead-time analysis 구현
3. [ ] SHAP 분석 적용

---

## 6. 예상 성능 목표

| Phase | MAE 목표 | R² 목표 | Best Epoch |
|-------|---------|---------|------------|
| 현재 | 0.666 | +0.074 | 1 |
| Phase 1 | 0.660 | +0.08 | 5+ |
| Phase 2 | 0.640 | +0.12 | 10+ |
| Phase 3 | 0.600 | +0.20 | 15+ |
| Phase 4 | 0.580 | +0.25 | 15+ |

---

## 7. 실험 계획

### v11: Overfitting Fix
```yaml
training:
  lr_warmup_epochs: 5
  scheduler: cosine_annealing_warm_restarts
  gradient_accumulation_steps: 4
  early_stopping_patience: 15
```

### v12: Solar Wind Weighted Loss (준비됨)
```yaml
training:
  regression_loss_type: "solar_wind_weighted"
  solar_wind_weighted:
    base_loss: "mse"
    weighting_mode: "multi_tier"  # NOAA G-Scale (1/2/4/8)
    combine_temporal: true
    temporal_weight_range: [0.5, 1.0]
    denormalize: true  # Use raw Ap values for tier boundaries
```

### v12b: Weighted Loss - threshold mode
```yaml
training:
  regression_loss_type: "solar_wind_weighted"
  solar_wind_weighted:
    weighting_mode: "threshold"
    threshold: 30.0  # G1 storm boundary
    high_weight: 10.0
```

### v12c: Weighted Loss - continuous mode
```yaml
training:
  regression_loss_type: "solar_wind_weighted"
  solar_wind_weighted:
    weighting_mode: "continuous"
    alpha: 5.0
    beta: 1.5
```

### v13: TCN Encoder (준비됨)
```yaml
model:
  model_type: tcn
  tcn_channels: [64, 128, 256]  # 3-layer TCN
  tcn_kernel_size: 3             # Receptive field = 29 timesteps
  tcn_dropout: 0.1
```

### v13b: TCN with deeper network
```yaml
model:
  model_type: tcn
  tcn_channels: [64, 128, 256, 512]  # 4-layer TCN
  tcn_kernel_size: 3                  # Receptive field = 57 timesteps
```

### v13c: TCN with larger kernel
```yaml
model:
  model_type: tcn
  tcn_channels: [64, 128, 256]
  tcn_kernel_size: 5              # Receptive field = 57 timesteps
```

### v14: Probabilistic Output
```yaml
model:
  output_type: probabilistic
  loss: nll
```
