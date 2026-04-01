# 데이터 제공 형식 분석 및 제안

## 1. 현재 상태

| 항목 | 값 |
|------|-----|
| Train | 86,440개 CSV 파일 (5.6 GB) |
| Validation | 4,022개 CSV 파일 (267 MB) |
| 파일 당 | 384행 × 24열 (datetime + 23 변수), ~65 KB |
| NumPy 환산 크기 | Train 3.1 GB, Val 0.14 GB (float32) |

### 성능 측정 결과

| 작업 | 시간 |
|------|------|
| CSV 1개 파싱 | ~1.6 ms |
| 통계 계산 (86k 파일 순회) | ~2.3분 (첫 실행, 이후 캐시) |
| NumPy 100개 이벤트 로드 | ~1.27 ms (**CSV 대비 127배 빠름**) |
| NumPy memory-map 단일 접근 | ~13 ms |

### 현재 방식의 병목

- **86,440개 파일** = 디렉토리 스캔만으로도 수초, inode 부담
- 매 sample마다 CSV 텍스트 파싱 (헤더 반복, 문자열→float 변환)
- 디스크 5.6 GB이나 실제 수치 데이터는 3.1 GB (오버헤드 ~80%)

---

## 2. 제안: 3가지 방식

### A안: NPY 파일 (추천)

Split별 numpy 배열 1개 + 메타데이터 1개:

```
dataset/
├── train.npy          # shape (86440, 384, 23), float32, ~3.1 GB
├── train_meta.csv     # timestamp (기준 시각 T)
├── validation.npy     # shape (4022, 384, 23), float32, ~142 MB
├── validation_meta.csv
└── variables.json     # ["v_avg", "v_min", ..., "hp30"]
```

| 장점 | 단점 |
|------|------|
| 로드 즉시 사용 | 사람이 직접 읽기 어려움 |
| memory-map 지원 (RAM 초과 가능) | git 관리 불가 |
| I/O 병목 제거 | 데이터 추가 시 재생성 필요 |

**제공 코드 (데이터 제공처용)**:
```python
import numpy as np
import pandas as pd
import glob, json

files = sorted(glob.glob("train/*.csv"))
events = []
timestamps = []
for f in files:
    df = pd.read_csv(f)
    events.append(df.drop(columns=["datetime"]).values)
    timestamps.append(os.path.basename(f).replace(".csv", ""))

np.save("train.npy", np.array(events, dtype=np.float32))
pd.DataFrame({"timestamp": timestamps}).to_csv("train_meta.csv", index=False)

with open("variables.json", "w") as f:
    json.dump(list(df.drop(columns=["datetime"]).columns), f)
```

### B안: NPZ 파일 (A안 + 압축)

```
dataset/
├── train.npz          # ~1.5 GB (압축)
├── validation.npz
└── variables.json
```

| 장점 | 단점 |
|------|------|
| 디스크 절반 | 압축 해제 시간 추가 |
| 메타데이터 포함 가능 | memory-map 불가 |

### C안: 현재 CSV 유지 + 자동 캐시

데이터 형식 변경 없이 파이프라인에서 첫 실행 시 자동 `.npy` 캐시 생성:

```
dataset/
├── train/              # 기존 86,440 CSV (변경 없음)
├── validation/
└── .cache/             # 자동 생성
    ├── train.npy
    └── validation.npy
```

| 장점 | 단점 |
|------|------|
| 데이터 제공 변경 불필요 | 첫 실행 ~3분 |
| CSV 원본 보존 (사람이 읽기 가능) | 디스크 이중 사용 (8.7 GB) |
| 파일 추가/삭제 유연 | 캐시 무효화 관리 필요 |

---

## 3. 비교 요약

| 기준 | A: NPY | B: NPZ | C: CSV+캐시 |
|------|--------|--------|-------------|
| 학습 속도 | **최고** | 좋음 | 최고 (캐시 후) |
| 디스크 | 3.1 GB | **~1.5 GB** | 8.7 GB |
| 첫 실행 | **즉시** | 해제 ~10s | ~3분 |
| 데이터 제공 변경 | 필요 | 필요 | **불필요** |
| 사람이 읽기 | 불가 | 불가 | **가능** |
| 데이터 추가/갱신 | 전체 재생성 | 전체 재생성 | **파일 추가만** |

---

## 4. 결론

**A안(NPY)을 데이터 제공처에 요청 + C안(캐시)을 파이프라인에 구현**

- NPY가 가능하면 → 즉시 사용
- CSV로만 받을 경우 → 캐시가 자동 처리
- 두 경로 모두 지원으로 유연성 확보

### 데이터 제공처 요청 사항

> Split별로 `(N, 384, 23)` shape의 float32 `.npy` 파일,
> 각 이벤트의 기준 시각을 담은 `meta.csv` (timestamp 컬럼),
> 변수 순서를 명시한 `variables.json`을 함께 제공해 주세요.
