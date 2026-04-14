# TODO / 할 일

---

## 1. PatchTST patch_len/stride Optimization / PatchTST patch_len/stride 최적화

**Status / 상태**: Not started / 미착수

### Background / 배경

The current default `patch_len=16, patch_stride=8` is adopted directly from the PatchTST paper (Nie et al., ICLR 2023) without project-specific tuning. The paper itself selected these values through ablation studies, not a systematic formula.

현재 기본값 `patch_len=16, patch_stride=8`은 PatchTST 논문에서 프로젝트별 튜닝 없이 그대로 채택한 것입니다. 논문 자체도 체계적 공식이 아닌 ablation study를 통해 선택했습니다.

### What to Do / 수행 내용

Run ablation experiments to find optimal `patch_len` and `patch_stride` for each input length in this project's solar wind dataset.

이 프로젝트의 태양풍 데이터셋에서 각 입력 길이에 대한 최적 `patch_len`과 `patch_stride`를 찾기 위한 ablation 실험을 수행합니다.

### Recommended Search Ranges / 권장 탐색 범위

General guidelines from the paper and best practices:

논문 및 모범 사례에서 도출한 일반적 가이드라인:

- `patch_len` should produce **at least 3-5 patches** for meaningful attention
- `patch_len`은 의미 있는 attention을 위해 **최소 3-5개 패치**를 생성해야 함
- `patch_stride` is typically `patch_len / 2` (50% overlap)
- `patch_stride`는 보통 `patch_len / 2` (50% overlap)
- `patch_len` covering ~10-25% of input length is common
- `patch_len`이 입력 길이의 ~10-25%를 커버하는 것이 일반적

| Input | seq_len | Suggested patch_len range | Patches (stride=patch/2) |
|-------|---------|--------------------------|--------------------------|
| 6h | 12 | 3, 4, 6 | 4-7 |
| 12h | 24 | 4, 6, 8, 12 | 4-9 |
| 1d | 48 | 8, 12, 16 | 5-11 |
| 2d | 96 | 12, 16, 24 | 7-15 |
| 3d | 144 | 16, 24, 32 | 8-15 |

### Experiment Plan / 실험 계획

```bash
# Example: ablation for 2d input with PatchTST
for patch_len in 12 16 24; do
    stride=$((patch_len / 2))
    python scripts/train.py --config-name=local +io=in2d_out12h +model=patchtst \
        model.patch_len=$patch_len model.patch_stride=$stride \
        experiment.name="ablation_patch${patch_len}_stride${stride}"
done
```

### Affected Models / 영향받는 모델

Only these 2 models use `patch_len`/`patch_stride`:
- `patchtst`
- `gnn_patchtst`

All other 7 models are unaffected by these parameters.

이 2개 모델만 `patch_len`/`patch_stride`를 사용합니다. 나머지 7개 모델은 이 파라미터에 영향을 받지 않습니다.
