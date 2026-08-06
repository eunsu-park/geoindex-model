# Where the ceiling is, mapped three ways

Run 2026-08-06 on the 23,514 validation anchors and 76,585 training anchors of the deep runs.
The ridge is only usable where it tracks the trained model; it does, at every lead (0.891 vs
0.851 at 0.5 h through 0.387 vs 0.384 at 12 h), so its headroom transfers.

```
horizon_scan.py    headroom by lead
target_compare.py  ap30 against hp30
bt_scan.py         can the field magnitude be predicted better?
```

## By lead — the short horizon is saturated

What perfect knowledge of the future wind would add, at each lead:

| lead | deep model | persistence | + envelope | + bz | **+ all** |
|---|---|---|---|---|---|
| 0.5 h | 0.851 | 0.870 | +0.002 | +0.000 | **+0.002** |
| 1 h | 0.815 | 0.763 | +0.006 | +0.008 | **+0.014** |
| 2 h | 0.729 | 0.661 | +0.023 | +0.055 | +0.074 |
| 3 h | 0.661 | 0.596 | +0.044 | +0.094 | +0.130 |
| 6 h | 0.539 | 0.462 | +0.100 | +0.152 | +0.229 |
| 12 h | 0.384 | 0.288 | +0.219 | +0.226 | +0.374 |

**A 0–3 h specialist has no information to extract.** At half an hour the entire prize is +0.002
and at one hour +0.014, because the input already contains the answer. The model does lose to
persistence at 0.5 h (0.851 against 0.870), but that is one lead of twenty-four and worth about
+0.02 — a fitting question against persistence, not a modelling one.

The phase/envelope ratio moves with lead: Bz is more than twice the envelope at 2–3 h and they
are level by 12 h. Short lead is more Bz-dominated, not less.

### Two designs that produced confident nonsense

Both are guarded in the script's docstring because either would have been reported as a result.

1. **Refitting the ridge at a shorter output window.** On the raw scale the fit is dominated by
   the heavy tail and loses to persistence by MAE skill −1.2 at 1 h, where the deep model scores
   −0.033. Refitting on log1p and inverting with `expm1` is worse: adding *true* future Bz then
   makes correlation fall by 0.094, which cannot happen in a correct fit and betrays the bias of
   `exp(E[log(1+y)])`. The fix was not to refit — a multi-output ridge fits each output column
   independently, so column k of the 12-hour fit already *is* the fit for lead k.
2. **Persistence taken as `ap[T]`.** The window convention is input `[T−24, T)` and target
   `[T, T+24)`, so `ap[T]` is the first *target*. Using it as persistence hands the baseline the
   answer and returns correlation 1.000 at the first lead.

## By target — the ceiling belongs to the wind, not to ap30

Same anchors, same features, only the target changes.

| | ap30 | hp30 |
|---|---|---|
| model, per-lead `rho` | 0.569 | **0.640** |
| persistence | 0.489 | 0.563 |
| headroom | +0.216 | +0.195 |
| phase (true future bz) | +0.140 — 65 % | +0.120 — **62 %** |
| envelope | +0.104 — 48 % | +0.094 — **48 %** |
| **v** — what imagery can reach | +0.037 | **+0.053** |
| **bt** — what it cannot | +0.077 | +0.052 |

The split is the same to within a few percent, so every bound in this investigation is a property
of the solar wind and transfers to hp30 rather than needing to be redone.

One difference matters. For hp30 speed is worth **+0.053** against ap30's +0.037, and bt is worth
less. hp30 is quasi-logarithmic and responds to sustained moderate activity, which is the
high-speed-stream regime, where ap30 is dominated by large CME storms. So **the imagery ceiling is
about 40 % higher for hp30 than for ap30** — still small, but hp30 is the next target index.

## By input length — bt is as predictable as it gets

bt is the largest single component of the envelope (+0.077) and only ~0.6 predictable at 12 h, so
it was the one crack worth probing. ICME sheaths and CIR compressions are multi-day structures, so
a longer window had a physical reason to help. Correlation of predicted with observed bt:

| input window | 0.5 h | 2 h | 6 h | **12 h** | 24 h |
|---|---|---|---|---|---|
| 1 h | 0.959 | 0.909 | 0.784 | **0.598** | 0.316 |
| 12 h *(current)* | 0.959 | 0.908 | 0.782 | **0.596** | 0.307 |
| 3 d | 0.877 | 0.827 | 0.701 | 0.528 | 0.297 |
| 27 d | 0.391 | 0.375 | 0.344 | 0.335 | 0.312 |
| persistence | 0.956 | 0.899 | 0.755 | 0.545 | 0.226 |

**One hour of history is everything, and more makes it worse.** The extra columns add variance
without signal. The envelope is closed from the input-length direction as well, and the result
mirrors what was already measured for the ap target itself, where one hour reached 99 % of the
best window.

## What these three close

- **B2, a 0–3 h specialist** — no information prize (+0.002 to +0.014).
- **B3, longer input for bt** — no crack; one hour is the whole signal.
- **E1, per-index rework** — unnecessary; hp30 splits the same way.

What they leave open is unchanged: the generative wind-to-index ensemble, which the tower-property
theorem does not close; a storm-probability product at one to three days of lead; and imagery
aimed at the 24–72 hour horizon, where no bound has been measured.
