# A one-to-three day storm probability

`storm_probability.py`, run 2026-08-06. Train 76,585 anchors (1995-02 to 2021-11), test 23,514
(2022-01 to 2025-12) — the same split as every other measurement here.

```
day 1   P( max ap30 over [T,      T+24 h] >= 48 )
day 2   P( max ap30 over [T+24 h, T+48 h] >= 48 )
day 3   P( max ap30 over [T+48 h, T+72 h] >= 48 )
```

A probability is not a conditional mean of a magnitude, so `sigma_pred/sigma_obs = rho` does not
apply and there is no damping to fight. That is the whole reason to look here after the 12-hour
curve closed.

The bar is not climatology. At this horizon the 27-day recurrence and the current activity level
are what a model has to beat; anything that beats climatology but not those has learned the solar
cycle. Brier skill against climatology, with AUC:

| | day 1 | day 2 | day 3 |
|---|---|---|---|
| test base rate | 0.307 | 0.309 | 0.297 |
| persistence of level | 0.136 / 0.706 | 0.038 / 0.588 | 0.008 / 0.551 |
| 27-day recurrence | 0.026 / 0.606 | 0.026 / 0.605 | 0.023 / 0.603 |
| recurrence + level | 0.146 / 0.719 | 0.052 / 0.633 | 0.034 / 0.613 |
| OMNI wind + index | **0.274 / 0.791** | 0.037 / 0.614 | 0.006 / 0.580 |
| everything | 0.268 / 0.792 | 0.051 / 0.644 | 0.023 / 0.621 |

## Day 1 works. Days 2 and 3 do not, and OMNI is the reason

**Day 1 is a usable product** — skill 0.274, AUC 0.791, and it beats the recurrence baseline
tenfold. It is also not new: a 0–24 h storm probability is the existing 12-hour model's capability
stretched by half a day.

**At day 2 the solar wind adds nothing at all.** "Everything" scores 0.051 against
"recurrence + level" at 0.052 — the wind features are worth zero once the level and the rotation
are in. **At day 3 they are worth less than nothing**: 0.023 against 0.034, so adding the wind
actively hurts.

That is the failure the test was built to detect, and the physics predicts it. The wind that will
be at Earth in two or three days has not arrived and is not observable from Earth; only the
corotating part carries over, and that is the recurrence, which is weak on its own (AUC 0.605).

**So an OMNI-only 1–3 day product fails its gate.** Days 2 and 3 are where the product would be
new, and there the model is doing little more than reciting the solar cycle.

## Which is precisely the argument for a coronagraph

Hand the model an oracle — whether a front arrives inside the target day and how large the jump
is, which is the most a perfect LASCO pipeline could ever deliver, since it cannot measure the
magnetic field — and then degrade the arrival time the way a real pipeline is degraded:

| Brier skill vs climatology | day 1 | day 2 | day 3 |
|---|---|---|---|
| OMNI only | 0.268 | 0.051 | 0.023 |
| perfect arrival | 0.314 | 0.085 | **0.058** |
| arrival ± 3 h | 0.309 | 0.082 | 0.056 |
| **arrival ± 6 h** | 0.305 | **0.078** | **0.052** |
| **arrival ± 10 h** | 0.299 | **0.074** | **0.047** |
| arrival ± 18 h | 0.289 | 0.066 | 0.039 |

**A day-wide window tolerates timing error far better than the 12-hour curve does.** At ±6 h the
day-3 product keeps 83 % of the oracle's gain and the day-2 product 79 %, against 37 % for the
12-hour curve; at ±10 h it is 69 % here against 17 % there. An error of a few hours usually still
lands the front on the right day, and a day is the unit the product is sold in.

At the ±6–10 h that published CME arrival forecasts achieve, the coronagraph channel is worth:

- **day 3: 0.023 → 0.047–0.052 — it more than doubles the skill**
- **day 2: 0.051 → 0.074–0.078 — about half again**

This reverses the conclusion the 12-hour analysis reached about LASCO. There the channel was
bounded at +0.013 to +0.020 of per-lead correlation and not worth a months-long ingest. Here it is
the dominant contributor to a product OMNI cannot build at all.

## Read the absolute numbers before committing to anything

Skill 0.05 is not skill 0.5. For scale, the 12-hour warning threshold work reaches Brier skill
0.482 against climatology. A day-3 storm probability with a *perfect* arrival forecast sits at
0.058 with AUC 0.661 — better than climatology, modestly discriminating, and nowhere near the
quality of the short-horizon product. The honest description of what a coronagraph buys is a weak
forecast where there is currently almost none, not a good one.

Two limits make even that optimistic. The oracle has perfect **detection** as well as perfect
timing, and a real pipeline misses faint events and raises false alarms; every row above is a
ceiling at its stated accuracy. And the front detector is a one-hour speed rise with the field
rising, so it will miss slow ICMEs arriving without a shock.

## Verdict

- **OMNI-only 1–3 day product: gated out.** Day 1 duplicates existing capability; days 2 and 3
  add nothing over recurrence.
- **The same product with a coronagraph: the strongest remaining case in the programme**, because
  it is the only place measured so far where imagery contributes the majority of a product's
  skill rather than a rounding error — but the product itself is weak in absolute terms and must
  be presented that way.
