# What imagery could be worth, bounded without ingesting an image

Three ridges, run 2026-08-06, all on the 23,514 validation anchors and the same 76,585 training
anchors as the deep runs. Ridge variant "ap + past wind" scores per-lead `rho` 0.569 against the
deep baseline's 0.567, so these numbers transfer.

```
wind_ablation.py     what the wind buys, and the envelope/phase split
recurrence_proxy.py  can the EUV coronal-hole channel be reached for free?
arrival_oracle.py    the ceiling on the coronagraph channel
```

## 1. The headroom, and how it splits

| given to the ridge | per-lead `rho` | vs the model |
|---|---|---|
| ap history only | 0.512 | −0.057 |
| persistence | 0.518 | — |
| **ap + past wind — the model's information** | **0.569** | — |
| + true future wind except Bz — the **envelope** | 0.673 | **+0.104** |
| + true future Bz only — the **phase** | 0.708 | **+0.140** |
| + all true future wind | 0.785 | +0.216 |

Two things follow. Solar wind is what lifts the forecast above persistence and it is worth
+0.057; without it the model is a persistence forecast. And the headroom is not all Bz — the
phase is the larger half, but the envelope is +0.104, because `bt` and `np` are both poorly
predicted at 12 h (0.565, 0.546) and both set the coupling amplitude. The earlier shorthand
"the predictable part is not geoeffective" is true of `v` and not of the wind as a whole.

Imaging measures density and emission, not magnetic field, so imagery can only aim at the
envelope. Everything below is about that +0.104.

## 2. The EUV coronal-hole channel is closed

A solar EUV image tells Earth one thing about the coming days: a coronal hole is facing us and a
stream will follow. The same feature already announced itself one rotation ago, and that record
is free and exact where the image is an estimate.

| added to the model's information | per-lead `rho` | gain |
|---|---|---|
| ap30 across the window 27 d earlier | 0.568 | −0.000 |
| speed across the window 27 d earlier | 0.568 | −0.001 |
| trailing 3-day and 27-day level | 0.572 | +0.003 |
| all three | 0.570 | +0.002 |
| *(true future envelope, the target)* | *0.673* | *+0.104* |

The recurrence reaches none of the envelope. Not because recurrence carries no information about
ap, but because it carries none the in-situ wind does not already have. **193 Å and 211 Å cannot
beat this, so the EUV coronal-hole channel is closed for the 12-hour product.** This is a
one-sided test and it says nothing about eruptions, which do not recur.

## 3. The coronagraph channel, bounded

A coronagraph cannot measure the field either, so the most a perfect LASCO pipeline delivers is
*a disturbance is coming and here is when*. Give the model exactly that — the true arrival step
and the true amplitude — and nothing built from images can beat it.

Fronts detected as a 1-hour speed rise ≥ 55.5 km/s (99.5th percentile) with the field rising:
1,431 events, about 46 per year. 5.3 % of validation windows contain one; 14.7 % of storm windows.

| subset | n | model | oracle | gain | full ceiling |
|---|---|---|---|---|---|
| all anchors | 23,514 | 0.569 | 0.606 | +0.038 | +0.216 |
| windows containing a front | 1,256 | 0.405 | 0.552 | +0.147 | +0.395 |
| **storm windows with a front** | **710** | **0.308** | **0.482** | **+0.174** | +0.474 |
| storm windows without a front | 4,119 | 0.382 | 0.381 | −0.001 | +0.334 |

The value is concentrated exactly where the model is worst and where a warning matters: the
onset windows. It is zero everywhere else, which is why it dilutes to +0.038 overall.

## 4. But arrival accuracy is the whole story

Re-run with the front slid by a random offset and the model retrained on the degraded version,
which is what a forecaster actually gets.

| arrival-time error | all anchors | storm windows with a front |
|---|---|---|
| perfect | +0.038 | +0.174 |
| ± 1 h | +0.034 | +0.150 |
| ± 2 h | +0.032 | +0.133 |
| ± 3 h | +0.029 | +0.120 |
| **± 6 h** | **+0.020** | **+0.064** |
| **± 10 h** | **+0.013** | **+0.030** |

Published CME arrival forecasts sit at ±6–10 h. At that accuracy the channel is worth **+0.013
to +0.020 of per-lead `rho` overall**, and +0.03 to +0.06 on onset windows. For comparison, the
clock change — the one intervention in the whole investigation that moved discrimination —
delivered +0.032, and cost a re-index rather than an image archive.

The curve collapses between ±3 h and ±6 h. **The entire case for a coronagraph pipeline rests on
whether arrival time can be forecast to better than about three hours**, which is beyond the
current state of the art for CME transit. If that ever changes, this becomes worth +0.12 on
onset windows and the calculation reverses.

## Limits of these bounds

- The jitter model degrades **timing only**. A real pipeline also misses events and raises false
  alarms, so every row above is optimistic at its stated accuracy.
- The front detector is crude: it will miss slow ICMEs that arrive without a shock and will catch
  some steepening CIRs.
- Ridges, on bow-shock-time data. The ridge tracks the deep model at the baseline, but this
  project's record on ridge-led predictions is one transfer in three — treat the numbers as
  bounds and orders of magnitude, not forecasts of a trained model's score.
