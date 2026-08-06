# What imagery could be worth, bounded without ingesting an image

Three ridges, run 2026-08-06, all on the 23,514 validation anchors and the same 76,585 training
anchors as the deep runs. Ridge variant "ap + past wind" scores per-lead `rho` 0.569 against the
deep baseline's 0.567, so these numbers transfer.

```
wind_ablation.py     what the wind buys, and the envelope/phase split
recurrence_proxy.py  can the EUV coronal-hole channel be reached for free?
arrival_oracle.py    the ceiling on the coronagraph channel
```

## 0. The logic: bounding an instrument nobody has built

No solar image is used anywhere in this folder. The bounds come from a substitution.

**The premise.** A solar image cannot reach Earth's magnetosphere directly; the causal chain is
Sun → solar wind → magnetosphere → ap30. So whatever an image knows about the next twelve hours of
ap30, it knows *through* the future solar wind.

**The substitution.** Therefore hand the model the **true future wind** — the actual measured
values it is trying to predict. That is strictly more than any image could supply, because a
perfect observation of an outcome dominates any forecast of it. The gain is an upper bound on the
whole channel, and it costs a ridge fit instead of an image archive.

**The split.** Divide the wind into what an instrument could plausibly deliver and what it could
not, and give each separately: speed and field magnitude, which a coronal hole announces; the
field direction, which nothing remote measures.

**The degradation.** Perfect knowledge is not on offer, so corrupt the true values the way a real
pipeline is wrong — a level error held across the window, a timing error on the whole trajectory —
and read off what survives at the accuracy real forecasts achieve.

### Where this can fail

1. **The premise can leak.** Solar EUV and X-ray flux reach Earth directly and change ionospheric
   conductivity, which magnetometers see. ap is constructed to remove the regular daily variation,
   so the leak should be small, but the bound does not cover it.
2. **The reachable/unreachable split is a physical judgement, not a measurement.** This folder got
   it wrong once, filing bt under "imaging cannot measure B" when a coronal hole announces the
   compression at the stream interface as well as the speed. The measurement says what each bucket
   is *worth*; assigning buckets to instruments is reasoning, and reasoning can be wrong.
3. **The degradation model is a guess.** Level and timing error are not the only ways a pipeline is
   wrong — it also misses faint events and raises false alarms, with correlated errors. Every row
   in every sweep is optimistic at its stated accuracy.
4. **The instrument understates the bound.** These are bounds on what a *ridge* extracts from
   perfect information, not on what any model could. Adding physically motivated products to the
   future-wind block — `v·bt`, `v·bt²`, `max(0, −bz)` and `v` times it — moves them:

   | true future information | linear | + products |
   |---|---|---|
   | v + bt, the coronal-hole channel | +0.107 | **+0.116** |
   | everything | +0.228 | **+0.270** |

   A deep non-linear model would extract more still. **So every number here is a floor on the
   bound rather than the bound**, and the imagery buckets should be read as "this scale, possibly
   somewhat more". Note the share moves the other way: the coronal-hole channel is 47 % of the
   linear headroom and 43 % of the enlarged one.

**What the logic answers, and what it does not.** It answers *is this worth building*, from above,
in hours rather than months. It does not answer *how good would it be* — for that you have to
build it. The two decisions it settled here are that the 12-hour curve does not justify an image
ingest, and that for a one-to-three day product imagery is the only route to information the
in-situ record does not contain at any window length.

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

## 2. What each component is worth — and which ones imaging can reach

**Corrected 2026-08-06.** The first version of this section measured every component from its
30-minute *average* column only, and asked what speed alone was worth. Both were wrong. A
30-minute mean cancels the Alfvenic fluctuation that actually drives the index, and a coronal hole
announces more than a speed. Redone over the full avg/min/max set, with the baseline given the
same:

| true future information given | per-lead `rho` | gain | needs the field DIRECTION? |
|---|---|---|---|
| v alone | 0.604 | +0.039 | no |
| **v + bt — a coronal hole's whole message** | **0.672** | **+0.107** | **no** |
| everything except bz | 0.683 | +0.118 | no |
| **bz amplitude only, no sign** | 0.692 | **+0.128** | **no** |
| everything except bz, + bz amplitude | 0.709 | **+0.145** | no |
| bz signed | 0.741 | +0.176 | yes |
| everything | 0.792 | +0.228 | yes |

(baseline, full past and no future information: 0.564)

**About 64 % of the headroom does not require knowing which way the field points.** The sign of Bz
is worth +0.083 of the +0.228 — the largest single piece, but a minority. The claim that imaging
cannot reach the ceiling *because* it cannot measure B was therefore too strong: it holds for the
sign and not for the rest.

And a coronal hole announces both a speed and the compression at the stream interface, so the
honest bound on the EUV channel is **+0.107**, not the +0.037 first reported here.

### What actually limits it: arrival time, not accuracy

Degrading the oracle the way a real pipeline is degraded — a level error held across the window,
because a forecast is wrong about the event rather than independently wrong every half hour, and a
timing error on the whole trajectory:

| level error | timing error | gain | kept |
|---|---|---|---|
| none | none | +0.107 | 100 % |
| 100 km/s, 2 nT | none | +0.097 | **91 %** |
| none | 6 h | +0.058 | 54 % |
| none | 12 h | +0.030 | 28 % |
| **100 km/s, 2 nT** | **12 h** | **+0.021** | **19 %** |
| 100 km/s, 2 nT | 24 h | +0.010 | 9 % |

**Getting the speed right barely matters** — 90 % of the gain survives a 200 km/s error, because
the shape inside the window is preserved. **Getting the arrival time right is everything.**
WSA-class high-speed-stream arrival forecasts sit near ±12–24 h at two to four days of lead, and
there the channel is worth **+0.01 to +0.02**.

That is the same place the independent arrival-oracle analysis landed (+0.013 to +0.020 at
±6–10 h). Two routes, one answer — but the reasoning first given for it was wrong, and the
literature is on the other side of it: coronal-hole area to stream speed is an established
relation and WSA is operational at NOAA for exactly this purpose. What the literature actually
supports is the narrow claim, that the *sign* of the arriving field is not predictable from
remote sensing.

### Can a longer in-situ window substitute for the image?

A coronal hole is a recurrent structure, so if the same hole hit Earth one rotation ago the
in-situ record may already contain what an image would tell us. `input_length_v2.py` asks that
directly, on the full average/minimum/maximum set, with a recurrence block that carries bt as well
as ap30 and speed, and with the lag searched over 26 to 29 days rather than assumed to be 27.

| input window | features | per-lead `rho` |
|---|---|---|
| **1 h** | 44 | **0.568** |
| 6 h | 264 | 0.566 |
| 12 h *(current)* | 528 | 0.564 |
| 1 d | 528 | 0.541 |
| 3 d | 528 | 0.475 |
| 7 d | 616 | 0.394 |
| 27 d | 594 | 0.179 |

| 12 h of recent history plus | per-lead `rho` | gain |
|---|---|---|
| nothing | 0.564 | — |
| a recurrence block at 27 d | 0.564 | −0.000 |
| recurrence blocks at 26, 27, 27.5, 28, 29 d | 0.565 | +0.000 |
| **the true future v and bt** | **0.672** | **+0.107** |

**A longer window reaches none of it.** One hour is the best window and every longer one is worse;
a recurrence block carrying the compression as well as the speed, with the lag searched, adds
nothing at all. Meanwhile the same information known perfectly is worth +0.107.

That is the strongest form of the argument *for* imagery, and it inverts how the earlier
input-length result should be read. It was recorded as another closed axis. Alongside the
corrected bound it means something else: **the information a coronal-hole image would supply is
not sitting in the in-situ record at any window length, so imagery is the only route to it.**

The physics is consistent. The in-situ record describes plasma that has already gone past. A
coronal hole persists across rotations, but its geoeffective expression at Earth depends on how
the hole has grown or closed since, on the current-sheet tilt, and on where Earth sits relative to
the stream — all of which change between rotations. The recurrence carries the *possibility* of a
stream and not its *timing or amplitude*, and timing is what the degradation table above shows to
be everything. An image gives the hole as it is now.

One caveat on the sweep. The ridge penalty is fixed, so longer windows are also carrying more
columns, and part of the decline is over-parameterisation rather than absence of signal. The
recurrence rows are the controlled comparison — same base, one block added — and they are flat.

### A retracted claim, and why it was wrong

The first version of this file closed the EUV channel outright, on the grounds that the 27-day
recurrence is a free and exact record of what an image can only estimate, and that adding it was
worth −0.000. The measurement was right and the inference was not: **the recurrence is a weak
proxy, not a strong proxy returning empty.**

| | per-lead `rho` |
|---|---|
| recurrence alone (ap30 one rotation ago) | 0.109 |
| recurrence alone (speed one rotation ago) | 0.111 |
| both | 0.121 |
| raw correlation, last rotation's window peak vs this one's | **0.139** |

And it adds **+0.000 to a model given no solar wind at all**, which is the test that settles it. A
genuinely redundant predictor still helps a model that lacks what it is redundant with. This one
does not, so it was never carrying the information in the first place.

Coronal holes are among the most persistent solar structures, but 27 days is a hard interval:
holes grow and close between rotations, the synodic period is 27.28 days and differential
rotation moves the lag with latitude, whether the CIR compression reaches Earth depends on a
current-sheet tilt that changes, and roughly half of storms are CME-driven and do not recur at
all. A weak instrument returning null proves nothing. The +0.037 bound above is the honest
version of the same question.

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
