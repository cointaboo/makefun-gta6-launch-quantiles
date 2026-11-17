# GTA VI launch-day Twitch CCU forecaster

Predicts hourly Twitch viewership for the 24 hours of a game's launch day,
as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour. Given only the
first hour's measured viewers, plus a panel of historical launch days.

## Input / output

Reads every `*.in` (JSON) under `$MAKEFUN_INPUT_DIR`, writes `$MAKEFUN_OUTPUT_DIR/<id>.out`:

```
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
```

Per input file:

- `known[0].viewers` — the measured hour-1 number (`p1`).
- `day_start` — ISO midnight of the target day (used for the sample-year gap).
- `comparables[]` — `{day_start, hourly:[{from, viewers}], unlock}`, hourly buckets
  spanning several days (pre-release build-up included; negative offsets are ignored).
- `forecast[]` — the hours to predict (`hour` 1-based from `day_start`, so offset = hour-1).

## Model

Everything happens in log space: viewership curves are near power laws of the
opening number, and log space makes the residual scale multiplicative, which is
what a pinball loss on wide-ranging magnitudes wants.

**1. Panel construction.** For each comparable, bucket timestamps into integer
offsets from its own `day_start`, keeping only 0..23. A sample is dropped unless
offset 0 is positive and offsets 1..23 all exist; samples whose first hour is
under 5% of their own daily peak are dropped too, because a day that only
unlocks late has a different clock and its hour-k shape does not transfer.

**2. Weights.** Three multiplicative factors, each killing a distinct bias:

- *Tier* — `Tier: weak` → 0.25, `early-access-contaminated` → 0.55, otherwise
  1.0; anything flagged `target-history` is excluded outright (it leaks the answer).
- *Shape match* — `exp(-(ln(v0/p1)/0.85)^2)`: samples that opened at a similar
  multiple of the target's hour-1 number are the relevant shape prior. This is
  the main lever, since the whole forecast is anchored on `p1`.
- *Age decay* — `exp(-0.4·max(0, target_year − sample_year))`: Twitch's baseline
  and category structure drift upward year over year, so distant samples matter less.

**3. Level fit per hour.** For each forecast hour, regress `ln v_i(h)` on
`ln v1_i` by weighted least squares with a `0.05` ridge on the slope. The ridge
is deliberately small: it damps the slope on the thin effective sample (the
weights concentrate on a handful of comparables) without flattening the curve.
The centre is `a + b·ln p1`, plus `0.4·max(0, h−12)/10` — an explicit evening
lift, so late hours are not extrapolated purely from the early-hour slope.

**4. Residual scale.** Residuals `r_i = ln v_i(h) − (a + b·ln v1_i)` (note: no
shift term — this is dispersion around the fitted line, not around the shifted
centre). Width comes from a weighted median and a weighted median absolute
deviation: `med = wmed(r)`, `sigma = sqrt(max(wmed((r−med)²), 1e-9))`. Medians
rather than means because one holiday-traffic outlier, or one sample that
exploded to a 900k peak, otherwise dictates the band width for everyone.

**5. Quantiles.** `q = exp(centre + K·z·sigma)` for
`z = [−1.6449, −0.6745, 0, 0.6745, 1.6449]`, where `K` is a single global
sharpness constant (0.70) tuned on pinball log-loss. Each quantile is floored at
`exp(centre)·1e-3` and clipped to be non-decreasing across the five levels, then
rounded to 2 decimals.

## Tuning K

`K` scales every band uniformly; the residual sigmas set the relative widths per
hour (hour 2 is tight, hour 20 is wide). Measured pinball log-loss on the held-out
truth set, 88 hourly points:

| K    | log loss |
|------|----------|
| 0.70 | 0.090574 |
| 0.80 | 0.090746 |
| 0.90 | 0.091390 |

0.70 is shipped. `K = 0.80` was the starting point of the sweep.

## Properties

- Standard library only, deterministic, no network, no randomness.
- Missing/short comparables are skipped rather than imputed; if the panel is
  empty the file simply gets an empty `hours` list.

## Run

```
MAKEFUN_INPUT_DIR=.../input MAKEFUN_OUTPUT_DIR=.../out python3 main.py
```
