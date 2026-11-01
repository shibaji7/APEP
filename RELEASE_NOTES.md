# APEP — JGR manuscript release (`paper-jgr-v1.0`)

Code, data products and figures supporting the manuscript on the ionospheric
response to the 14 October 2023 annular and 8 April 2024 total Great American
Eclipses (GAEs).

## Summary

> We present a multipoint investigation of the ionospheric response to the
> 14 October 2023 annular and 8 April 2024 total Great American Eclipses (GAEs)
> using coordinated observations from five Digisondes, the Wallops Island VIPIR,
> the Millstone Hill incoherent scatter radar, and continental-scale GNSS TEC
> measurements.

## Key results

- **Clear E- and F2-region depletions in both events** across peak obscurations
  of 0.30–0.82, observed at local times 10:24–15:22 LT.
- **E region** depletion scales quasi-linearly with obscuration; the **F2 region**
  lags peak obscuration by 20–30 min, consistent with the longer O⁺
  recombination timescale.
- **No G-condition** (NmF1 ≥ NmF2) in either event, even at 0.82 obscuration:
  high obscuration alone is insufficient without adequate F2-peak height
  reduction and favourable local-time/seasonal conditions.
- **White Sands skymaps** show a bifurcated echo population during the 2024
  recovery phase; **Wallops VIPIR** frequency–time–intensity plots show
  cusp-like features confined to 3–6 MHz.
- **Idealized ray tracing** shows both signatures can arise from eclipse-excited
  wave structures at different horizontal scales, i.e. distinct HF manifestations
  of a common wave field.
- **GNSS TEC**: along-track relative change in minima exceeds 40 % (2023 and 2024)
  relative to the expected baseline.

## Figures in this release

All manuscript figures are in `manuscript_figures/jgr/` (PNG, plus PDF where
produced). Generating code is in `apep/jgr/`.

- **Figures 2 and 4 (ionosonde time series) were redone** as 5×4 grids: rows are
  the five stations sorted by peak obscuration; columns are foF2, h′F2/hmF2,
  foE, h′E/hmE. Each panel shows the eclipse-day series (smoothed, gaps ≤ 90 min
  interpolated) against the quiet-day median, with the eclipse interval shaded
  and the peak obscuration 𝒪ᵖ₁₉₃ annotated. Journal-style fonts, thin lines,
  no heavy markers; AGU double-column width.
  - Figure 2 (2023, 13–21 UT) is built entirely from `Datasets/Data/*_scaled.csv`
    and `*_median.csv`; WS833/KR835 use the despiked event files in
    `Datasets/processing/final/` when present.
  - Figure 4 (2024, 16–21 UT) is built from SAO data via `jgr_2024_scaled.py`.
  - Shared code: `apep/jgr/jgr_timeseries_5x4.py`
    (`jgr_2023_scaled.py` is a thin wrapper).

## Known data caveats

- AU930 h′E sits in 100–110 km steps against an ~85 km median (2023); the 2024
  AU930 hmE median CSV is flat at 110 km.
- AU930 foF2 saturates near 9.9 MHz after ~18:30 UT (2023).
- BC840 h′F2 shows a ~340 km excursion near maximum obscuration (2023).
- KR835/WS833 median CSVs lack true hmF2/hmE; 2024 KR835/WS833 median CSVs do not
  exist (baseline falls back to the SAO mean); WS833 2024 has no data.

## Reproducing the time-series figures

```bash
conda run -n apep python -m apep.jgr.jgr_timeseries_5x4 --year 2023   # Figure 2
conda run -n apep python apep/jgr/jgr_2024_scaled.py                  # Figure 4
```
