"""Full analysis pipeline — all six vipir/analysis modules chained together.

Demonstrates the complete post-trace-identification workflow using a real
VIPIR sounding (WI937_2022233235902.RIQ):

    PolarizationClassifier  →  label every echo O / X / ambiguous
    SpreadFAnalyzer         →  detect and classify spread-F
    TrueHeightInversion     →  convert virtual to true height (O-mode)
    IonogramScaler          →  derive foF2, foE, MUF(3000), M(3000)F2
    IrregularityAnalyzer    →  EP structure function and spectral index α
    NeXtYZInverter (Lite)   →  3-D WSI electron density profile + tilts

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Run each analysis module in order.
3. Print one-line summaries for each result.
4. Produce a 3×2 diagnostic figure.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_full_pipeline.png``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from _load_data import load_echoes, fit_drift_from_df
import glob
import datetime as dt

from pynasonde.vipir.analysis import (
    IonogramScaler,
    PolarizationClassifier,
    TrueHeightInversion,
)
from pynasonde.digisonde.digi_utils import setsize
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

fname_glob = "/tmp/apep2/WI937_2024099*.RIQ"
font_size = 12
setsize(font_size)

# Fixed height grid for drift velocity reindexing (bin centres, km)
_DRIFT_H = np.arange(10.0, 1000.0, 20.0)
# Output paths
_NC_OUT = "figures/2024/vipir/apep2_processed.nc"

# ── EchoExtractor parameters ───────────────────────────────────────────────
_EXTRACTOR_SNR_DB       = 3.0
_EXTRACTOR_MIN_H_KM     = 60.0
_EXTRACTOR_MAX_H_KM     = 1000.0
_EXTRACTOR_MIN_RX_DIR   = 3
_EXTRACTOR_MAX_ECHO_PS  = 5

# ── IonogramFilter parameters ──────────────────────────────────────────────
_FILTER_EP_MAX_DEG          = 90.0
_FILTER_DBSCAN_ENABLED      = True
_FILTER_DBSCAN_EPS          = 1.0
_FILTER_DBSCAN_MIN_SAMPLES  = 5
_FILTER_RANSAC_ENABLED      = True
_FILTER_RANSAC_RESIDUAL_KM  = 100.0
_FILTER_RANSAC_MIN_SAMPLES  = 10
_FILTER_RANSAC_N_ITER       = 200
_FILTER_RANSAC_POLY_DEG     = 3
_FILTER_RANSAC_MIN_INLIER   = 0.3
_FILTER_MULTIHOP_ENABLED    = True
_FILTER_MULTIHOP_ORDERS     = (2, 3)
_FILTER_MULTIHOP_H_TOL_KM   = 50.0
_FILTER_MULTIHOP_SNR_MARGIN = 6.0

# ── Drift height-bin parameters ────────────────────────────────────────────
_DRIFT_BIN_KM    = 20.0
_DRIFT_MIN_ECHO  = 6
_DRIFT_SNR_WEIGHT = True
_DRIFT_N_SIGMA   = 2.5

files = sorted(glob.glob(fname_glob))
print(f"Found {len(files)} files matching glob: {fname_glob}")
timestamps = [dt.datetime.strptime(f.split("_")[-1].split(".")[0], "%Y%j%H%M%S") for f in files]
records = []   # accumulate one dict per processed file

for f, t in zip(files, timestamps):
    print(f"  {f}  → timestamp: {t}")

    # ---------------------------------------------------------------------------
    # Step 1: Load RIQ file — use configs[1] for the older vipir_version=0 format
    # ---------------------------------------------------------------------------

    df, station = load_echoes(
        f,
        snr_threshold_db=_EXTRACTOR_SNR_DB,
        ep_max_deg=_FILTER_EP_MAX_DEG,
        dbscan_enabled=_FILTER_DBSCAN_ENABLED,
        ransac_enabled=_FILTER_RANSAC_ENABLED,
    )
    print(f"[{station}]  Filtered echoes: {len(df)}")


    # ---------------------------------------------------------------------------
    # Step 2: Plot diagnostics — 2×3 grid
    # ---------------------------------------------------------------------------
    #
    #  (A) Ionogram          (B) XL vs height       (C) YL vs height
    #  (D) XL–YL map         (E) Doppler velocity   (F) Polarization PP
    #

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 9),
        constrained_layout=True,
    )
    fig.suptitle(
        f"VIPIR Echo Extraction — {t}  (vipir_version=0)",
        fontsize=font_size + 2,
    )

    freq_mhz = df["frequency_khz"] / 1e3 if not df.empty else None
    amp_vmin = df["amplitude_db"].quantile(0.05) if not df.empty else 0
    amp_vmax = df["amplitude_db"].quantile(0.95) if not df.empty else 1

    # ── (A) Ionogram: frequency vs virtual height, colour = amplitude ────────────
    ax = axes[0, 0]
    if not df.empty:
        sc = ax.scatter(
            freq_mhz,
            df["height_km"],
            c=df["amplitude_db"],
            cmap="plasma",
            s=4,
            vmin=amp_vmin,
            vmax=amp_vmax,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No echoes detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
    ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
    ax.set_title("(A) Ionogram", fontsize=font_size)
    ax.set_ylim(50, 1000)
    # ax.set_xscale("log")   # WI937 uses a log frequency sweep — log x-axis matches data spacing.

    # ── (B) XL vs virtual height, colour = frequency ────────────────────────────
    ax = axes[0, 1]
    xl_mask = df["xl_km"].notna() if not df.empty else []
    if not df.empty and xl_mask.any():
        sc = ax.scatter(
            df.loc[xl_mask, "xl_km"],
            df.loc[xl_mask, "height_km"],
            c=freq_mhz[xl_mask],
            cmap="viridis",
            s=4,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "XL all NaN\n(insufficient receivers)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("XL — Eastward (km)", fontsize=font_size)
    ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
    ax.set_title("(B) XL vs Height", fontsize=font_size)
    ax.set_ylim(50, 1000)

    # ── (C) YL vs virtual height, colour = frequency ────────────────────────────
    ax = axes[0, 2]
    yl_mask = df["yl_km"].notna() if not df.empty else []
    if not df.empty and yl_mask.any():
        sc = ax.scatter(
            df.loc[yl_mask, "yl_km"],
            df.loc[yl_mask, "height_km"],
            c=freq_mhz[yl_mask],
            cmap="viridis",
            s=4,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "YL all NaN\n(insufficient receivers)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("YL — Northward (km)", fontsize=font_size)
    ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
    ax.set_title("(C) YL vs Height", fontsize=font_size)
    ax.set_ylim(50, 1000)

    # ── (D) XL–YL echolocation map, colour = amplitude ──────────────────────────
    ax = axes[1, 0]
    dir_mask = (df["xl_km"].notna() & df["yl_km"].notna()) if not df.empty else []
    if not df.empty and dir_mask.any():
        sc = ax.scatter(
            df.loc[dir_mask, "xl_km"],
            df.loc[dir_mask, "yl_km"],
            c=df.loc[dir_mask, "amplitude_db"],
            cmap="plasma",
            s=6,
            alpha=0.7,
            vmin=amp_vmin,
            vmax=amp_vmax,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No direction data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("XL — Eastward (km)", fontsize=font_size)
    ax.set_ylabel("YL — Northward (km)", fontsize=font_size)
    ax.set_title("(D) Echolocation Map (XL, YL)", fontsize=font_size)

    # ── (E) Doppler velocity vs virtual height, colour = frequency ──────────────
    ax = axes[1, 1]
    v_mask = df["velocity_mps"].notna() if not df.empty else []
    if not df.empty and v_mask.any():
        sc = ax.scatter(
            df.loc[v_mask, "velocity_mps"],
            df.loc[v_mask, "height_km"],
            c=freq_mhz[v_mask],
            cmap="coolwarm",
            s=4,
            alpha=0.6,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No velocity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("V* — Phase-path velocity (m/s)", fontsize=font_size)
    ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
    ax.set_title("(E) Doppler Velocity", fontsize=font_size)
    ax.set_ylim(50, 1000)

    # ── (F) Polarization PP vs virtual height, colour = frequency ───────────────
    ax = axes[1, 2]
    pp_mask = df["polarization_deg"].notna() if not df.empty else []
    if not df.empty and pp_mask.any():
        sc = ax.scatter(
            df.loc[pp_mask, "polarization_deg"],
            df.loc[pp_mask, "height_km"],
            c=freq_mhz[pp_mask],
            cmap="RdBu",
            s=4,
            alpha=0.7,
            vmin=-90,
            vmax=90,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "PP all NaN\n(no orthogonal antenna pairs)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=font_size - 1,
        )
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("PP — Polarization (°)", fontsize=font_size)
    ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
    ax.set_title("(F) Polarization PP", fontsize=font_size)
    ax.set_ylim(50, 1000)

    # ---------------------------------------------------------------------------
    # Step 3: Save figure
    # ---------------------------------------------------------------------------

    out = f"figures/2024/vipir/echo_extraction_wi937_{t.strftime('%H%M%S')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out}")

    # ── 5. PolarizationClassifier ─────────────────────────────────────────────────
    pol_clf = PolarizationClassifier(o_mode_sign=-1, pp_ambiguous_threshold_deg=30.0)
    pol_res = pol_clf.fit(df)
    ann = pol_res.annotated_df
    print(pol_res.summary())

    # ── 6. TrueHeightInversion ────────────────────────────────────────────────────
    edp = TrueHeightInversion(monotone_enforce=True).fit_from_df(ann[ann["mode"] == "O"])
    print(edp.summary())

    # ── 7. IonogramScaler ─────────────────────────────────────────────────────────
    params = IonogramScaler(min_echoes_for_layer=4, n_bootstrap=200).fit(ann)
    print(params.summary())

    df_vel_filt = fit_drift_from_df(
        df,
        height_bin_km=_DRIFT_BIN_KM,
        min_echoes=_DRIFT_MIN_ECHO,
        snr_weight=_DRIFT_SNR_WEIGHT,
        n_sigma=_DRIFT_N_SIGMA,
    )
    print("\n=== Drift velocity (filtered echoes) ===")
    print(
        df_vel_filt[
            ["height_bin_km", "vx_mps", "vy_mps", "vz_mps", "residual_mps", "n_echoes"]
        ].to_string(index=False)
    )

    # ── 8. Diagnostic figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
    fig.suptitle(
        f"vipir/analysis — Full pipeline diagnostic\n({station}) / {t}",
        fontsize=13,
    )

    freq_mhz = ann["frequency_khz"] / 1e3
    mode_colors = {"O": "steelblue", "X": "firebrick", "ambiguous": "grey"}

    # ── Panel A: Ionogram coloured by mode ────────────────────────────────────────
    ax = axes[0, 0]
    for mode, grp in ann.groupby("mode"):
        ax.scatter(
            freq_mhz[grp.index],
            grp["height_km"],
            c=mode_colors.get(mode, "k"),
            s=4,
            alpha=0.4,
            label=mode,
        )
    ax.set(
        xlabel="Frequency (MHz)",
        ylabel="Height (km)",
        title="(A) Ionogram — O/X/ambiguous",
        ylim=(60, 800),
    )
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Panel B: PP vs height ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    for mode, grp in ann.groupby("mode"):
        ax.scatter(
            grp["polarization_deg"],
            grp["height_km"],
            c=mode_colors.get(mode, "k"),
            s=4,
            alpha=0.4,
            label=mode,
        )
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set(
        xlabel="PP (degrees)",
        ylabel="Height (km)",
        title="(B) PP vs height",
        xlim=(-180, 180),
        ylim=(60, 800),
    )
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Panel C: True-height EDP ──────────────────────────────────────────────────
    ax = axes[1, 0]
    if edp.n_layers > 0:
        ax.plot(
            edp.plasma_freq_mhz,
            edp.true_height_km,
            "o-",
            color="steelblue",
            ms=5,
            label="fp(h) true height",
        )
        ax.plot(
            edp.plasma_freq_mhz,
            edp.virtual_height_km,
            "s--",
            color="grey",
            ms=4,
            alpha=0.5,
            label="h*(f) virtual",
        )
        if not np.isnan(edp.foF2_mhz):
            ax.axvline(
                edp.foF2_mhz,
                color="firebrick",
                lw=1,
                ls=":",
                label=f"foF2={edp.foF2_mhz:.2f} MHz",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient O-mode echoes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlabel="Plasma frequency (MHz)",
        ylabel="Height (km)",
        title="(C) TrueHeightInversion — EDP",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel D: Scaled parameters bar ────────────────────────────────────────────
    ax = axes[1, 1]
    params.plot(ax=ax)
    ax.set_title("(D) IonogramScaler — foE, foF2, MUF(3000)")

    ax = axes[2, 0]
    if not df_vel_filt.empty:
        vf = df_vel_filt.dropna(subset=["vx_mps"])
        h = vf["height_bin_km"].values
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.plot(vf["vx_mps"], h, "o-", color="tab:blue", ms=5, label="Vx East")
        ax.plot(vf["vy_mps"], h, "s-", color="tab:orange", ms=5, label="Vy North")
        ax.plot(vf["vz_mps"], h, "^-", color="tab:green", ms=5, label="Vz Up")
    ax.set(
        xlabel="Velocity (m/s)",
        ylabel="Height (km)",
        title="(H) 3-D drift — filtered",
        ylim=(50, 1000),
    )
    ax.legend(fontsize=font_size - 3)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax2 = ax.twiny()
    if not df_vel_filt.empty:
        vf = df_vel_filt.dropna(subset=["residual_mps"])
        ax.plot(
            vf["residual_mps"],
            vf["height_bin_km"],
            "o-",
            color="tab:purple",
            ms=4,
            label="RMS filtered",
        )
        ax2.plot(
            vf["n_echoes"],
            vf["height_bin_km"],
            "D-",
            color="tab:red",
            ms=4,
            label="N filtered",
        )
    ax.set_xlabel("RMS LOS residual (m/s)", fontsize=font_size - 1, color="tab:purple")
    ax2.set_xlabel("Echoes per bin", fontsize=font_size - 1, color="tab:red")
    ax.set_ylabel("Height (km)", fontsize=font_size - 1)
    ax.set_title("(I) Fit quality: raw vs filtered", fontsize=font_size)
    ax.set_ylim(50, 1000)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=font_size - 3, loc="lower right")
    ax.grid(True, alpha=0.3)

    out = f"figures/2024/vipir/analysis_full_pipeline_wi937_{t.strftime('%H%M%S')}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {out}")


    # ---------------------------------------------------------------------------
    # Step 4: Accumulate results for consolidated xarray output
    # ---------------------------------------------------------------------------

    # Count O/X from polarization-annotated echoes
    mode_counts = ann["mode"].value_counts() if not ann.empty and "mode" in ann.columns else {}

    records.append(dict(
        time=t,
        # echo DataFrame (annotated with O/X/ambiguous)
        echo_df=ann.copy() if not ann.empty else ann,
        # EDP profile arrays
        edp_true_height_km=edp.true_height_km.copy() if edp.n_layers > 0 else np.array([]),
        edp_plasma_freq_mhz=edp.plasma_freq_mhz.copy() if edp.n_layers > 0 else np.array([]),
        edp_virtual_height_km=edp.virtual_height_km.copy() if edp.n_layers > 0 else np.array([]),
        # EDP scalars
        foF2_mhz=edp.foF2_mhz,
        hmF2_km=edp.hmF2_km,
        n_edp_layers=edp.n_layers,
        # IonogramScaler scalars
        sc_foE_mhz=params.foE_mhz,
        sc_foF2_mhz=params.foF2_mhz,
        sc_h_prime_E_km=params.h_prime_E_km,
        sc_h_prime_F2_km=params.h_prime_F2_km,
        sc_MUF3000_mhz=params.MUF3000_mhz,
        sc_M3000F2=params.M3000F2,
        sc_foF2_sigma_mhz=params.foF2_sigma_mhz,
        # polarisation counts
        n_O=int(mode_counts.get("O", 0)),
        n_X=int(mode_counts.get("X", 0)),
        n_ambiguous=int(mode_counts.get("ambiguous", 0)),
        # drift velocity profile
        drift_df=df_vel_filt.copy() if not df_vel_filt.empty else df_vel_filt,
    ))
    break   # demo with just the first file for now

# =============================================================================
# Build consolidated xarray.Dataset from all processed files
# =============================================================================

if not records:
    print("No records to consolidate.")
else:
    n_t = len(records)

    # ── helper: pad 1-D array to length n with NaN ────────────────────────────
    def _pad(arr, n):
        out = np.full(n, np.nan, dtype=float)
        m = min(len(arr), n)
        if m > 0:
            out[:m] = arr[:m]
        return out

    # ── EDP profiles (time, edp_layer) — padded to max layers ─────────────────
    max_edp = max(max(r["n_edp_layers"] for r in records), 1)
    edp_true_h = np.vstack([_pad(r["edp_true_height_km"],    max_edp) for r in records])
    edp_fp     = np.vstack([_pad(r["edp_plasma_freq_mhz"],   max_edp) for r in records])
    edp_virt_h = np.vstack([_pad(r["edp_virtual_height_km"], max_edp) for r in records])

    # ── Drift profiles (time, drift_h) — reindexed to fixed _DRIFT_H grid ─────
    n_dh = len(_DRIFT_H)
    vx_mat    = np.full((n_t, n_dh), np.nan)
    vy_mat    = np.full((n_t, n_dh), np.nan)
    vz_mat    = np.full((n_t, n_dh), np.nan)
    vres_mat  = np.full((n_t, n_dh), np.nan)
    vcond_mat = np.full((n_t, n_dh), np.nan)
    vn_mat    = np.full((n_t, n_dh), np.nan)

    for i, r in enumerate(records):
        df_d = r["drift_df"]
        if df_d.empty:
            continue
        for _, row in df_d.iterrows():
            h = row["height_bin_km"]
            idx = int(np.argmin(np.abs(_DRIFT_H - h)))
            if np.abs(_DRIFT_H[idx] - h) < 15.0:   # within half a 20-km bin
                vx_mat[i, idx]    = row.get("vx_mps", np.nan)
                vy_mat[i, idx]    = row.get("vy_mps", np.nan)
                vz_mat[i, idx]    = row.get("vz_mps", np.nan)
                vres_mat[i, idx]  = row.get("residual_mps", np.nan)
                vcond_mat[i, idx] = row.get("condition_number", np.nan)
                vn_mat[i, idx]    = row.get("n_echoes", np.nan)

    # ── Echo data — ragged per timestamp, saved separately as Parquet ─────────
    # Echo counts vary per sounding (50–3000+), so padding into xarray would
    # bloat the array by the worst-case file.  Instead concatenate all echoes
    # into a flat DataFrame with a 'time' column and write Parquet.
    _PARQUET_OUT = _NC_OUT.replace(".nc", "_echoes.parquet")
    echo_frames = []
    for r in records:
        df_e = r["echo_df"].copy()
        if not df_e.empty:
            df_e.insert(0, "time", r["time"])
            echo_frames.append(df_e)

    if echo_frames:
        import pandas as pd
        df_echoes = pd.concat(echo_frames, ignore_index=True)
        os.makedirs(os.path.dirname(_PARQUET_OUT), exist_ok=True)
        df_echoes.to_parquet(_PARQUET_OUT, index=False)
        print(f"Echo DataFrame saved → {_PARQUET_OUT}  "
              f"({len(df_echoes):,} rows × {df_echoes.shape[1]} cols)")

    # ── CF metadata ────────────────────────────────────────────────────────────
    _SCALAR_META = {
        "foF2_mhz":          ("MHz",  "Critical frequency F2 layer (EDP)"),
        "hmF2_km":           ("km",   "Peak height F2 layer (EDP)"),
        "n_edp_layers":      ("1",    "Number of EDP lamination layers"),
        "sc_foE_mhz":        ("MHz",  "Critical frequency E layer (scaler)"),
        "sc_foF2_mhz":       ("MHz",  "Critical frequency F2 layer (scaler)"),
        "sc_h_prime_E_km":   ("km",   "Virtual height E layer (scaler)"),
        "sc_h_prime_F2_km":  ("km",   "Virtual height F2 layer (scaler)"),
        "sc_MUF3000_mhz":    ("MHz",  "MUF for 3000-km path"),
        "sc_M3000F2":        ("1",    "Transmission factor MUF(3000)/foF2"),
        "sc_foF2_sigma_mhz": ("MHz",  "Bootstrap sigma foF2"),
        "n_O":               ("1",    "Number of O-mode echoes"),
        "n_X":               ("1",    "Number of X-mode echoes"),
        "n_ambiguous":       ("1",    "Number of ambiguous-mode echoes"),
    }

    scalar_dvars = {
        k: (
            ("time",),
            np.array([r[k] for r in records], dtype=float),
            {"units": units, "long_name": lname},
        )
        for k, (units, lname) in _SCALAR_META.items()
    }

    # ── Assemble Dataset ───────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        coords={
            "time":      ("time",      [r["time"] for r in records]),
            "edp_layer": ("edp_layer", np.arange(max_edp),
                          {"long_name": "EDP lamination layer index"}),
            "drift_h":   ("drift_h",   _DRIFT_H,
                          {"units": "km", "long_name": "Drift height bin centre"}),
        },
        data_vars={
            **scalar_dvars,
            # EDP profiles
            "edp_true_height_km":    (("time", "edp_layer"), edp_true_h,
                                      {"units": "km",  "long_name": "True height EDP"}),
            "edp_plasma_freq_mhz":   (("time", "edp_layer"), edp_fp,
                                      {"units": "MHz", "long_name": "Plasma frequency EDP"}),
            "edp_virtual_height_km": (("time", "edp_layer"), edp_virt_h,
                                      {"units": "km",  "long_name": "Virtual height EDP"}),
            # drift profiles
            "vx_mps":             (("time", "drift_h"), vx_mat,
                                   {"units": "m/s", "long_name": "Eastward drift velocity"}),
            "vy_mps":             (("time", "drift_h"), vy_mat,
                                   {"units": "m/s", "long_name": "Northward drift velocity"}),
            "vz_mps":             (("time", "drift_h"), vz_mat,
                                   {"units": "m/s", "long_name": "Upward drift velocity"}),
            "drift_residual_mps": (("time", "drift_h"), vres_mat,
                                   {"units": "m/s", "long_name": "Mean LS residual"}),
            "drift_cond_number":  (("time", "drift_h"), vcond_mat,
                                   {"units": "1",   "long_name": "LS condition number"}),
            "drift_n_echoes":     (("time", "drift_h"), vn_mat,
                                   {"units": "1",   "long_name": "Echoes used in drift fit"}),
        },
        attrs={
            "title":    "APEP2 VIPIR full-pipeline consolidated output",
            "station":  station,
            "created":  dt.datetime.utcnow().isoformat() + "Z",
            "source":   fname_glob,
            "pipeline": (
                "EchoExtractor → IonogramFilter → PolarizationClassifier "
                "→ TrueHeightInversion → IonogramScaler → fit_drift_from_df"
            ),
            # ── EchoExtractor provenance ───────────────────────────────────
            "extractor_snr_threshold_db":    _EXTRACTOR_SNR_DB,
            "extractor_min_height_km":       _EXTRACTOR_MIN_H_KM,
            "extractor_max_height_km":       _EXTRACTOR_MAX_H_KM,
            "extractor_min_rx_for_direction":_EXTRACTOR_MIN_RX_DIR,
            "extractor_max_echoes_per_pulset":_EXTRACTOR_MAX_ECHO_PS,
            # ── IonogramFilter provenance ──────────────────────────────────
            "filter_ep_max_deg":             _FILTER_EP_MAX_DEG,
            "filter_dbscan_enabled":         int(_FILTER_DBSCAN_ENABLED),
            "filter_dbscan_eps":             _FILTER_DBSCAN_EPS,
            "filter_dbscan_min_samples":     _FILTER_DBSCAN_MIN_SAMPLES,
            "filter_ransac_enabled":         int(_FILTER_RANSAC_ENABLED),
            "filter_ransac_residual_km":     _FILTER_RANSAC_RESIDUAL_KM,
            "filter_ransac_min_samples":     _FILTER_RANSAC_MIN_SAMPLES,
            "filter_ransac_n_iter":          _FILTER_RANSAC_N_ITER,
            "filter_ransac_poly_degree":     _FILTER_RANSAC_POLY_DEG,
            "filter_ransac_min_inlier_frac": _FILTER_RANSAC_MIN_INLIER,
            "filter_multihop_enabled":       int(_FILTER_MULTIHOP_ENABLED),
            "filter_multihop_orders":        str(_FILTER_MULTIHOP_ORDERS),
            "filter_multihop_h_tol_km":      _FILTER_MULTIHOP_H_TOL_KM,
            "filter_multihop_snr_margin_db": _FILTER_MULTIHOP_SNR_MARGIN,
            # ── Drift height-bin provenance ────────────────────────────────
            "drift_height_bin_km":           _DRIFT_BIN_KM,
            "drift_min_echoes_per_bin":      _DRIFT_MIN_ECHO,
            "drift_snr_weight":              int(_DRIFT_SNR_WEIGHT),
            "drift_sigma_rejection":         _DRIFT_N_SIGMA,
        },
    )

    os.makedirs(os.path.dirname(_NC_OUT), exist_ok=True)
    ds_out.to_netcdf(_NC_OUT)
    print(f"\nxarray Dataset saved → {_NC_OUT}")
    print(ds_out)
    print(f"\nOutputs")
    print(f"  Profiles / scalars : {_NC_OUT}")
    print(f"  Raw echoes (ragged): {_PARQUET_OUT}")