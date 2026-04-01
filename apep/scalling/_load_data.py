"""Shared data-loading helper for vipir/analysis examples.

Loads, extracts, and filters echoes from a real VIPIR RIQ file using the
standard pipeline:  RiqDataset → EchoExtractor → IonogramFilter → DataFrame.

Columns in the returned DataFrame
----------------------------------
frequency_khz, height_km, xl_km, yl_km,
polarization_deg, residual_deg, velocity_mps, amplitude_db
"""

import pandas as pd
import numpy as np

from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset


def load_echoes(
    fname,
    snr_threshold_db=3.0,
    ep_max_deg=90.0,
    dbscan_enabled=True,
    ransac_enabled=True,
):
    """Load, extract, and filter echoes from *fname*.

    Parameters
    ----------
    fname : str
        Path to the ``.RIQ`` file (relative to repository root).
    snr_threshold_db : float
        Minimum SNR for echo acceptance in EchoExtractor.
    ep_max_deg : float
        Maximum planar-wavefront residual (EP) to keep.
    dbscan_enabled : bool
        Whether to run the DBSCAN noise-removal stage.
    ransac_enabled : bool
        Whether to run the RANSAC trace-fit stage.

    Returns
    -------
    df : pd.DataFrame
        Filtered echo DataFrame.
    label : str
        Short human-readable label derived from the filename.
    """
    riq = RiqDataset.create_from_file(
        fname,
        unicode="latin-1",
        vipir_config=VIPIR_VERSION_MAP.configs[1],
    )

    print(f"Loaded RIQ : {fname}")
    print(f"  Pulsets  : {len(riq.pulsets)}")
    print(f"  Receivers: {riq.sct.station.rx_count}")
    print(f"  Freq start: {riq.sct.frequency.base_start:.1f} kHz")
    print(f"  Freq end  : {riq.sct.frequency.base_end:.1f} kHz")
    print(f"  Freq steps: {riq.sct.frequency.base_steps}")
    print(f"  Log step  : {riq.sct.frequency.log_step:.4f} (fraction)")
    print(f"  Linear step: {riq.sct.frequency.linear_step:.2f} kHz")
    print(f"  tune_type  : {riq.sct.frequency.tune_type}")
    print(f"  pulse_count: {riq.sct.frequency.pulse_count}")
    extractor = EchoExtractor(
        sct=riq.sct,
        pulsets=riq.pulsets,
        snr_threshold_db=snr_threshold_db,
        min_height_km=60.0,
        max_height_km=1000.0,
        min_rx_for_direction=3,
        max_echoes_per_pulset=5,
    )
    extractor.extract()

    filt = IonogramFilter(
        rfi_enabled=True,
        ep_filter_enabled=True,
        ep_max_deg=ep_max_deg,
        multihop_enabled=True,
        multihop_orders=(2, 3),
        multihop_height_tol_km=50.0,
        multihop_snr_margin_db=6.0,
        dbscan_enabled=dbscan_enabled,
        dbscan_eps=1.0,
        dbscan_min_samples=5,
        dbscan_features=(
            "frequency_khz",
            "height_km",
            "velocity_mps",
            "amplitude_db",
            "residual_deg",
        ),
        ransac_enabled=ransac_enabled,
        ransac_residual_km=100.0,
        ransac_min_samples=10,
        ransac_n_iter=200,
        ransac_poly_degree=3,
        ransac_min_inlier_fraction=0.3,
        temporal_enabled=False,
    )

    df = filt.filter(extractor)

    import os

    label = os.path.basename(fname).split("_")[0]  # e.g. "WI937" or "PL407"
    return df, label


def _fit_group_df(
    grp: pd.DataFrame, min_echoes: int, snr_weight: bool, n_sigma: float
) -> dict | None:
    """Weighted LS fit of [Vx, Vy, Vz] from one height-bin DataFrame."""
    valid = grp.dropna(subset=["xl_km", "yl_km", "velocity_mps"])
    if len(valid) < min_echoes:
        return None

    xl = valid["xl_km"].to_numpy(float)
    yl = valid["yl_km"].to_numpy(float)
    h = valid["height_km"].to_numpy(float)
    v = valid["velocity_mps"].to_numpy(float)

    R = np.sqrt(h**2 + xl**2 + yl**2)
    R = np.where(R > 0, R, h)
    l = xl / R
    m = yl / R
    n = np.sqrt(np.maximum(0.0, 1.0 - l**2 - m**2))
    A = np.column_stack([l, m, n])

    if snr_weight and "snr_db" in valid.columns:
        w = 10.0 ** (valid["snr_db"].fillna(0.0).to_numpy(float) / 20.0)
    else:
        w = np.ones(len(valid))

    mask = np.ones(len(valid), dtype=bool)
    vel = np.zeros(3)
    for _ in range(5):
        if mask.sum() < min_echoes:
            return None
        Aw = A[mask] * w[mask, None]
        vw = v[mask] * w[mask]
        vel, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)
        res = np.abs(v - A @ vel)
        std = res[mask].std()
        if std == 0:
            break
        new_mask = res < n_sigma * std
        if new_mask.sum() < min_echoes:
            break
        mask = new_mask

    Aw = A[mask] * w[mask, None]
    vw = v[mask] * w[mask]
    vel, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)
    res = np.abs(v[mask] - A[mask] @ vel)
    cond = np.linalg.cond(Aw)

    return {
        "vx_mps": float(vel[0]),
        "vy_mps": float(vel[1]),
        "vz_mps": float(vel[2]),
        "residual_mps": float(res.mean()),
        "condition_number": float(cond),
        "n_echoes": int(mask.sum()),
        "n_rejected": int(len(valid) - mask.sum()),
    }


def fit_drift_from_df(
    df: pd.DataFrame,
    height_bin_km: float = 50.0,
    min_echoes: int = 6,
    snr_weight: bool = True,
    n_sigma: float = 2.5,
) -> pd.DataFrame:
    """Height-binned 3-D drift velocity from a filtered echo DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Echo DataFrame (must contain xl_km, yl_km, height_km, velocity_mps).
    height_bin_km : float
        Bin width in km.
    min_echoes, snr_weight, n_sigma
        Passed to the weighted LS fit.

    Returns
    -------
    pd.DataFrame
        Columns: height_bin_km, vx_mps, vy_mps, vz_mps,
                 residual_mps, condition_number, n_echoes, n_rejected.
    """
    df = df.copy()
    df["_bin"] = (df["height_km"] / height_bin_km).astype(
        int
    ) * height_bin_km + height_bin_km / 2.0
    rows = []
    for b, grp in df.groupby("_bin"):
        r = _fit_group_df(grp, min_echoes, snr_weight, n_sigma)
        if r is not None:
            r["height_bin_km"] = float(b)
            rows.append(r)
    if not rows:
        return pd.DataFrame(
            columns=[
                "height_bin_km",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "residual_mps",
                "condition_number",
                "n_echoes",
                "n_rejected",
            ]
        )
    return pd.DataFrame(rows).sort_values("height_bin_km").reset_index(drop=True)