"""Generate frequency–time interval panels for multiple frequency bands.

This script samples VIPIR NGI ionogram archives, averages O-mode power within
specified frequency bands, and renders a compact grid of frequency–time plots
sharing a single colour bar. It is memory-conscious: the NGI snapshots are
processed sequentially and only the band-averaged power profiles are retained.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.extend(
    [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[2]),
    ]
)

import datetime as dt
import math
import os
import shutil

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D

import scienceplots
from pynasonde.vipir.ngi.source import DataSource
import utils
from pynasonde.digisonde.digi_utils import get_digisonde_info

plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
mpl.rcParams.update({"xtick.labelsize": 18, "ytick.labelsize": 18, "font.size": 18})


MODE = "O"
RANGE_LIMITS = (50.0, 400.0)
DATE_LIM = (dt.datetime(2024, 4, 8, 16), dt.datetime(2024, 4, 8, 23))

BANDS = [
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 12.0),
]

def eclipse_window(times: Iterable[dt.datetime], obscuration: np.ndarray, threshold: float = 0.05) -> Dict[str, dt.datetime]:
    """Return start/peak/end for 1-Of once it exceeds the threshold."""
    times = np.asarray(list(times), dtype=object)
    flipped = 1.0 - np.asarray(obscuration, dtype=float)

    valid = np.isfinite(flipped)
    if not np.any(valid):
        return {}

    flipped = flipped[valid]
    times = times[valid]

    above = flipped >= threshold
    if not np.any(above):
        return {}

    start_idx = np.argmax(above)
    end_idx = len(above) - np.argmax(above[::-1]) - 1
    peak_idx = np.nanargmax(flipped)

    return {
        "start": times[start_idx],
        "peak": times[peak_idx],
        "end": times[end_idx],
    }


def load_band_cubes(folder: str, bands: list[tuple[float, float]]):
    """Return band-averaged power matrices and the associated range axis."""

    logger.info("Loading NGI datasets from %s", folder)
    ds = DataSource(source_folder=folder)
    ds.load_data_sets(0, -1, n_jobs=24)

    band_data = {band: {"times": [], "values": []} for band in bands}
    range_axis: np.ndarray | None = None

    rng_min, rng_max = RANGE_LIMITS

    for dataset in ds.datasets:
        time = dt.datetime(
            dataset.year,
            dataset.month,
            dataset.day,
            dataset.hour,
            dataset.minute,
            dataset.second,
        )
        if not (DATE_LIM[0] <= time <= DATE_LIM[1]):
            continue

        freq_axis = np.asarray(dataset.Frequency, dtype=float) / 1e3
        rng_axis = np.asarray(dataset.Range, dtype=float)
        rng_mask = (rng_axis >= rng_min) & (rng_axis <= rng_max)
        if not np.any(rng_mask):
            continue

        if range_axis is None:
            range_axis = rng_axis[rng_mask]
        elif range_axis.shape[0] != np.sum(rng_mask):
            logger.warning("Range axis changed shape at %s; skipping snapshot", time)
            continue

        power_cube = getattr(dataset, f"{MODE}_mode_power")
        if power_cube.shape[1] != rng_axis.size:
            logger.warning("Unexpected range dimension at %s; skipping", time)
            continue

        for band in bands:
            f_min, f_max = band
            freq_mask = (freq_axis >= f_min) & (freq_axis < f_max)
            if not np.any(freq_mask):
                continue
            band_power = power_cube[freq_mask][:, rng_mask]
            band_mean = np.nanmean(band_power, axis=0)
            band_data[band]["times"].append(time)
            band_data[band]["values"].append(band_mean)

    band_grids: dict[tuple[float, float], dict[str, np.ndarray] | None] = {}
    for band, payload in band_data.items():
        times = payload["times"]
        values = payload["values"]
        if not times:
            band_grids[band] = None
            continue
        order = np.argsort(times)
        sorted_times = np.array(times, dtype="datetime64[ns]")[order]
        sorted_values = np.array(values)[order]
        band_grids[band] = {
            "times": sorted_times,
            "matrix": sorted_values,
        }

    return range_axis, band_grids


def plot_multi_band_fti(
    folder: str,
    bands: list[tuple[float, float]],
    date: dt.datetime,
    station: str,
    fig_file: Path,
):
    stn_info = get_digisonde_info("WP937")
    range_axis, band_grids = load_band_cubes(folder, bands)
    if range_axis is None:
        logger.warning("No range data available; aborting plot generation")
        return

    available = [grid for grid in band_grids.values() if grid]
    if not available:
        logger.warning("No data available in requested frequency bands")
        return

    mins, maxs = [], []
    for grid in available:
        values = grid["matrix"].astype(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        mins.append(finite.min())
        maxs.append(finite.max())

    if not mins:
        logger.warning("Band matrices contain only NaNs; aborting plot generation")
        return

    vmin, vmax = min(mins), max(maxs)

    n_panels = len(bands)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    im = None
    for idx, band in enumerate(bands):
        ax = axes_flat[idx]
        grid = band_grids.get(band)
        if not grid:
            ax.set_visible(False)
            continue

        times = pd.to_datetime(grid["times"])
        matrix = grid["matrix"].astype(float)
        if matrix.size == 0:
            ax.set_visible(False)
            continue

        t_mesh, r_mesh = np.meshgrid(
            mdates.date2num(times),
            range_axis,
            indexing="ij",
        )

        im = ax.pcolormesh(
            t_mesh,
            r_mesh,
            matrix,
            shading="auto",
            cmap="Spectral",
            vmin=vmin,
            vmax=vmax,
        )
        ax.text(
            0.02,
            0.94,
            f"({chr(65+idx)}) {band[0]:.0f}–{band[1]:.0f} MHz",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlim(DATE_LIM)
        ax.set_ylim(RANGE_LIMITS)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(r"%H"))

        iind, eind = (
            np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 17, 30) for t in times])),
            np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 21, 0) for t in times]))
        )
        segment_times = times[iind:eind]
        Of = utils.get_eclipse_contours_pyEclipse(
            segment_times, stn_info["LAT"], stn_info["LONG"], 
            wl="193", 
            file_ext="_150km_alleof.nc",
            data_folder="data/2024/mask/",
        )
        peak_of = np.nanmax(1-Of)
        ecl_data = eclipse_window(segment_times[::5], Of[::5], threshold=0.01)
        for k in ecl_data.keys():
            ax.axvline(ecl_data[k], color="k", linestyle="--", linewidth=1.0, alpha=0.7)
        
        if idx == 0:
            ax.text(
                0.05,
                1.15,
                r"$\mathcal{O}_{193}^p$=%.2f"%peak_of,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                fontweight="bold",
            )

        tax = ax.twinx()
        tax.plot(
            segment_times[::5],
            Of[::5],
            color="k",
            linestyle="-",
            linewidth=1.0,
            alpha=0.8,
            label="_nolegend_",  # keeps existing legend untouched
        )
        tax.tick_params(axis="y", length=0, labelright=False)


    for ax in axes_flat:
        if not ax.get_visible():
            continue
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Virtual Height (km)")
        else:
            ax.set_ylabel("")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Time (UT)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    fig.suptitle(
        f"VIPIR ({station}) – Multi-band FTI ({date:%d %b %Y})",
        fontsize=15,
        fontweight="bold",
        y=0.85,
    )

    if im is not None:
        ref_ax = axes_flat[min(len(bands) - 1, len(axes_flat) - 1)]
        cpos = [0.9, 0.2, 0.025, 0.2]
        cax = fig.add_axes(cpos)
        cbar = fig.colorbar(
            im,
            ax=ref_ax,
            cax=cax,
        )
        cbar.set_label("O-mode Power (dB)")

    fig.tight_layout(rect=[0.03, 0.1, 0.9, 0.94])
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_file, dpi=300)
    os.system(f"cp {fig_file}  manuscript_figures/Figure07.png")
    plt.close(fig)
    logger.info(f"Saved multi-band FTI figure to {fig_file}")


def main():
    data_root = Path(
        os.environ.get(
            "VIPIR_SPEED_DEMON_ROOT",
            "/media/chakras4/Crucial X9/Solar_Eclipse_2024/public/WI937/individual/",
        )
    )
    temp_root = Path("/tmp/vipir_fti")
    station = "WI937"

    date, doy = dt.datetime(2024, 4, 8), "099"
    src = data_root / "2024" / doy / "ionogram"
    tmp = temp_root / doy / "ionogram"
    if not tmp.exists():
        shutil.copytree(src, tmp)

    try:
        plot_multi_band_fti(
            folder=str(tmp),
            bands=BANDS,
            date=date,
            station=station,
            fig_file=Path(f"figures/fti_multiband.{station}.{date:%Yj}.png"),
        )
    finally:
        pass


if __name__ == "__main__":
    main()
