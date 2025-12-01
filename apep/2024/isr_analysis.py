from pathlib import Path
import sys
sys.path.extend([
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2]),
])

import datetime as dt
import pandas as pd
import numpy as np

from typing import Dict, Iterable, List, Tuple

from loguru import logger
import h5py
import os
from matplotlib.colors import LogNorm
import utils

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
size = 15
import matplotlib as mpl
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
mpl.rcParams.update(
    {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
)

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

def fetch_isr_data(file_path: str = "data/2024/mlh240408g.003.hdf5"):
    """Fetch ISR data from a NetCDF file and return as an xarray Dataset."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    DS = dict(header=dict(), data=None)
    with h5py.File(file_path, 'r') as f:
        head = []
        for h in range(len(f["Metadata"]["Data Parameters"][()])):
            key = f["Metadata"]["Data Parameters"][()][h]
            DS["header"][key[0].decode('utf-8')] = dict(
                name=key[1].decode('utf-8'),
                units=key[3].decode('utf-8'),
                description=key[-1].decode('utf-8'),
            )
            head.append(key[0].decode('utf-8'))
        # Convert the structured np.void records into a plain list of dicts
        table_layout = f["Data"]["Table Layout"][()]
        records = []
        for row in table_layout:
            record = []
            for value in row:
                if isinstance(value, bytes):
                    record.append(value.decode("utf-8").strip())
                else:
                    record.append(value)
            records.append(record)
        DS["data"] = pd.DataFrame(records, columns=head)
        # Convert UNIX timestamp columns to timezone-aware datetimes for analysis
        for col in ("UT1_UNIX", "UT2_UNIX"):
            if col in DS["data"].columns:
                DS["data"][col] = pd.to_datetime(
                    DS["data"][col],
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
    return DS

def plot_isr_data(ds):
    """Plot electron density (NE) vs. UT1 time and altitude."""
    if not ds or ds.get("data") is None:
        logger.error("Dataset is empty; call fetch_isr_data first.")
        return None, None

    df = ds["data"]
    required_cols = {"UT1_UNIX", "GDALT", "NE", "GDLAT", "GLON"}
    if not required_cols.issubset(df.columns):
        logger.error(f"Missing required columns: {required_cols - set(df.columns)}")
        return None, None

    data = df.dropna(subset=list(required_cols)).copy()
    if data.empty:
        logger.error("No valid rows to plot after dropping NaNs.")
        return None, None

    # Bin by altitude/time to create a dense grid for pcolormesh
    alt_bin_size = 5  # km
    time_bin = "5min"
    data = data[data["GDALT"].between(100, 500)]
    data["GDALT_BIN"] = (data["GDALT"] / alt_bin_size).round(0) * alt_bin_size
    data["UT1_BIN"] = data["UT1_UNIX"].dt.floor(time_bin)

    pivot = (
        data.groupby(["GDALT_BIN", "UT1_BIN"])["NE"]
        .median()
        .unstack()
        .sort_index()
    )

    time_index = pivot.columns
    if getattr(time_index, "tz", None) is not None:
        time_index = time_index.tz_convert("UTC").tz_localize(None)

    time_vals = mdates.date2num(time_index.to_pydatetime())
    alt_vals = pivot.index.to_numpy()
    X, Y = np.meshgrid(time_vals, alt_vals)
    peak_series = pivot.idxmax(axis=0).astype(float)
    peak_altitudes = (
        peak_series.rolling(window=10, center=True, min_periods=1).median().to_numpy()
    )

    start_day = time_index.min().floor("D")
    end_day = start_day + pd.Timedelta(days=1)
    xtick_times = pd.date_range(start=start_day, end=end_day, freq="2H")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    pcm = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=pivot.to_numpy().ravel(),
        # shading="nearest",
        cmap="hsv",
        s=100,
        marker="s",
        norm=LogNorm(vmin=1e10, vmax=1e12),
    )
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Millstone Hill ISR / 08 April 2024")
    ax.set_ylim(100, 500)
    ax.set_xlim(
        mdates.date2num(start_day.to_pydatetime()),
        mdates.date2num(end_day.to_pydatetime()),
    )
    ax.set_xticks(mdates.date2num(xtick_times.to_pydatetime()))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.plot(time_vals, peak_altitudes, "k+", markersize=6, markeredgewidth=1.5)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"Electron Density, $m^-3$")
    ax.set_xlim(
        dt.datetime(2024, 4, 8, 14, 0),
        dt.datetime(2024, 4, 9),
    )


    segment_times = data.UT1_UNIX.unique().tolist()[17*60:22*60]
    print(segment_times[0], segment_times[-1])
    Of = utils.get_eclipse_contours_pyEclipse(
        segment_times, data["GDLAT"].tolist()[0], 
        data["GLON"].tolist()[0], 
        wl="193", 
        file_ext="_150km_alleof.nc",
        data_folder="data/2024/mask/",
    )
    peak_of = np.nanmax(1-Of)
    ecl_data = eclipse_window(segment_times, Of, threshold=0.01)
    tax = ax.twinx()
    tax.plot(
        segment_times,
        Of,
        color="k",
        linestyle=":",
        linewidth=4.0,
        alpha=0.8,
        label="_nolegend_",  # keeps existing legend untouched
    )
    tax.yaxis.set_visible(False)
    ax.text(0.02, 1.1, r"$\mathcal{O}_{193}^p$: %.2f"%peak_of, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    for k in ecl_data.keys():
        ax.axvline(ecl_data[k], color="k", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()

    output_path = Path("figures/2024/isr.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path("manuscript_figures/Figure09.png"), dpi=300, bbox_inches="tight")

    return fig, ax

if __name__ == "__main__":
    DS = fetch_isr_data()
    print(DS["data"].columns)
    plot_isr_data(DS)
    
