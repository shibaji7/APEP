from pathlib import Path
import sys

sys.path.extend(
    [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[2]),
    ]
)

import datetime as dt
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from fetch import create_local_folder, copy2local
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.digisonde.parsers.dvl import DvlExtractor
import utils

size = 15
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

DEFAULT_STATIONS: List[Tuple[str, Dict[str, str]]] = [
    ("KR835", {"mode": "SKYWAVE", "label": "KR835 / Kirtland"}),
    ("WS833", {"mode": "WSMR", "label": "WS833 / White Sands"}),
]

VELOCITY_SPECS = [
    ("Vx", "$V_x$", "#d62728"),
    ("Vy", "$V_y$", "#1f77b4"),
    ("Vz", "$V_z$", "#2f4f4f"),
]

HEIGHT_COLORS = {
    "Hb": "#9467bd",
    "Ht": "#8c564b",
    "Hmid": "#2ca02c",
}


def copy_dvl(date: dt.datetime, mode: str) -> Path:
    base = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/{mode}_DPS4D_{date.strftime('%Y_%m_%d')}/"
    local, remote = create_local_folder(base)
    local_path = Path(local)
    if not any(local_path.glob("*.DVL")):
        copy2local(local, remote, ext="*.DVL")
    return local_path


def load_dvl_dataframe(date: dt.datetime, station: str, mode: str) -> pd.DataFrame:
    local = copy_dvl(date, mode)
    df = DvlExtractor.load_DVL_files(
        folders=[str(local)],
        ext="*.DVL",
        n_procs=8,
        extract_time_from_name=True,
        extract_stn_from_name=True,
    )
    if "ursi_tag" in df.columns:
        df = df[df["ursi_tag"].str.upper() == station.upper()]
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in [
        "Vx",
        "Vx_err",
        "Vy",
        "Vy_err",
        "Vz",
        "Vz_err",
        "Hb",
        "Ht",
        "lat",
        "lon",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def set_velocity_axis(ax, comp: str, label: str, color: str, data: pd.DataFrame):
    series = data[comp].astype(float)
    ax.plot(data["datetime"], series, color=color, lw=1.2)
    err_key = f"{comp}_err"
    if err_key in data.columns:
        err = data[err_key].astype(float)
        ax.fill_between(
            data["datetime"],
            series - err,
            series + err,
            color=color,
            alpha=0.15,
        )
    vals = series.to_numpy()
    if np.all(np.isnan(vals)):
        vmax = 100.0
    else:
        vmax = np.nanmax(np.abs(vals))
        if not np.isfinite(vmax) or vmax < 20:
            vmax = 100.0
    ax.set_ylim(-1.1 * vmax, 1.1 * vmax)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel(f"{label} (m/s)", color=color)
    ax.tick_params(axis="y", colors=color)
    target_date = dt.date(2024, 4, 8)
    time_limits = (
        dt.datetime.combine(target_date, dt.time.min),
        dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min),
    )
    ax.set_xlim(time_limits)
    return vmax


def plot_station_column(
    p: int,
    axes: np.ndarray,
    df: pd.DataFrame,
    stn_code: str,
    metadata: Dict[str, str],
):
    times = df["datetime"].tolist()
    stn_info = get_digisonde_info(stn_code)
    iind, eind = (
        np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 15, 10) for t in times])),
        np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 21) for t in times])),
    )
    segment_times = times[iind:eind]
    Of = utils.get_eclipse_contours_pyEclipse(
        segment_times,
        stn_info["LAT"],
        stn_info["LONG"],
        wl="193",
        file_ext="_150km_alleof.nc",
        data_folder="data/2024/mask/",
    )
    peak_of = np.nanmax(1 - Of)
    marks = eclipse_window(segment_times, 1 - Of, threshold=0.01)
    title = metadata.get("label") or stn_code
    if stn_info:
        title = f"{stn_code} / {stn_info['STATIONNAME']}"

    for (comp, label, color), ax in zip(VELOCITY_SPECS, axes[:3]):
        vmax = set_velocity_axis(ax, comp, label, color, df)
        if comp == "Vz":
            Ofx = savgol_filter(Of, window_length=7, polyorder=3, mode="interp")
            amplitude = -1 * vmax * (0.5 - (1 - Ofx)) * 5
            eclipse_curve = np.diff(amplitude)
            ax.plot(
                segment_times[1:],
                eclipse_curve,
                color="#ff7f50",
                lw=3,
                ls="-",
                alpha=0.85,
                label="_nolegend_",
            )
        # else:
        #     Ofx = savgol_filter(Of, window_length=7, polyorder=3, mode="interp")
        #     amplitude = vmax * (0.5 - (1 - Ofx))
        #     amplitude = amplitude - max(amplitude)
        #     ax.plot(
        #         segment_times,
        #         amplitude,
        #         color="#ff7f50",
        #         lw=1.3,
        #         ls="-",
        #         alpha=0.85,
        #         label="_nolegend_",
        #     )
        for key, style in {
            "start": {"ls": "--", "alpha": 0.6},
            "peak": {"ls": "--", "alpha": 0.85},
            "end": {"ls": "--", "alpha": 0.6},
        }.items():
            if marks:
                ax.axvline(
                    marks[key],
                    color="k",
                    linestyle=style["ls"],
                    linewidth=0.9,
                    alpha=style["alpha"],
                )
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter("%H"))
        tax = ax.twinx()
        tax.yaxis.set_visible(False)
        tax.plot(
            segment_times,
            Of,
            color="k",
            linestyle="-",
            linewidth=1.0,
            alpha=0.8,
            label="_nolegend_",  # keeps existing legend untouched
        )

    axes[0].set_title(title, fontweight="bold")
    axes[0].text(
        0.02,
        0.95,
        r"$\mathcal{O}_{193}^p$: %.2f" % peak_of,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    height_ax = axes[3]
    hb = df["Hb"].astype(float)
    ht = df["Ht"].astype(float)
    hmid = 0.5 * (hb + ht)
    
    if hmid.notna().any():
        height_ax.plot(
            df["datetime"], hmid, color=HEIGHT_COLORS["Hmid"], lw=1.2, label="(Hb+Ht)/2", ls="-"
        )
    height_ax.set_ylabel("Virtual height (km)")
    height_ax.set_ylim(150, 500)
    height_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    height_ax.xaxis.set_major_formatter(DateFormatter("%H"))

    obs_ax = height_ax.twinx()
    obs_ax.yaxis.set_visible(False)
    obs_ax.plot(
        segment_times,
        Of,
        color="k",
        linestyle="-",
        linewidth=1.0,
        alpha=0.8,
        label="_nolegend_",  # keeps existing legend untouched
    )
    obs_ax.set_ylim(0, 1.05)
    obs_ax.set_ylabel("Obscuration", color="k")
    obs_ax.tick_params(axis="y", colors="k")

    for key, style in {
        "start": {"ls": "--", "alpha": 0.6},
        "peak": {"ls": "--", "alpha": 0.85},
        "end": {"ls": "--", "alpha": 0.6},
    }.items():
        if marks:
            height_ax.axvline(
                marks[key],
                color="k",
                linestyle=style["ls"],
                linewidth=0.9,
                alpha=style["alpha"],
            )

    if marks:
        height_ax.text(
            marks["peak"],
            height_ax.get_ylim()[0] + 10,
            "Peak",
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=90,
            color="k",
        )

    height_ax.grid(True, ls="--", alpha=0.2)
    for i, ax in enumerate(axes):
        ax.text(0.95, 0.9, f"({chr(65+i)}-{p})", ha="right", va="center", transform=ax.transAxes)


def build_legend(fig: plt.Figure):
    handles = [
        Line2D([], [], color=spec[2], lw=1.5, label=spec[1]) for spec in VELOCITY_SPECS
    ]
    handles.extend(
        [
            # Line2D([], [], color=HEIGHT_COLORS["Hb"], lw=1.0, label="Hb"),
            # Line2D([], [], color=HEIGHT_COLORS["Ht"], lw=1.0, label="Ht"),
            Line2D([], [], color=HEIGHT_COLORS["Hmid"], lw=1.0, label=r"$H_v$"),
            # Line2D([], [], color="k", lw=1.0, ls="--", label="Obscuration"),
        ]
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=12,
        columnspacing=1.5,
        handlelength=2.5,
        bbox_to_anchor=(0.5, 0.92),
    )


def create_dvl_summary(
    date: dt.datetime = dt.datetime(2024, 4, 8),
    stations: List[Tuple[str, Dict[str, str]]] = None,
    outfile: Path = Path("figures/2024/dvl_summary.png"),
):
    stations = stations or DEFAULT_STATIONS
    ncols = len(stations)
    fig, axes = plt.subplots(
        4,
        ncols,
        figsize=(6 * ncols, 10),
        sharex="col",
        constrained_layout=False,
    )
    if ncols == 1:
        axes = np.atleast_2d(axes).reshape(4, 1)

    for col, (stn_code, meta) in enumerate(stations):
        df = load_dvl_dataframe(date, stn_code, meta["mode"])
        if df.empty:
            raise ValueError(f"No DVL records found for {stn_code} in mode {meta['mode']}")
        column_axes = axes[:, col]
        plot_station_column(col+1, column_axes, df, stn_code, meta)
        for row_ax in column_axes[:-1]:
            row_ax.tick_params(labelbottom=False)
        column_axes[-1].set_xlabel("Time (UT)")

    build_legend(fig)
    fig.suptitle(
        f"Doppler Response during 08 Apr 2024 Eclipse",
        fontsize=16,
        fontweight="bold",
        y=0.94,
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.96, 0.93])

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig("manuscript_figures/Figure06.png", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def get_eclipse_info(times, stn_info):
    iind, eind = (
        np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 15, 0) for t in times])),
        np.argmin(np.abs([t - dt.datetime(2024, 4, 8, 21, 0) for t in times]))
    )
    
    segment_times = times[iind+1:eind-1]
    Of = utils.get_eclipse_contours_pyEclipse(
        segment_times, stn_info["LAT"], stn_info["LONG"], 
        wl="193", 
        file_ext="_150km_alleof.nc",
        data_folder="data/2024/mask/",
    )
    peak_of = np.nanmax(1-Of)
    ecl_data = eclipse_window(segment_times, Of, threshold=0.01)
    return (ecl_data, peak_of)

def create_dvl_summary_matlab():
    date = dt.date(2024, 4, 8)
    time_limits = (
        dt.datetime.combine(date, dt.time.min),
        dt.datetime.combine(date + dt.timedelta(days=1), dt.time.min),
    )
    stations = DEFAULT_STATIONS
    ncols = len(stations)
    data_dicts = []
    tags = ["(A) KR835 / Kirtland", "(B) WS833 / WSMR"]

    for col, (stn_code, meta) in enumerate(stations):
        df = load_dvl_dataframe(date, stn_code, meta["mode"])
        ecl_data, peak_of = get_eclipse_info(
            df["datetime"].to_list(),
            stn_info = get_digisonde_info(stn_code)
        )
        data_dicts.append(dict(
            dataset=df,
            xlim=time_limits,
            title_txt=tags[col] + (r" $\mathcal{O}_{193}^p=%.2f$"%peak_of),
            draw_legend=False,
            xlabel_txt="",
            vlines = [ecl_data["start"], ecl_data["peak"], ecl_data["end"]],
            vline_styles = ["--", "-", "--"]
        ))

    import sys
    sys.path.append("apep/")
    from matlab_engine import CreateFig
    fig = CreateFig(fig_path="manuscript_figures/pdfs/")
    fig.generate_doppler_figure(
        data_dicts, 
        "Figure06.pdf",
        fig_title = "08 April, 2024 Total Eclipse",
        fig_shape=(12, 3), fontsize=30,
    )
    return

if __name__ == "__main__":
    # create_dvl_summary()
    create_dvl_summary_matlab()
