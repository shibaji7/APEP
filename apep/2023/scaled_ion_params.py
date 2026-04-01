from pathlib import Path
import sys
sys.path.extend([
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2]),
])

import datetime as dt
import math

from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D

# import iricore
import utils
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.digisonde.parsers.sao import SaoExtractor

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

# Colours chosen to keep frequencies and heights visually distinct.
COLORS = {
    "foF2": "#1f77b4",  # blue
    "foE": "#d62728",  # red
    "hmF2": "#2ca02c",  # green
    "hEs": "#ff7f0e",  # orange
}


def _to_utc_datetime(value) -> dt.datetime:
    """Convert StartTimeUTC strings of the form 'YYYY-MM-DD offset HH:MM:SS.sss' to naive UTC datetimes."""
    if isinstance(value, dt.datetime):
        return value
    try:
        date_part, offset_minutes, time_part = value.split()
    except ValueError:
        raise ValueError(f"Unexpected StartTimeUTC format: {value!r}")

    base_time = dt.datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S.%f")
    tzinfo = dt.timezone(dt.timedelta(minutes=int(offset_minutes)))
    local = base_time.replace(tzinfo=tzinfo).astimezone(dt.timezone.utc).replace(tzinfo=None)
    return base_time


def _extract_peak(freqs: np.ndarray, heights: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return peak frequency and associated height within the masked region."""
    if freqs.size == 0 or not np.any(mask):
        return np.nan, np.nan

    idx = np.where(mask)[0]
    sub_freqs = freqs[idx]
    sub_heights = heights[idx]
    valid = ~np.isnan(sub_freqs)
    if not np.any(valid):
        return np.nan, np.nan
    valid_idx = idx[valid]
    best_local_idx = np.argmax(sub_freqs[valid])
    peak_idx = valid_idx[best_local_idx]
    return freqs[peak_idx], heights[peak_idx]


def compute_iri_series(
    times: Iterable[dt.datetime],
    lat: float,
    lon: float,
    alt_profile: Tuple[int, int, int] = (90, 450, 5),
    version: int = 20,
) -> Dict[str, np.ndarray]:
    """Sample IRI and derive foF2/foE (MHz) and hmF2/hEs (km) for each timestamp."""
    heights = np.arange(alt_profile[0], alt_profile[1] + alt_profile[2], alt_profile[2], dtype=float)
    times = list(times)

    foF2, foE, hmF2, hEs = [], [], [], []
    for event in times:
        iri_output = iricore.iri(event, list(alt_profile), lat, lon, version)
        edens = np.asarray(iri_output.edens, dtype=float)
        freqs = utils.plasma_freq_mhz(edens)

        f2_freq, f2_height = _extract_peak(freqs, heights, heights >= 160)
        es_freq, es_height = _extract_peak(freqs, heights, (heights >= 90) & (heights <= 140))

        foF2.append(f2_freq if f2_freq > 0 else np.nan)
        hmF2.append(f2_height if f2_height > 0 else np.nan)
        foE.append(es_freq if es_freq > 0 else np.nan)
        hEs.append(es_height if es_height > 0 else np.nan)

    return {
        "foF2": np.array(foF2, dtype=float),
        "foE": np.array(foE, dtype=float),
        "hmF2": np.array(hmF2, dtype=float),
        "hEs": np.array(hEs, dtype=float),
    }


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive index ranges where mask is True."""
    indices = np.where(mask)[0]
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return list(zip(starts, ends))


def prepare_scaled_dataframe(extractor: SaoExtractor) -> np.ndarray:
    scaled = extractor.get_scaled_datasets_xml()
    datetimes = [_to_utc_datetime(rec.StartTimeUTC) for rec in extractor.sao.SAORecord]
    scaled["datetime"] = datetimes
    scaled.sort_values("datetime", inplace=True)
    return scaled

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

def plot_station_panel(j, ax, scaled, iri_series, stn_info, time_limits):
    times = scaled["datetime"].to_list()
    freq_ax = ax
    height_ax = ax.twinx()

    # Frequencies
    freq_ax.scatter(times, scaled["foF2"], s=12, color=COLORS["foF2"], ls="None", alpha=0.7, label="foF2 obs")
    freq_ax.plot(times, iri_series["foF2"], color=COLORS["foF2"], lw=1.2, ls="-", label="foF2 IRI")
    freq_ax.scatter(times, scaled["foEs"], s=12, color=COLORS["foE"], alpha=0.7, ls="None", label="foE obs")
    freq_ax.plot(times, iri_series["foE"], color=COLORS["foE"], lw=1.2, ls="-", label="foE IRI")
    freq_ax.set_xlim(time_limits)
    freq_ax.set_ylim(1, 15)
    freq_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    freq_ax.xaxis.set_major_formatter(DateFormatter("%H"))
    freq_ax.grid(True, linestyle="--", alpha=0.2, linewidth=0.5)
    iind, eind = (
        np.argmin(np.abs([t - dt.datetime(2023, 10, 14, 14, 0) for t in times])),
        np.argmin(np.abs([t - dt.datetime(2023, 10, 14, 21, 0) for t in times]))
    )
    segment_times = times[iind:eind]
    Of = utils.get_eclipse_contours_pyEclipse(
        segment_times, stn_info["LAT"], stn_info["LONG"], 
        wl="of", 
        file_ext="_150km_193_1.nc",
        data_folder="data/2023/mask/",
    )
    peak_of = np.nanmax(1-Of)
    ecl_data = eclipse_window(segment_times, Of, threshold=0.01)
    freq_ax.text(0.02, 0.95, r"$\mathcal{O}_{193}^p$: %.2f"%peak_of, transform=freq_ax.transAxes, ha="left", va="top", fontsize=12)
    for k in ecl_data.keys():
        freq_ax.axvline(ecl_data[k], color="k", linestyle="--", linewidth=1.0, alpha=0.7)

    # Heights
    height_ax.scatter(times, scaled["hmF2"], s=14, marker="s", color=COLORS["hmF2"], ls="None", alpha=0.7, label="hmF2 obs")
    height_ax.plot(times, iri_series["hmF2"], color=COLORS["hmF2"], ls="-", lw=1.2, label="hmF2 IRI")
    height_ax.scatter(times, scaled["h`Es"], s=14, marker="s", color=COLORS["hEs"], ls="None", alpha=0.7, label="hmE obs")
    height_ax.plot(times, iri_series["hEs"], color=COLORS["hEs"], ls="-", lw=1.2, label="hmE IRI")
    height_ax.set_ylim(80, 420)
    # Add obs
    ymin, ymax = height_ax.get_ylim()
    of_min, of_max = np.nanmin(Of), np.nanmax(Of)
    of_scaled = Of * (ymax - 60)
    height_ax.plot(
        segment_times,
        of_scaled,
        color="k",
        linestyle="-",
        linewidth=1.0,
        alpha=0.8,
        label="_nolegend_",  # keeps existing legend untouched
    )


    freq_ax.set_ylabel("Frequency (MHz)", color=COLORS["foF2"])
    height_ax.set_ylabel("Height (km)", color=COLORS["hmF2"])

    title = f"({chr(65+j)}) {stn_info['URSI']}/{stn_info['STATIONNAME']}"
    freq_ax.text(0.02, 1.02, title, transform=freq_ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")


def discover_station_files(year: int, limit: int = 6) -> List[Path]:
    base_dir = Path(f"data/{year}")
    if not base_dir.exists():
        return []
    station_files = sorted(base_dir.glob("*_SAO.XML"))
    return station_files[:limit]


def build_legend(fig):
    legend_elements = [
        Line2D([], [], linestyle="none", marker="o", color=COLORS["foF2"], label="foF2 obs", alpha=0.7),
        Line2D([], [], linestyle="-", marker=None, color=COLORS["foF2"], label="foF2 IRI"),
        Line2D([], [], linestyle="none", marker="o", color=COLORS["foE"], label="foE obs", alpha=0.7),
        Line2D([], [], linestyle="-", marker=None, color=COLORS["foE"], label="foE IRI"),
        Line2D([], [], linestyle="none", marker="s", color=COLORS["hmF2"], label="hmF2 obs", alpha=0.7),
        Line2D([], [], linestyle="-", marker=None, color=COLORS["hmF2"], label="hmF2 IRI"),
        Line2D([], [], linestyle="none", marker="s", color=COLORS["hEs"], label="hmE obs", alpha=0.7),
        Line2D([], [], linestyle="-", marker=None, color=COLORS["hEs"], label="hmE IRI"),
    ]
    fig.legend(
        legend_elements,
        [h.get_label() for h in legend_elements],
        loc="upper center",
        ncol=len(legend_elements),
        frameon=False,
        fontsize=15,
        bbox_to_anchor=(0.5, 0.88),
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

def get_eclipse_info(times, stn_info):
    iind, eind = (
        np.argmin(np.abs([t - dt.datetime(2023, 10, 14, 14, 0) for t in times])),
        np.argmin(np.abs([t - dt.datetime(2023, 10, 14, 21, 0) for t in times]))
    )
    segment_times = times[iind:eind]
    Of = utils.get_eclipse_contours_pyEclipse(
        segment_times, stn_info["LAT"], stn_info["LONG"], 
        wl="of", 
        file_ext="_150km_193_1.nc",
        data_folder="data/2023/mask/",
    )
    peak_of = np.nanmax(1-Of)
    ecl_data = eclipse_window(segment_times, Of, threshold=0.01)
    return (ecl_data, peak_of)

def find_deltas(stn_info, ecl, bgc, times):
    ecl, bgc = (
        ecl[
            (ecl.datetime>=times[0])
            & (ecl.datetime<=times[1])
        ],
        bgc[
            (bgc.datetime>=times[0])
            & (bgc.datetime<=times[1])
        ]
    )
    dfoF2, dhmF2, dfoE, dhmE = (
        bgc.foF2.max() - ecl.foF2.min(),
        ecl.hmF2.max() - bgc.hmF2.min(),
        bgc.foE.max() - ecl.foE.min(),
        ecl.hmE.max() - bgc.hmE.min()
    )
    from pynasonde.vipir.ngi.utils import TimeZoneConversion
    tzc = TimeZoneConversion(
                lat=stn_info["LAT"], long=stn_info["LONG"]
            )
    lt = tzc.utc_to_local_time(
                [times[2]]
            )[0]
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(stn_info["URSI"], "%.2f"%dfoF2, "%.2f"%dfoE,"%.2f"%dhmF2,  "%.2f"%dhmE, lt)
    return


def matlab_main():
    target_date = dt.date(2023, 10, 14)
    time_limits = (
        dt.datetime.combine(target_date, dt.time.min),
        dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min),
    )

    stations = ["AU930", "BC840", "AL945", "KR835"]
    tags = ["(A) AU930 / Austin", "(B) BC840 / Boulder", "(C) AL945 / Alpena", "(D) KR835 / Kirtland", "(E) WS833 / WSMR"]
    from fetch import get_bgc_iono_data
    data_dicts = []
    
    for i, stn in enumerate(stations):
        ds, background = get_bgc_iono_data(target_date, time_limits, stn)
        stn_info = get_digisonde_info(stn)
        ecl_data, peak_of = get_eclipse_info(
            ds["datetime"].to_list(),
            stn_info = stn_info
        )
        data_dicts.append(dict(
            dataset=ds,
            background=background,
            xlim=time_limits,
            title_txt=tags[i] + (r" $\mathcal{O}_{193}^p=%.2f$"%peak_of),
            draw_legend=(i==3),
            xlabel_txt="",
            vlines = [ecl_data["start"], ecl_data["peak"], ecl_data["end"]],
            vline_styles = ["--", "-", "--"]
        ))
        find_deltas(stn_info, ds.copy(), background.copy(), [ecl_data["start"], ecl_data["end"],ecl_data["peak"] ])
    
    local = [
        f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_13/",
        f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/",
        f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_15/"
    ]
    scaled = SaoExtractor.load_SAO_files(
        folders=local,
        func_name="scaled",
        n_procs=12,
    )
    stn_info = get_digisonde_info("WS833")
    ecl_data, peak_of = get_eclipse_info(
        ds["datetime"].to_list(),
        stn_info = stn_info
    )
    scaled.rename(columns={"hEs": "h`Es"}, inplace=True)
    datasets = scaled[
        (scaled["datetime"] >= pd.to_datetime(time_limits[0])) & 
        (scaled["datetime"] < pd.to_datetime(time_limits[1]))
    ]
    datasets = datasets[["datetime", "foF2", "hmF2", "foE", "hmE"]]
    bgc = scaled[
        (scaled["datetime"] < pd.to_datetime(time_limits[0])) | 
        (scaled["datetime"] >= pd.to_datetime(time_limits[1]))
    ]
    bgc["dtime"] = bgc.datetime.dt.time
    bgc = bgc.groupby(bgc.datetime.dt.time).agg(
        {
            "foF2": np.nanmean, 
            "hmF2": np.nanmean,
            "foE": np.nanmean,
            "hmE": np.nanmean,
        }
    ).reset_index()
    bgc.datetime = pd.to_datetime(bgc["datetime"], format="%H:%M:%S")
    fixed_date = dt.date(target_date.year, target_date.month, target_date.day)
    days_offset = fixed_date - dt.date(1900, 1, 1)
    bgc.datetime += pd.to_timedelta(days_offset.days, unit="d")
    data_dicts.append(dict(
        dataset=datasets,
        background=bgc,
        xlim=time_limits,
        title_txt=tags[-1]+ (r" $\mathcal{O}_{193}^p=%.2f$"%peak_of),
        draw_legend=False,
        xlabel_txt="Time, UT",
        vlines = [ecl_data["start"], ecl_data["peak"], ecl_data["end"]],
        vline_styles = ["--", "-", "--"]
    ))
    find_deltas(stn_info, datasets.copy(), bgc.copy(), [ecl_data["start"], ecl_data["end"],ecl_data["peak"] ])

    import sys
    sys.path.append("apep/")
    from matlab_engine import CreateFig
    fig = CreateFig(fig_path="manuscript_figures/pdfs/")
    fig.generate_scaled_TS_figure(
        data_dicts, 
        "Figure02.pdf",
        fig_title = "14 October, 2023 Annular Eclipse",
        fig_shape=(12, 1), fontsize=24,
    )
    return

def main():
    target_date = dt.date(2023, 10, 14)
    time_limits = (
        dt.datetime.combine(target_date, dt.time.min),
        dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min),
    )

    station_files = discover_station_files(target_date.year)
    if not station_files:
        raise FileNotFoundError(f"No SAO XML files found under data/{target_date.year}")

    panels = max(6, len(station_files))
    ncols = 2
    nrows = math.ceil(panels / ncols)

    fig = plt.figure(
        figsize=(12, 3.2 * nrows),
    )
    gs = fig.add_gridspec(nrows, ncols*2)
    ax_positions = [gs[0,0:2], gs[0,2:], gs[1,0:2], gs[1,2:]]
    # ax_positions = [gs[0,0:2], gs[1,0:2], gs[2,0:2], gs[3,0:2]]
    axes = [fig.add_subplot(pos) for pos in ax_positions]


    # for ax in axes[-1:]:
    #     ax.set_visible(False)
    j = 0
    for ax, sao_path in zip(axes, station_files):
        code = sao_path.name.split("_")[0]
        extractor = SaoExtractor(str(sao_path), extract_time_from_name=False, extract_stn_from_name=False)
        extractor.extract_xml()
        extractor.date = target_date
        extractor.local_time = target_date
        extractor.stn_info = get_digisonde_info(code.upper())

        for rec in extractor.sao.SAORecord:
            rec.StartTimeUTC = _to_utc_datetime(rec.StartTimeUTC)

        scaled = prepare_scaled_dataframe(extractor)
        iri_series = compute_iri_series(
            scaled["datetime"].tolist(),
            extractor.stn_info["LAT"],
            extractor.stn_info["LONG"],
        )
        plot_station_panel(j, ax, scaled, iri_series, extractor.stn_info, time_limits)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Time (UT)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        j += 1
    
    local = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/"
    scaled = SaoExtractor.load_SAO_files(
        folders=[local],
        func_name="scaled",
        n_procs=12,
    )
    scaled.rename(columns={"hEs": "h`Es"}, inplace=True)
    stn_info = get_digisonde_info("WS833")
    iri_series = compute_iri_series(
        scaled["datetime"].tolist(),
        stn_info["LAT"],
        stn_info["LONG"],
    )
    # ax = fig.add_subplot(gs[4, 0:2])
    ax = fig.add_subplot(gs[2, 1:3])
    plot_station_panel(j, ax, scaled, iri_series, stn_info, time_limits)
    ax.set_xlabel("Time (UT)")

    build_legend(fig)
    fig.suptitle("Plasma Frequency Response during 14 Oct 2023 Eclipse", fontsize=15, y=0.90, fontweight="bold")
    # fig.suptitle("(1) Plasma Frequency Response during 14 Oct 2023 Eclipse", fontsize=15, y=0.87, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0.07, 0.97, 0.88])

    output_dir = Path("figures/2023")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ionosonde_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig("manuscript_figures/Figure02.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    matlab_main()
