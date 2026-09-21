"""
JGR Figures 2 (14 Oct 2023) and 4 (08 Apr 2024): 5 x 4 ionosonde time series.

Rows    : stations (AU930, BC840, AL945, KR835, WS833)
Columns : foF2 | h'F2 | foE | h'E
Each panel: eclipse-day scaled values (solid, colour) over the quiet-day
median baseline (grey dashed), with the eclipse interval shaded.

Data (Datasets/Data): <STN>_<YEAR>_scaled.csv (event day) and
<STN>_<YEAR>_median.csv (quiet-day median vs. hour of day).

Usage:  python -m apep.jgr.jgr_timeseries_5x4 [--year 2023|2024|both]
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def find_up(rel):
    """First existing `<ancestor>/<rel>` walking up from this file (works from a worktree)."""
    for p in HERE.parents:
        if (p / rel).exists():
            return p / rel
    return REPO / rel


DATA_DIR = find_up("Datasets/Data")
OUT_DIR = REPO / "manuscript_figures" / "jgr"

STATIONS = ["AU930", "BC840", "AL945", "KR835", "WS833"]
STATION_LABEL = {
    "AU930": "AU930 Austin", "BC840": "BC840 Boulder", "AL945": "AL945 Alpena",
    "KR835": "KR835 Kirtland", "WS833": "WS833 White Sands",
}

# key, event csv columns, median csv columns (first populated wins), title, unit.
# AU/BC/AL medians come from DIDBase (hF2/hE); KR/WS medians use h_prime_*.
COLUMNS = [
    ("foF2", ["foF2_mhz"], ["foF2_median"], r"$f_oF_2$", "MHz"),
    ("hF2", ["h_prime_F2_km"], ["h_prime_F2_median", "hF2_median"], r"$h'F_2$", "km"),
    ("foE", ["foE_mhz"], ["foE_median"], r"$f_oE$", "MHz"),
    ("hE", ["h_prime_E_km"], ["h_prime_E_median", "hE_median"], r"$h'E$", "km"),
]

EVENTS = {
    2023: dict(date=dt.date(2023, 10, 14), window=(13, 21), mask_window=(14, 21), fig="Figure02",
               mask="data/2023/mask", wl="of", ext="_150km_193_1.nc"),
    2024: dict(date=dt.date(2024, 4, 8), window=(16, 21), fig="Figure04",
               mask="data/2024/mask", wl="193", ext="_150km_alleof.nc"),
}

EVENT_COLOR = "#0B3C7A"
BASE_COLOR = "#8C8C8C"
SHADE_COLOR = "#F2C14E"
GAP_MINUTES = 25
MAX_INTERP_GAP = "90min"  # linearly interpolate gaps up to this length; None disables
INTERP_STEP = "5min"
SMOOTH = "30min"
SMOOTH_MEAN = "15min"
SMOOTH_HEIGHT = "60min"
SMOOTH_HEIGHT_MEAN = "30min"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "xtick.minor.visible": True,
        "lines.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _first_col(df, names):
    for n in names:
        if n in df.columns and df[n].notna().any():
            return df[n]
    return pd.Series(np.nan, index=df.index)


def fill_gaps(df, max_gap=None, step=INTERP_STEP):
    """Linear (time) interpolation across gaps up to `max_gap`; longer gaps are left empty."""
    max_gap = MAX_INTERP_GAP if max_gap is None else max_gap
    if df.empty or not max_gap or len(df) < 2:
        return df
    df = df.sort_values("t").drop_duplicates("t")
    t = df["t"].to_numpy()
    v = df["v"].to_numpy(float)
    limit, dstep = pd.Timedelta(max_gap), pd.Timedelta(step)
    ts, vs = [t[0]], [v[0]]
    for i in range(1, len(t)):
        gap = pd.Timestamp(t[i]) - pd.Timestamp(t[i - 1])
        if dstep < gap <= limit:
            k = np.arange(1, int(gap / dstep) + (1 if gap % dstep else 0))
            tk = pd.Timestamp(t[i - 1]) + k * dstep
            tk = tk[tk < pd.Timestamp(t[i])]
            ts.extend(tk)
            vs.extend(np.interp(tk.astype("int64"), [pd.Timestamp(t[i - 1]).value, pd.Timestamp(t[i]).value],
                                [v[i - 1], v[i]]))
        ts.append(t[i])
        vs.append(v[i])
    return pd.DataFrame({"t": pd.to_datetime(ts), "v": vs})


def despike(df, window=SMOOTH, mean_window=SMOOTH_MEAN):
    """Centred time-based running median (optionally followed by a running mean)."""
    if df.empty or not window:
        return df
    v = df.set_index("t")["v"].rolling(window, center=True, min_periods=1).median()
    if mean_window:
        v = v.rolling(mean_window, center=True, min_periods=1).mean()
    return pd.DataFrame({"t": v.index, "v": v.to_numpy()})


def load_station(stn, year, smooth=True):
    ev = DATA_DIR / f"{stn}_{year}_scaled.csv"
    md = DATA_DIR / f"{stn}_{year}_median.csv"
    if not (ev.exists() and md.exists()):
        return None
    ed = pd.read_csv(ev, parse_dates=["time"])
    md = pd.read_csv(md)
    base_t = pd.Timestamp(EVENTS[year]["date"]) + pd.to_timedelta(md["time_of_day"], unit="h")
    out = {}
    for key, ecols, mcols, _, _ in COLUMNS:
        e = pd.DataFrame({"t": ed["time"], "v": _first_col(ed, ecols)}).dropna()
        b = pd.DataFrame({"t": base_t, "v": _first_col(md, mcols)}).dropna()
        e = fill_gaps(e.sort_values("t"))
        if smooth:
            e = despike(e, SMOOTH_HEIGHT, SMOOTH_HEIGHT_MEAN) if key in ("hF2", "hE") else despike(e)
        out[key] = (e, fill_gaps(b.sort_values("t")))
    return out


def eclipse_window(stn, year):
    """dict(start, peak, end, obsc) from the pyEclipse masks, or {} if unavailable."""
    cfg = EVENTS[year]
    try:
        sys.path.extend([str(REPO), str(REPO / "apep")])
        import utils
        from pynasonde.digisonde.digi_utils import get_digisonde_info

        info = get_digisonde_info(stn)
        d, (h0, h1) = cfg["date"], cfg.get("mask_window", cfg["window"])
        times = [dt.datetime(d.year, d.month, d.day, h0) + dt.timedelta(minutes=5 * i)
                 for i in range(int((h1 - h0) * 12))]
        Of = np.asarray(utils.get_eclipse_contours_pyEclipse(
            times, info["LAT"], info["LONG"], wl=cfg["wl"], file_ext=cfg["ext"],
            data_folder=str(find_up(cfg["mask"])) + "/"), float)
        obsc = 1.0 - Of
        idx = np.where(np.isfinite(obsc) & (obsc >= 0.01))[0]
        if not len(idx):
            return {}
        return dict(start=times[idx[0]], end=times[idx[-1]],
                    peak=times[int(np.nanargmax(np.where(np.isfinite(obsc), obsc, -1)))],
                    obsc=float(np.nanmax(obsc)))
    except Exception as exc:
        print(f"  eclipse window unavailable for {stn}: {exc}")
        return {}


def plot_line(ax, df, **kw):
    """Plot a series, breaking the line across data gaps."""
    if df.empty:
        return
    t = df["t"].to_numpy()
    v = df["v"].to_numpy(float)
    dts = np.diff(t).astype("timedelta64[s]").astype(float)
    brk = np.where(dts > GAP_MINUTES * 60)[0] + 1
    for ts, vs in zip(np.split(t, brk), np.split(v, brk)):
        ax.plot(ts, vs, **kw)


def column_limits(data, key, window, date):
    t0 = pd.Timestamp(date) + pd.Timedelta(hours=window[0])
    t1 = pd.Timestamp(date) + pd.Timedelta(hours=window[1])
    lo, hi = [], []
    for d in data.values():
        if d is None:
            continue
        for df in d[key]:
            s = df[(df["t"] >= t0) & (df["t"] <= t1)]["v"]
            if len(s):
                lo.append(s.quantile(0.01))
                hi.append(s.quantile(0.99))
    if not lo:
        return None
    a, b = min(lo), max(hi)
    pad = 0.08 * (b - a)
    return a - pad, b + pad


def median_baseline(stn, year):
    """Quiet-day medians (true heights hmF2/hmE) from <STN>_<YEAR>_median.csv, keyed like COLUMNS."""
    f = DATA_DIR / f"{stn}_{year}_median.csv"
    if not f.exists():
        return {}
    md = pd.read_csv(f)
    t = pd.Timestamp(EVENTS[year]["date"]) + pd.to_timedelta(md["time_of_day"], unit="h")
    cols = {"foF2": "foF2_median", "hF2": "hmF2_median", "foE": "foE_median", "hE": "hmE_median"}
    out = {}
    for key, c in cols.items():
        if c in md.columns and md[c].notna().any():
            out[key] = fill_gaps(pd.DataFrame({"t": t, "v": md[c]}).dropna().sort_values("t"))
    return out


def frames_to_station_data(ev, bg, smooth=True, outlier_iqr=3.0, median_bg=None):
    """Convert SAO-style frames (datetime, foF2, hmF2, foE, hmE) to the plot structure."""
    old = {"foF2": "foF2", "hF2": "hmF2", "foE": "foE", "hE": "hmE"}
    out = {}
    for key, col in old.items():
        e = pd.DataFrame({"t": pd.to_datetime(ev["datetime"]), "v": ev[col]}).dropna()
        if outlier_iqr and len(e) > 4:
            q1, q3 = e["v"].quantile([0.25, 0.75])
            e = e[e["v"].between(q1 - outlier_iqr * (q3 - q1), q3 + outlier_iqr * (q3 - q1))]
        e = fill_gaps(e.sort_values("t"))
        if smooth:
            e = despike(e, SMOOTH_HEIGHT, SMOOTH_HEIGHT_MEAN) if key in ("hF2", "hE") else despike(e)
        b = pd.DataFrame({"t": pd.to_datetime(bg["datetime"]), "v": bg[col]}).dropna() \
            if bg is not None and len(bg) else pd.DataFrame({"t": [], "v": []})
        b = fill_gaps(b.sort_values("t"))
        if median_bg and key in median_bg:
            out[key] = (e, median_bg[key])
            continue
        if median_bg is not None:
            print(f"  note: no median-CSV baseline for {key}; using SAO mean")
        if smooth and len(b):
            b = despike(b, "45min", "45min") if key in ("hF2", "hE") else despike(b, "30min", "30min")
        out[key] = (e, b)
    return out


def split_event_background(df, time_limits):
    """Split an SAO frame into the event window and a time-of-day mean of the remaining days."""
    t0, t1 = pd.to_datetime(time_limits[0]), pd.to_datetime(time_limits[1])
    ev = df[(df["datetime"] >= t0) & (df["datetime"] < t1)][["datetime", "foF2", "hmF2", "foE", "hmE"]]
    rest = df[(df["datetime"] < t0) | (df["datetime"] >= t1)].copy()
    tod = rest["datetime"].dt.time
    bg = rest.groupby(tod)[["foF2", "hmF2", "foE", "hmE"]].mean().reset_index()
    bg["datetime"] = [dt.datetime.combine(t0.date(), x) for x in bg["datetime"]]
    return ev, bg


def make_figure(year, out_dir=OUT_DIR, smooth=True):
    data = {s: load_station(s, year, smooth) for s in STATIONS}
    draw_figure(year, data, out_dir)


def draw_figure(year, data, out_dir=OUT_DIR, titles=None, baseline_label="Quiet-day median"):
    """data: {station: {key: (event_df, baseline_df)} or None}; titles overrides column titles."""
    apply_style()
    titles = titles or {}
    cfg = EVENTS[year]
    date, window = cfg["date"], cfg["window"]
    t0 = dt.datetime(date.year, date.month, date.day, window[0])
    t1 = dt.datetime(date.year, date.month, date.day, window[1])

    limits = {c[0]: column_limits(data, c[0], window, date) for c in COLUMNS}

    fig, axes = plt.subplots(5, 4, figsize=(7.08, 7.6), sharex=True)
    eclipses = {s: eclipse_window(s, year) if data[s] is not None else {} for s in STATIONS}
    order = sorted(STATIONS, key=lambda s: -eclipses[s].get("obsc", -1.0))  # highest obscuration first
    for i, stn in enumerate(order):
        d = data[stn]
        ecl = eclipses[stn]
        for j, (key, _, _, label, unit) in enumerate(COLUMNS):
            ax = axes[i, j]
            ax.set_xlim(t0, t1)
            if limits[key]:
                ax.set_ylim(*limits[key])
            if d is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", color="#999999", fontsize=7)
            else:
                ev, base = d[key]
                if ecl:
                    ax.axvspan(ecl["start"], ecl["end"], color=SHADE_COLOR, alpha=0.22,
                               lw=0, zorder=0)
                    ax.axvline(ecl["peak"], color="#B07A00", lw=0.6, ls=":", zorder=1)
                plot_line(ax, base, color=BASE_COLOR, lw=1.1, ls="--", zorder=2)
                plot_line(ax, ev, color=EVENT_COLOR, lw=1.0, marker="o", ms=1.4,
                          mew=0, zorder=3)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            if i == 0:
                ax.set_title(titles.get(key, f"{label} ({unit})"), pad=4)
            if j == 0:
                ax.set_ylabel(STATION_LABEL[stn], fontsize=6.5, labelpad=3)
                if ecl:
                    ax.text(0.97, 0.06, r"$\mathcal{O}^p_{193}$=%.2f" % ecl["obsc"],
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.text(0.03, 0.93, f"({chr(97 + i * 4 + j)})", transform=ax.transAxes,
                    fontsize=7, fontweight="bold", va="top", ha="left")

    for ax in axes[-1]:
        ax.set_xlabel("Time (UT, hours)")

    handles = [
        plt.Line2D([], [], color=EVENT_COLOR, lw=1.0, marker="o", ms=1.8,
                   label=f"{date:%d %b %Y} (eclipse day, smoothed)"),
        plt.Line2D([], [], color=BASE_COLOR, lw=1.1, ls="--", label=baseline_label),
        mpl.patches.Patch(color=SHADE_COLOR, alpha=0.35, label="Eclipse interval"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=1.8,
               columnspacing=1.8)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.06, top=0.925,
                        hspace=0.16, wspace=0.28)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{cfg['fig']}.{ext}"
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="both", choices=["2023", "2024", "both"])
    ap.add_argument("--raw", action="store_true", help="no running-median despiking")
    ap.add_argument("--no-interp", action="store_true", help="do not interpolate data gaps")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    if a.no_interp:
        MAX_INTERP_GAP = None
    for y in ([2023, 2024] if a.year == "both" else [int(a.year)]):
        make_figure(y, a.out, smooth=not a.raw)
