"""
JGR Figure 02: ionosonde response during the 14 Oct 2023 eclipse (5 x 4 time series, 13-21 UT).

foF2 and foE: Datasets/Data/<STN>_2023_scaled.csv (event) and <STN>_2023_median.csv (baseline).
hmF2 and hmE: DIDBase/SAO scaled values (data/2023/*_SAO.XML; WSMR SAO files for WS833). Their
baseline is the median CSV where populated, else the time-of-day mean of the SAO files.
Plot style is shared with jgr_timeseries_5x4.py.
"""

import datetime as dt
import os

import pandas as pd
import sys
from pathlib import Path

sys.path.extend([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])])

from apep.jgr.jgr_timeseries_5x4 import (
    OUT_DIR, STATIONS, draw_figure, find_up, frames_to_station_data, load_station, median_baseline,
    split_event_background,
)

YEAR = 2023
TIME_LIMITS = (dt.datetime(2023, 10, 14, 13, 0), dt.datetime(2023, 10, 14, 21, 0))
WS833_ROOT = "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/"
WS833_FOLDERS = [f"{WS833_ROOT}WSMR_DPS4D_2023_10_{d}/" for d in (13, 14, 15)]
WS833_FALLBACK = [f"Datasets/Data/WSMR_Extracted/2023-10-{d}/" for d in (13, 14, 15)]
TITLES = {"foF2": r"$f_oF_2$ (MHz)", "hF2": r"$hmF_2$ (km)", "foE": r"$f_oE$ (MHz)", "hE": r"$hmE$ (km)"}


def load_bgc(stn):
    from fetch import get_bgc_iono_data

    result = get_bgc_iono_data(TIME_LIMITS[0].date(), TIME_LIMITS, stn)
    return None if result is None else result


def load_ws833():
    from pynasonde.digisonde.parsers.sao import SaoExtractor

    folders = [f for f in WS833_FOLDERS if any(Path(f).glob("*.SAO"))]
    if not folders:
        folders = [str(find_up(f)) + "/" for f in WS833_FALLBACK if find_up(f).exists()]
    if not folders:
        print("Warning: WS833 SAO folders not found")
        return None
    df = SaoExtractor.load_SAO_files(folders=folders, func_name="scaled", n_procs=4)
    return split_event_background(df, TIME_LIMITS)


def create_figure_02(output_dir=OUT_DIR, smooth=True):
    os.chdir(find_up("data").parent)  # fetch.get_bgc_iono_data uses cwd-relative data/<year>/
    data = {}
    for stn in STATIONS:
        try:
            res = load_ws833() if stn == "WS833" else load_bgc(stn)
        except Exception as exc:
            print(f"Warning: could not load {stn}: {exc}")
            res = None
        data[stn] = None if res is None else frames_to_station_data(*res, smooth=smooth, median_bg=median_baseline(stn, YEAR))
        csv = load_station(stn, YEAR, smooth)  # foF2 / foE event data from Datasets/Data/<STN>_2023_scaled.csv
        if csv is not None:
            data[stn] = data[stn] or {}
            empty = (pd.DataFrame({"t": [], "v": []}),) * 2
            for key in ("hF2", "hE"):
                data[stn].setdefault(key, empty)
            for key in ("foF2", "foE"):
                data[stn][key] = csv[key]
    draw_figure(YEAR, data, output_dir, titles=TITLES, baseline_label="Quiet-day median")


if __name__ == "__main__":
    create_figure_02()
