"""
JGR Figure 04: ionosonde response during the 08 Apr 2024 eclipse (5 x 4 time series).

Data: DIDBase/SAO scaled values (data/2024/*_SAO.XML for AU930, BC840, AL945, KR835; WSMR SAO
files for WS833). Quiet-day baseline = Datasets/Data/<STN>_<YEAR>_median.csv; where a column is missing there
(e.g. hmF2/hmE at KR835/WS833, or 2024 KR835/WS833) it falls back to the time-of-day mean of the SAO files.
Plot style is shared with jgr_timeseries_5x4.py.
"""

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.extend([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])])

from apep.jgr.jgr_timeseries_5x4 import (
    OUT_DIR, STATIONS, draw_figure, find_up, frames_to_station_data, median_baseline, split_event_background,
)

YEAR = 2024
TIME_LIMITS = (dt.datetime(2024, 4, 8, 16, 0), dt.datetime(2024, 4, 8, 21, 0))
WS833_ROOT = "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/"
WS833_FOLDERS = [f"{WS833_ROOT}WSMR_DPS4D_2024_04_{d}/" for d in ("07", "08", "09")]
TITLES = {"foF2": r"$f_oF_2$ (MHz)", "hF2": r"$hmF_2$ (km)", "foE": r"$f_oE$ (MHz)", "hE": r"$hmE$ (km)"}


def load_bgc(stn):
    from fetch import get_bgc_iono_data

    return get_bgc_iono_data(TIME_LIMITS[0].date(), TIME_LIMITS, stn)


def load_kr835():
    from pynasonde.digisonde.parsers.sao import SaoExtractor

    df = SaoExtractor.extract_SAO(
        "data/2024/KR835_20240407(098)_SAO.XML", func_name="scaled",
        extract_time_from_name=False, extract_stn_from_name=False,
    )
    df["datetime"] = pd.to_datetime(
        df["datetime"].str.replace(r"\s+-\d+\s+", " ", regex=True), errors="coerce"
    )
    return split_event_background(df, TIME_LIMITS)


def load_ws833():
    from pynasonde.digisonde.parsers.sao import SaoExtractor

    folders = [f for f in WS833_FOLDERS if any(Path(f).glob("*.SAO"))]
    if not folders:
        print("Warning: WS833 SAO folders not found")
        return None
    df = SaoExtractor.load_SAO_files(folders=folders, func_name="scaled", n_procs=4)
    return split_event_background(df, TIME_LIMITS)


LOADERS = {"KR835": load_kr835, "WS833": load_ws833}


def create_figure_04(output_dir=OUT_DIR, smooth=True):
    os.chdir(find_up("data").parent)  # data/<year>/ paths are cwd-relative
    data = {}
    for stn in STATIONS:
        try:
            res = LOADERS.get(stn, lambda s=stn: load_bgc(s))()
        except Exception as exc:
            print(f"Warning: could not load {stn}: {exc}")
            res = None
        data[stn] = None if res is None else frames_to_station_data(*res, smooth=smooth, median_bg=median_baseline(stn, YEAR))
    draw_figure(YEAR, data, output_dir, titles=TITLES, baseline_label="Quiet-day median")


if __name__ == "__main__":
    create_figure_04()
