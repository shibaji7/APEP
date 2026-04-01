import shutil
import os
import glob
import numpy as np
from loguru import logger
import datetime as dt
import pandas as pd

def get_bgc_iono_data(date:dt.datetime, date_lim: list,station:str="AL945", func_name:str="scaled"):
    file = f"data/{date.strftime('%Y')}/{station.upper()}_*_SAO.XML"
    files = glob.glob(file)
    if len(files) == 0:
        logger.warning(f"No BGC iono data found for {station} on {date.strftime('%Y-%m-%d')}")
        return None
    files.sort()
    logger.info(f"Found {len(files)} BGC iono data files for {station} on {date.strftime('%Y-%m-%d')}")
    from pynasonde.digisonde.parsers.sao import SaoExtractor
    datasets = SaoExtractor.extract_SAO(
        files[0], func_name=func_name, 
        extract_time_from_name=False,
        extract_stn_from_name=False,
    )
    datasets["datetime"] = pd.to_datetime(
        datasets["datetime"].str.replace(r"\s+-\d+\s+", " ", regex=True), 
        errors="coerce"
    )
    hmE, foE = np.array(datasets["hmE"]), np.array(datasets["foE"])
    hmE[np.isnan(foE)] = np.nan
    datasets["hmE"] = hmE
    bgc = datasets[
        (datasets["datetime"] < pd.to_datetime(date_lim[0])) | 
        (datasets["datetime"] >= pd.to_datetime(date_lim[1]))
    ]
    datasets = datasets[
        (datasets["datetime"] >= pd.to_datetime(date_lim[0])) & 
        (datasets["datetime"] < pd.to_datetime(date_lim[1]))
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
    fixed_date = dt.date(date.year, date.month, date.day)
    days_offset = fixed_date - dt.date(1900, 1, 1)
    bgc.datetime += pd.to_timedelta(days_offset.days, unit="d")
    datasets = datasets[["datetime", "foF2", "hmF2", "foE", "hmE"]]
    return datasets, bgc

def create_local_folder(
    base:str="/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_13/",
    clean:bool=False
):
    local = base
    if clean: shutil.rmtree(local, ignore_errors=True)
    os.makedirs(
        local, 
        exist_ok=True
    )
    logger.info(f"Created local folder: {local}")
    remote = local.replace("/tmp/", "/media/")
    logger.info(f"Remote folder: {remote}")
    return local, remote

def copy2local(local:str, remote:str, ext: str="*", ):
    remote_files = glob.glob(os.path.join(remote, ext))
    for remote_file in remote_files:
        fname = remote_file.split("/")[-1]
        shutil.copy2(remote_file, os.path.join(local, fname))
    return

if __name__ == "__main__":
    # tmp = "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/"
    # local, remote = create_local_folder(tmp)
    # copy2local(local, remote)
    datasets = get_bgc_iono_data(dt.datetime(2023, 10, 14), "AU930")
    import sys
    sys.path.append("apep/")
    from pynasonde.matlab_lib.matlab_engine import CreateFig
    fig = CreateFig()
    fig.generate_scaled_TS_figure([dict(
        dataset=datasets,
        xlim=[dt.datetime(2023,10,14), dt.datetime(2023,10,15)],
        title_txt="(A) AU930 / Austin",
    )], fig_file_name="example_scaled_TS.png"
    )
    fig.close()