from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.parsers.rsf import RsfExtractor

import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt

from pynasonde.digisonde.digi_utils import setsize
import utils
import glob

def create_direc_panels(
    regex,
    nrows=1,
    ncols=1,
    font_size=15,
    figsize=(4, 4),
    fig_title="",
    fname="",
    date=None,
    date_lims=[dt.datetime(2023, 10, 14, 14), dt.datetime(2023, 10, 14, 18)],
):
    files = glob.glob(regex)
    files.sort()
    sorted_files = [
        f
        for f in files if (
            dt.datetime.strptime(f.split("/")[-1].split("_")[-1].replace(".RSF", ""), "%Y%j%H%M%S") >= date_lims[0] and\
            dt.datetime.strptime(f.split("/")[-1].split("_")[-1].replace(".RSF", ""), "%Y%j%H%M%S") <= date_lims[1]
        )
    ]
    data = []
    for i, f in enumerate(sorted_files):
        rsf = RsfExtractor(f, True, True)
        if rsf.date >= date_lims[0] and rsf.date <= date_lims[1]:
            rsf.extract()
            df = rsf.to_pandas()
            df = df[df.height <= 600]
            data.append(df)
            if i==10: break
    sao = SaoSummaryPlots(
        nrows = nrows,
        ncols = ncols,
        font_size = font_size,
        figsize = figsize,
        fig_title=fig_title
    )
    setsize(font_size)
    ax = sao.get_axes(del_ticks=False)
    ax.yaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.yaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    ax.yaxis.set_major_formatter(DateFormatter("%H"))
    ax.set_ylim(date_lims)
    ax.set_ylabel(r"Time, UT")
    ax.set_xlabel(r"Height, km")
    ax.set_xlim([-600, 600])
    sao.save(fname)
    sao.close()
    return