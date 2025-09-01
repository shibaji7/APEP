from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.digi_plots import SkySummaryPlots, SaoSummaryPlots
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt

from pynasonde.digisonde.digi_utils import setsize
import utils


def create_dvl_plots(date=dt.datetime(2023, 10, 14), mode="SKYWAVE"):
    local, remote = copy(date, mode=mode)
    dvl_df = DvlExtractor.load_DVL_files(
        [local],
        n_procs=12,
    )

    obs = utils.create_eclipse_path_local(
        dvl_df.datetime, dvl_df.lat.tolist()[0], dvl_df.lon.tolist()[0]
    )

    dvlplot = SkySummaryPlots(
        figsize=(5, 3),
        nrows=3,
        ncols=1,
        subplot_kw=None,
        draw_local_time=False,
    )


    yparams = ["Vx", "Vy", "Vz"]
    colors = ["r", "b", "k"]
    errors = ["Vx_err", "Vy_err", "Vz_err"]
    labels = ["$V_x$", "$V_y$", "$V_z$"],
    
    for yparam, color, error, label in zip(yparams, colors, labels):
        ax = dvlplot.axes[0]
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.set_xlim([date, date + dt.timedelta(1)])
        ax.scatter(dvl_df)
    
    # ax = dvlplot.axes[1]
    # axt = ax.twinx()
    # axt.scatter(dvl_df.datetime, 0.5 * (dvl_df.Hb + dvl_df.Ht), marker="D", s=3, color="m")
    # axt.set_ylabel("Virtual Height, km", fontdict={"color": "m"})
    # axt.set_ylim(250, 500)
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    # ax.set_xlim([date, date + dt.timedelta(1)])

    ax = dvlplot.axes[2]
    ax.set_ylim(-20, 20)
    ax = dvlplot.axes[2]
    axt = ax.twinx()
    axt.plot(dvl_df.datetime, 1 - obs, ls="--", lw=0.9, color="k")
    axt.set_ylabel("Obscuration")
    axt.set_ylim(0, 1.2)
    axt.set_yticks([0, .5, 1.])
    axt.set_yticklabels([1, .5, 0])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_xlim([date, date + dt.timedelta(1)])

    dvlplot.save(f"figures/{date.year}/{mode}_{date.strftime('%Y%m%d')}_dvl.png")
    dvlplot.close()
    
    

def plot_summary_2023(date=dt.datetime(2023, 10, 14)):

    digis = {
        "BC840": pd.read_csv(f"data/{date.year}/BC840.csv", parse_dates=["date"]),
        "AU930": pd.read_csv(f"data/{date.year}/AU930.csv", parse_dates=["date"]),
        "KR835": SaoExtractor.load_SAO_files(
            folders=["/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"],
            func_name="scaled",
            n_procs=12,
        ),
        "WS833": SaoExtractor.load_SAO_files(
            folders=["/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/"],
            func_name="scaled",
            n_procs=12,
        )
    }
    sao = SaoSummaryPlots(
        nrows = 2,
        ncols = 1,
        font_size = 15,
        figsize = (5, 3),
    )
    setsize(15)
    ax = sao.get_axes(del_ticks=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.xaxis.set_major_formatter(DateFormatter("%H"))
    ax.set_xlim([dt.datetime(2023, 10, 14, 13), dt.datetime(2023, 10, 14, 22)])
    ax.set_ylabel(r"$foF_2$, MHz")
    # ax.set_xticklabels([])
    dates, foF2_new = utils.interpolate_missing_values(digis["WS833"], "datetime", "foF2")
    ax.plot(dates, foF2_new, lw=0.8, color="r", label=r"$WS833_i$")
    dates, foF2_new = utils.interpolate_missing_values(digis["KR835"], "datetime", "foF2")
    ax.plot(dates, foF2_new, lw=0.8, color="b", label=r"$KR835_i$")
    ax.set_ylim(5, 15)
    ax.set_xlabel(r"Time, UT")
    dates, foF2_new = utils.interpolate_missing_values(digis["BC840"], "date", "foF2")
    ax.plot(dates, foF2_new, lw=0.8, color="k", label=r"$BC840_i$")
    dates, foF2_new = utils.interpolate_missing_values(digis["AU930"], "date", "foF2")
    ax.plot(dates, foF2_new, lw=0.8, color="m", label=r"$AU930_i$")
    ax.set_ylim(5, 15)
    # print(digis["KR835"].foEs.tolist())
    ax.legend(loc=2, fontsize=8)

    ax = sao.get_axes(del_ticks=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.xaxis.set_major_formatter(DateFormatter("%H"))
    ax.set_xlim([dt.datetime(2023, 10, 14, 13), dt.datetime(2023, 10, 14, 22)])
    ax.set_ylabel(r"$foE_{(s)}$, km")
    ax.set_xlabel(r"Time, UT")
    ax.plot(digis["WS833"].datetime, digis["WS833"].foEs, "ro", label=r"$WS833$", ms=0.8)
    ax.plot(digis["KR835"].datetime, digis["KR835"].foEs, "bo", label=r"$KR835$", ms=0.8)
    ax.set_ylim(2, 4)
    ax.legend(loc=2, fontsize=8)

    sao.save("figures/2023/summary.png")
    sao.close()
    return