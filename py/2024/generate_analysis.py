import sys
sys.path.extend(["py/", "py/2024/"])
import datetime as dt
import matplotlib.dates as mdates
import os
import glob

from fetch import create_local_folder, copy2local
import utils

from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.parsers.sky import SkyExtractor
from pynasonde.digisonde.digi_plots import (
    SkySummaryPlots, SaoSummaryPlots, 
    SkySummaryPlots,
)

def copy(date):
    os.makedirs("figures/2024/", exist_ok=True)
    base = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_{date.strftime('%Y_%m_%d')}/"
    local, remote = create_local_folder(base)
    copy2local(local, remote)
    return local, remote

def generate_digisonde_pfh_profiles(
    date, func_name, fig_title="", draw_local_time=False
):
    local, remote = copy(date)
    df = SaoExtractor.load_SAO_files(
        folders=[local],
        func_name=func_name,
        n_procs=12,
    )
    df.ed = df.ed / 1e6
    sao_plot = SaoSummaryPlots(
        figsize=(6, 3), fig_title=fig_title, draw_local_time=draw_local_time
    )
    sao_plot.add_TS(
        df,
        zparam="ed",
        prange=[0, 1],
        zparam_lim=10,
        cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
        plot_type="scatter",
        scatter_ms=20,
    )
    time = df.datetime.unique()
    obs = utils.create_eclipse_path_local(time, df.lat.tolist()[0], df.lon.tolist()[0])
    ax = sao_plot.axes
    axt = ax.twinx()
    axt.plot(df.datetime.unique(), 1 - obs, ls="--", lw=0.9, color="k")
    axt.set_ylabel("Obscuration")
    axt.set_ylim(0, 1.2)
    axt.set_yticks([0, .5, 1.])
    axt.set_yticklabels([1, .5, 0])
    ax.set_xlim([date, date+dt.timedelta(1)])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    sao_plot.save(f"figures/2024/{date.strftime('%Y%m%d')}_sao.png")
    sao_plot.close()
    return

def create_dvl_plots(date=dt.datetime(2024, 10, 14)):
    local, remote = copy(date)
    dvl_df = DvlExtractor.load_DVL_files(
        [local],
        n_procs=12,
    )

    obs = utils.create_eclipse_path_local(
        dvl_df.datetime, dvl_df.lat.tolist()[0], dvl_df.lon.tolist()[0]
    )

    dvlplot = SkySummaryPlots.plot_dvl_drift_velocities(
        dvl_df, fname=None, draw_local_time=False, figsize=(5, 3)
    )
    ax = dvlplot.axes[0]
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_xlim([date, date + dt.timedelta(1)])
    
    
    ax = dvlplot.axes[1]
    axt = ax.twinx()
    axt.scatter(dvl_df.datetime, 0.5 * (dvl_df.Hb + dvl_df.Ht), marker="D", s=3, color="m")
    axt.set_ylabel("Virtual Height, km", fontdict={"color": "m"})
    axt.set_ylim(250, 500)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_xlim([date, date + dt.timedelta(1)])

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

    dvlplot.save(f"figures/2024/{date.strftime('%Y%m%d')}_dvl.png")
    dvlplot.close()
    return
    
def create_sky_maps(date):
    os.makedirs(f"figures/2024/sky/{date.strftime('%Y%m%d')}/", exist_ok=True)
    local, remote = copy(date)
    files = glob.glob(os.path.join(local, "*.SKY"))
    files.sort()
    j = 0
    for f in files:
        extractor = SkyExtractor(f, True, True,)
        extractor.extract()
        df = extractor.to_pandas()
        skyplot = SkySummaryPlots()
        skyplot.plot_skymap(
            df,
            zparam="spect_dop_freq",
            text=f"Skymap:\n {extractor.stn_code} / {extractor.date.strftime('%H:%M:%S UT, %d %b %Y')}",
            # cmap="jet",
            clim=[-1, 1],
            rlim=6,
        )
        skyplot.save(f"figures/2024/sky/{date.strftime('%Y%m%d')}/{extractor.date.strftime('%H%M')}_sky.png")
        skyplot.close()
        j+=1
    return


if __name__ == "__main__":
    dates = [
        # dt.datetime(2024, 4, 7),
        dt.datetime(2024, 4, 8),
        # dt.datetime(2024, 4, 9),
    ]
    for date in dates:
        create_dvl_plots(date)
        generate_digisonde_pfh_profiles(
            date,
            "height_profile",
            fig_title="",
        )
        create_sky_maps(date)