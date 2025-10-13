import sys
sys.path.extend(["py/", "py/2023/"])
import datetime as dt
import matplotlib.dates as mdates
import os
import glob
import pandas as pd

from fetch import create_local_folder, copy2local
import utils
from skymap_stack import create_skymaps_panels
from summary_param_satck import plot_summary_2023
from direc import create_direc_panels

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.parsers.sky import SkyExtractor
from pynasonde.digisonde.digi_plots import (
    SkySummaryPlots, SaoSummaryPlots, 
    SkySummaryPlots,
)

def copy(date, mode="SKYWAVE"):
    os.makedirs("figures/2023/", exist_ok=True)
    base = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/{mode}_DPS4D_{date.strftime('%Y_%m_%d')}/"
    local, remote = create_local_folder(base)
    copy2local(local, remote)
    return local, remote

def generate_digisonde_pfh_profiles(
    date, func_name, fig_title="", draw_local_time=False,
    mode="SKYWAVE",
):
    local, remote = copy(date, mode=mode)
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
    sao_plot.save(f"figures/2023/{mode}_{date.strftime('%Y%m%d')}_sao.png")
    sao_plot.close()
    return

def create_dvl_plots(date=dt.datetime(2023, 10, 14), mode="SKYWAVE"):
    local, remote = copy(date, mode=mode)
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

    dvlplot.save(f"figures/2023/{mode}_{date.strftime('%Y%m%d')}_dvl.png")
    dvlplot.close()
    return
    
def create_sky_maps(date, mode="SKYWAVE"):
    os.makedirs(f"figures/2023/sky/{date.strftime('%Y%m%d')}/", exist_ok=True)
    local, remote = copy(date, mode=mode)
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
        skyplot.save(f"figures/2023/sky/{mode}_{date.strftime('%Y%m%d')}/{extractor.date.strftime('%H%M')}_sky.png")
        skyplot.close()
        j+=1
    return

def plot_ionosonde_data(
    date=dt.datetime(2023, 10, 14), code="BC840"
):
    file = f"data/{date.year}/{code}.csv"
    o = pd.read_csv(file, parse_dates=["date"])
    dates, foF2_new = utils.interpolate_missing_values(o, "date", "foF2")
    print(o.head())
    p = SaoSummaryPlots(
        figsize=(6, 3), fig_title="", draw_local_time=False
    )
    ax = p.get_axes(del_ticks=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    ax.plot(o.date, o.foF2, marker="o", markersize=3, color="b", label="foF2")
    ax.plot(dates, foF2_new, marker="o", markersize=1, color="r", label="foF2-i")
    ax.set_xlabel("Time (UT)")
    ax.set_ylim([0, 12])
    ax.set_ylabel("foFs (MHz)")
    p.save(f"figures/2023/{code}_ts.png")
    p.close()
    return

if __name__ == "__main__":
    dates = [
        dt.datetime(2023, 10, 13),
        dt.datetime(2023, 10, 14),
        dt.datetime(2023, 10, 15)
    ]
    mode = "SKYWAVE"
    for date in dates:
        create_dvl_plots(date, mode=mode)
        # generate_digisonde_pfh_profiles(
        #     date,
        #     "height_profile",
        #     fig_title="",
        # )
        # create_sky_maps(date)
        pass

    # folder = "/home/chakras4/OneDrive/Chakras4/Projects/ERAU.SAIL.Projects/byProjects/APEP/Downloaded Datastes/2023/AU930/"
    # utils.extract_all_ionosonde_datasets_from_inogram_image(folder, date=dt.datetime(2023, 10, 14), code="AU930")
    # plot_ionosonde_data(code="AU930")
    # copy(dt.datetime(2023, 10, 14))
    # copy(dt.datetime(2023, 10, 14), mode="WSMR")
    create_skymaps_panels(
        [
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287143315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287145715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287145715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287150915.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287152115.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287153315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287154515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287155715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287163315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287164515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287170915.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287172115.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287173315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287174515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287175715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287180915.SKY",
            #"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287182115.SKY",
            #"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287182115.SKY",
        ], 
        nrows=4, ncols=4,
        fig_title="14 Oct 2023 / KR835", 
        fname="figures/2023/sky_stack_KR835.png"
    )

    create_skymaps_panels(
        [
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287143315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287145715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287145715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287150915.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287152115.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287153315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287154515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287155715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287163315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287164515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287170915.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287172115.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287173315.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287174515.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287175715.SKY",
            "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/WS833_2023287180915.SKY",
            #"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287182115.SKY",
            #"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287182115.SKY",
        ], 
        nrows=4, ncols=4,
        fig_title="14 Oct 2023 / WS833", 
        fname="figures/2023/sky_stack_WS833.png"
    )
    # plot_summary_2023()
    # create_direc_panels(
    #     regex="/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/*.RSF",
    #     fname="figures/2023/direc_stack_WS833.png",
    # )
