from xarray import open_dataset
import datetime as dt
import sys
import scipy.io as io
import h5py as hf


from pathlib import Path
import sys

sys.path.extend(
    [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[2]),
    ]
)

from geometry import *
import utils
import matplotlib as mpl

size = 30
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


def read_datasets(dates):
    dtec = []
    for d in dates:
        ds = open_dataset(f"data/{d.year}/{d.strftime('%Y%m%d%H')}_dtec.nc")
        i = int(d.minute/5)
        dtec.append(dict(
            time=d,
            data=ds.dtec.values[:, :, i],
            lat=ds.lat.values,
            lon=ds.lon.values,
        ))
        ds.close()
    return dtec

def read_tec_observations_glonas(
    date,
    fname="data/2024/fulltimedata_04_08_2024_32to36lat_-104to108lon.mat"
):
    timedt,ipplats,ipplons,dtec1,dtec2,el,az = read_tec_azimuth_fulltime(fname)

    timedt = pd.to_datetime(timedt).tolist()
    index = np.argmin([np.abs((date-d).total_seconds())for d in timedt])
    df = pd.DataFrame()
    (
        df["ipplats"], 
        df["ipplons"], 
        df["dtec1"], 
        df["dtec2"], 
        df["el"], 
        df["az"],
    ) = (
        ipplats[index], 
        ipplons[index], 
        dtec1[index], 
        dtec2[index], 
        el[index],
        az[index],
    )
    return df


def read_tec_azimuth_fulltime(ftec):
    alltec=hf.File(ftec,"r")
    timestamps=alltec["UTT"]
    times=np.concatenate(timestamps,axis=0)
    #######
    timedt=np.array([])
    ipplats=np.ones(len(times)-1,dtype=object)*np.NaN
    ipplons=np.ones(len(times)-1,dtype=object)*np.NaN
    dtec1=np.ones(len(times)-1,dtype=object)*np.NaN
    dtec2=np.ones(len(times)-1,dtype=object)*np.NaN
    el=np.ones(len(times)-1,dtype=object)*np.NaN
    az=np.ones(len(times)-1,dtype=object)*np.NaN
    arrs=alltec["fulltimedata"]
    for tidx,ts in enumerate(times[:-1]):
        tsdt=dt.datetime.fromordinal(int(ts)) + dt.timedelta(days=ts%1)-dt.timedelta(days=366)
        ###
        ref=arrs[tidx][0]
        dset=np.array(alltec[ref])
        try:
            nentries,nstat=np.shape(dset)
        except:
            continue
        ###
        ipplat=dset[3]
        ipplon=dset[4]
        filt1=dset[5]
        filt2=dset[6]
        elev=dset[1]
        azm=dset[2]
        # filt3=dset[6]
        ###
        timedt=np.append(timedt,tsdt)
        ipplats[tidx]=ipplat
        ipplons[tidx]=ipplon
        dtec1[tidx]=filt1
        dtec2[tidx]=filt2
        el[tidx]=elev
        az[tidx]=azm
    return timedt,ipplats,ipplons,dtec1,dtec2,el,az


def create_2024_tec(
    central_longitude=-120,
    central_latitude=30,
    extent=[-120, -80, 30, 50],
    # extent=[-115, -100, 25, 40],
):
    # setsize(30)
    fig = plt.figure(figsize=(9*3, 9*2.5), dpi=300)
    proj = cartopy.crs.Stereographic(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )

    dates = [dt.datetime(2024, 4, 8, 17, 30) + dt.timedelta(minutes=20*i) for i in range(9)]
    datas = read_datasets(dates)
    for i, d in enumerate(dates):
        print(d)
        ax = fig.add_subplot(
            331+i,
            projection="CartoBase",
            map_projection=proj,
            coords="geo",
            plot_date=d
        )
        data = datas[i]

        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        ax.overaly_coast_lakes(lw=0.4, alpha=1)
        mark_lons = np.arange(extent[0], extent[1]+1, 20)
        plt_lats = np.arange(extent[2], extent[3]+1, 15)
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.2)
        gl.xlocator = mticker.FixedLocator(mark_lons)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.n_steps = 90
        ax.mark_latitudes(plt_lats, fontsize="xx-small", color="k")
        ax.mark_longitudes(mark_lons, fontsize="xx-small", color="k")
        
        Lat, Lon = np.meshgrid(data["lat"], data["lon"])
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            Lon, Lat
        )
        im = ax.pcolor(
            xyz[:, :, 0],
            xyz[:, :, 1],
            data["data"].T,
            cmap="Spectral",
            vmax=1, vmin=-1,
            transform=proj,
            shading='auto',
            zorder=3,
            alpha=0.6,
        )
        if i==0:
            ax.text(-0.05, 0.99, "Coords: Geo", ha="left", va="top", rotation=90, transform=ax.transAxes)
            ax.text(0.05, 1.05, "8 April, 2024 GAE", ha="left", va="center", transform=ax.transAxes)
        if i==len(dates)-1:
            pos = ax.get_position()
            mpos = [0.025, 0.0125, 0.015, 0.5]
            cpos = [
                pos.x1 + mpos[0],
                pos.y0 + mpos[1],
                mpos[2],
                pos.height * mpos[3],
            ]
            cax = fig.add_axes(cpos)
            cb = fig.colorbar(im, ax=ax, cax=cax)
            cb.set_label(r"dTEC, TECu")
        ax.text(0.05, 0.05, f"({chr(65+i)}) {d.strftime('%H%M UT')}", ha="left", va="center", transform=ax.transAxes)


        Lat, Lon = np.meshgrid(np.arange(0, 90, 0.5), np.arange(-160, 0, 0.5))
        p = utils.get_fov_eclipse(
            [d], Lat, Lon, limit=None
        )
        p = np.nanmax(p, axis=0)
        p[p<=0] = np.nan
        p[p>1] = np.nan
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            Lon, Lat
        )
        cf = ax.contour(
            xyz[:, :, 0],
            xyz[:, :, 1],
            p,
            levels=np.arange(0., 1.01, 0.25),
            transform=proj,
            zorder=1,
            cmap="gray"
        )
        ax.clabel(cf, inline=True, fontsize=20, fmt='%.2f', colors="k", zorder=3)
        # if i==2:
        #     pos = ax.get_position()
        #     mpos = [0.025, 0.0125, 0.015, 0.5]
        #     cpos = [
        #         pos.x1 + mpos[0],
        #         pos.y0 + mpos[1],
        #         mpos[2],
        #         pos.height * mpos[3],
        #     ]
        #     cax = fig.add_axes(cpos)
        #     cb = fig.colorbar(cf, ax=ax, cax=cax)
        #     cb.set_label(r"Obscuration ($\mathcal{O}$)")

    plt.savefig(
        "figures/2024/dtec.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    return

if __name__ == "__main__":
    # read_tec_observations_glonas(dt.datetime(2024,4,8,16))
    create_2024_tec()