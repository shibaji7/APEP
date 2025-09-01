from xarray import open_dataset
import datetime as dt
import sys
sys.path.append("py/")

from geometry import *
import utils

def create_2023_tec(
    central_longitude=-120,
    central_latitude=30,
    extent=[-150, -70, 10, 70],
):
    setsize(30)
    fig = plt.figure(figsize=(9*3, 9*2.5), dpi=300)
    proj = cartopy.crs.Stereographic(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )

    dates = [dt.datetime(2023, 10, 14, 15) + dt.timedelta(minutes=20*i) for i in range(9)]
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
        ax.overaly_coast_lakes(lw=0.4, alpha=0.4)
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
            vmax=1, vmin=-2,
            transform=proj,
            shading='auto',
            zorder=3,
            alpha=0.6,
        )
        if i==0:
            ax.text(-0.05, 0.99, "Coords: Geo", ha="left", va="top", rotation=90, transform=ax.transAxes)
            ax.text(0.05, 1.05, "14 October, 2023 GAE", ha="left", va="center", transform=ax.transAxes)
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
        "figures/2023/dtec.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    return


def create_2024_tec(
    central_longitude=-120,
    central_latitude=30,
    extent=[-150, -70, 10, 70],
):
    setsize(30)
    fig = plt.figure(figsize=(9*3, 9*2.5), dpi=300)
    proj = cartopy.crs.Stereographic(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )

    dates = [dt.datetime(2024, 4, 8, 16, 30) + dt.timedelta(minutes=20*i) for i in range(9)]
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
        ax.overaly_coast_lakes(lw=0.4, alpha=0.4)
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

if __name__ == "__main__":
    create_2023_tec()
    create_2024_tec()