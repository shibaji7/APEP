import cartopy
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime as dt

import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from descartes import PolygonPatch
from matplotlib.projections import register_projection
from shapely.geometry import LineString, MultiLineString, Polygon, mapping


def setsize(size=6):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots

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
    return

def date_string(date, label_style="web"):
    # Set the date and time formats
    dfmt = "%d %b %Y" if label_style == "web" else "%d %b %Y,"
    tfmt = "%H:%M"
    stime = date
    date_str = "{:{dd} {tt}} UT".format(stime, dd=dfmt, tt=tfmt)
    return date_str


class CartoBase(GeoAxes):
    name = "CartoBase"

    def __init__(self, *args, **kwargs):
        if "map_projection" in kwargs:
            map_projection = kwargs.pop("map_projection")
        else:
            map_projection = cartopy.crs.NorthPolarStereo()
            print(
                "map_projection keyword not set, setting it to cartopy.crs.NorthPolarStereo()"
            )
        # first check if datetime keyword is given!
        # it should be since we need it for aacgm
        if "plot_date" in kwargs:
            self.plot_date = kwargs.pop("plot_date")
        else:
            raise TypeError(
                "need to provide a date using 'plot_date' keyword for aacgmv2 plotting"
            )
        # Now work with the coords!
        supported_coords = ["geo", "aacgmv2", "aacgmv2_mlt"]
        if "coords" in kwargs:
            self.coords = kwargs.pop("coords")
            if self.coords not in supported_coords:
                err_str = "coordinates not supported, choose from : "
                for _n, _sc in enumerate(supported_coords):
                    if _n + 1 != len(supported_coords):
                        err_str += _sc + ", "
                    else:
                        err_str += _sc
                raise TypeError(err_str)
        else:
            self.coords = "geo"
            print("coords keyword not set, setting it to aacgmv2")
        # finally, initialize te GeoAxes object
        super().__init__(map_projection=map_projection, *args, **kwargs)
        return

    def overaly_coast_lakes(self, resolution="50m", color="black", **kwargs):
        """
        Overlay AACGM coastlines and lakes
        """
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        # overaly coastlines
        feature = cartopy.feature.NaturalEarthFeature(
            "physical", "land", resolution, edgecolor="face", facecolor="lightgray"
        )
        self.add_feature(cartopy.feature.COASTLINE, **kwargs)
        # self.add_feature(feature)
        # ax.coastlines(resolution=resolution)

    def add_feature(self, feature, **kwargs):
        # Now we"ll set facecolor as None because aacgm doesn"t close
        # continents near equator and it turns into a problem
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        if "facecolor" in kwargs:
            print(
                "manually setting facecolor keyword to none as aacgm fails for fill! want to know why?? think about equator!"
            )
        kwargs["facecolor"] = "none"
        if self.coords == "geo":
            super().add_feature(feature, **kwargs)
        else:
            aacgm_geom = self.get_aacgm_geom(feature)
            aacgm_feature = cartopy.feature.ShapelyFeature(
                aacgm_geom, cartopy.crs.Geodetic(), **kwargs
            )
            super().add_feature(aacgm_feature, **kwargs)

    def mark_latitudes(self, lat_arr, lon_location=-110, **kwargs):
        """
        mark the latitudes
        Write down the latitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        """
        if isinstance(lat_arr, list):
            lat_arr = np.array(lat_arr)
        else:
            if not isinstance(lat_arr, np.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        # make an array of lon_location
        lon_location_arr = np.full(lat_arr.shape, lon_location)
        proj_xyz = self.projection.transform_points(
            cartopy.crs.PlateCarree(), lon_location_arr, lat_arr
        )
        # plot the lats now!
        out_extent_lats = False
        for _np, _pro in enumerate(proj_xyz[..., :2].tolist()):
            # check if lats are out of extent! if so ignore them
            lat_lim = self.get_extent(crs=cartopy.crs.Geodetic())[2::]
            if (lat_arr[_np] >= min(lat_lim)) and (lat_arr[_np] <= max(lat_lim)):
                self.text(
                    _pro[0],
                    _pro[1],
                    r"$%s^{\circ}$" % str(lat_arr[_np]),
                    **kwargs,
                    alpha=0.5,
                )
            else:
                out_extent_lats = True
        if out_extent_lats:
            print("some lats were out of extent ignored them")

    def mark_longitudes(self, lon_arr=np.arange(-180, 180, 60), lat_location=20,**kwargs):
        """
        mark the longitudes
        Write down the longitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        This is also trickier compared to latitudes!
        """
        if isinstance(lon_arr, list):
            lon_arr = np.array(lon_arr)
        else:
            if not isinstance(lon_arr, np.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        # make an array of lon_location
        lat_location_arr = np.full(lon_arr.shape, lat_location)
        proj_xyz = self.projection.transform_points(
            cartopy.crs.PlateCarree(), lon_arr, lat_location_arr
        )
        # plot the lats now!
        out_extent_lats = False
        for _np, _pro in enumerate(proj_xyz[..., :2].tolist()):
            # check if lats are out of extent! if so ignore them
            lon_lim = self.get_extent(crs=cartopy.crs.Geodetic())[::2]
            if (lon_arr[_np] >= min(lon_lim)) and (lon_arr[_np] <= max(lon_lim)):
                self.text(
                    _pro[0],
                    _pro[1],
                    r"$%s^{\circ}$" % str(lon_arr[_np]),
                    **kwargs,
                    alpha=0.5,
                )
            else:
                out_extent_lats = True
        if out_extent_lats:
            print("some lats were out of extent ignored them")


register_projection(CartoBase)


def create_map(
    central_longitude=-120,
    central_latitude=30,
    date=dt.datetime.now(),
    extent=[-150, -70, 10, 70],
    fname="figures/map.png",
):
    import sys
    sys.path.append("py/")
    import utils

    fig = plt.figure(figsize=(2*3.5, 3.5), dpi=300)
    proj = cartopy.crs.Stereographic(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )
    # this creats a 'geoaxes' object and sets the projection to a cool looking orthographic projection
    ax = fig.add_subplot(
        122,
        projection="CartoBase",
        map_projection=proj,
        coords="geo",
        plot_date=date,
    )
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

    track = pd.read_csv("data/2024/trajectory_150km.csv")
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        track["lon"].values,
        track["lat"].values,
    )
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        color="red",
        lw=0.5, ls="--"
    )
    # Overlay 2024 Eclipse
    lat, lon, Of = utils.get_eclipse_of_pyEclipse(
        [dt.datetime(2024, 4, 8, 15) + dt.timedelta(minutes=5*d) for d in range(82)],
        wl="193", 
        data_folder="data/2024/mask/",
    )
    Of[Of <= 0] = np.nan
    Of[Of >= 1] = np.nan
    p = np.nanmin(Of, axis=0)
    p = 1-p
    Lon, Lat = np.meshgrid(lon, lat)
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        Lon, Lat
    )
    CS = ax.contour(
        xyz[:, :, 0],
        xyz[:, :, 1],
        p,
        levels=np.arange(0., 1.1, 0.1),
        colors="k", linewidths=0.5
    )
    CS.clabel(fmt="%1.1f", fontsize=8)
    cf = ax.contourf(
        xyz[:, :, 0],
        xyz[:, :, 1],
        p,
        levels=np.arange(0., 1.1, 0.1),
        cmap="Blues",
        transform=proj,
        extend="max",
    )
    cax = fig.add_axes([0.92, 0.3, 0.01, 0.4]) 
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(r"Obscuration ($\mathcal{O}_{193}$)")
    
    names = ["AL945", "AU930", "BC840", "KR835", "WS833", "WP937", "WI937"]
    stations_lats = np.array([45.07, 30.40, 40.00, 35.00, 33.72, 37.90, 37.90])
    stations_lons = np.array([276.44, 262.30, 254.70, 253.47, 253.26, 284.50, 284.50])
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        stations_lons, stations_lats
    )
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        marker="^",
        color="m",
        s=5,
    )
    xoffset = [-1, 1, -1, 1, 1, 1, -1]
    yoffset = [1, -1, 1, -1, -2, -1, 1]
    j = 0
    for lat, lon, name in zip(stations_lats, stations_lons, names):
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            np.array([lon]), np.array([lat])
        )
        ax.text(
            lon+xoffset[j],
            lat+yoffset[j],
            name,
            transform=cartopy.crs.PlateCarree(),
            fontsize=4,
            color="k",
        )
        j+=1
    ax.text(0.05, 1.05, "(B) 8 April, 2024 GAE", ha="left", va="center", transform=ax.transAxes)

    # this creats a 'geoaxes' object and sets the projection to a cool looking orthographic projection
    ax = fig.add_subplot(
        121,
        projection="CartoBase",
        map_projection=proj,
        coords="geo",
        plot_date=date,
    )
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

    
    times = [dt.datetime(2023, 10, 14, 14) + dt.timedelta(minutes=5*d) for d in range(72)]
    lat, lon, Of = utils.get_eclipse_of_pyEclipse(
        times,
        wl="of", 
        file_ext="_150km_193_1.nc",
        data_folder="data/2023/mask/"
    )
    Of[Of <= 0] = np.nan
    Of[Of >= 1] = np.nan
    p = np.nanmin(Of, axis=0)
    p = 1-p
    Lon, Lat = np.meshgrid(lon, lat)

    track = pd.read_csv("data/2023/central_line_centers.csv")
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        track.CenterLongitudeDeg, track.CenterLatitudeDeg
    )
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        color="red",
        lw=0.5, ls="--"
    )
    
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        Lon, Lat
    )
    CS = ax.contour(
        xyz[:, :, 0],
        xyz[:, :, 1],
        p,
        levels=np.arange(0., 1.1, 0.1),
        colors="k", linewidths=0.5
    )
    CS.clabel(fmt="%1.1f", fontsize=8)
    cf = ax.contourf(
        xyz[:, :, 0],
        xyz[:, :, 1],
        p,
        levels=np.arange(0., 1.1, 0.1),
        cmap="Blues",
        transform=proj,
        extend="max",
    )

    
    names = ["AL945", "AU930", "BC840", "KR835", "WS833"]
    stations_lats = np.array([45.07, 30.40, 40.00, 35.00, 33.72])
    stations_lons = np.array([276.44, 262.30, 254.70, 253.47, 253.26])
    xyz = ax.projection.transform_points(
        cartopy.crs.PlateCarree(),
        stations_lons, stations_lats
    )
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        marker="^",
        color="m",
        s=5,
    )
    j=0
    for lat, lon, name in zip(stations_lats, stations_lons, names):
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            np.array([lon]), np.array([lat])
        )
        ax.text(
            lon+xoffset[j],
            lat+yoffset[j],
            name,
            transform=cartopy.crs.PlateCarree(),
            fontsize=4,
            color="k",
        )
        j+=1
    ax.text(-0.05, 0.99, "Coords: Geo", ha="left", va="top", rotation=90, transform=ax.transAxes)
    ax.text(0.05, 1.05, "(A) 14 October, 2023 GAE", ha="left", va="center", transform=ax.transAxes)

    fig.subplots_adjust(
        # left=0.05,
        # right=0.92,
        # top=0.95,
        # bottom=0.05,
        wspace=0.1,
        hspace=0.1,
    )
    plt.savefig(
        fname,
        dpi=1000,
        bbox_inches="tight",
    )
    plt.savefig(
        "manuscript_figures/Figure01.png",
        dpi=1000,
        bbox_inches="tight",
    )
    plt.savefig(
        "manuscript_figures/pdfs/Figure01.pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    plt.close()
    

if __name__ == "__main__":
    create_map()