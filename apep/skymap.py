
from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.parsers.sky import SkyExtractor

from utils import get_eclipse_contours_pyEclipse, get_fov_pyEclipse
import numpy as np
from pysolar import solar
from datetime import timezone

import numpy as np

from geopy.distance import distance as geo_distance
from geopy.point import Point

from tec import read_tec_observations_glonas

def latlon_from_xy_geopy(
    lat0_deg: float,
    lon0_deg: float,
    x_east_km,
    y_north_km,
):
    """
    Convert local ENU offsets (x_east_km, y_north_km) from (lat0, lon0)
    into destination lat/lon using geopy's great-circle 'destination'.

    Parameters
    ----------
    lat0_deg, lon0_deg : float
        Starting latitude/longitude in degrees.
    x_east_km, y_north_km : array_like or float
        East and North offsets in kilometers (same shape or broadcastable).

    Returns
    -------
    lat_out, lon_out : ndarray
        Lat/lon arrays matching the broadcasted shape of x/y.
        NaNs in x/y propagate to outputs.
    """
    x = np.asarray(x_east_km, dtype=float)
    y = np.asarray(y_north_km, dtype=float)
    # Broadcast
    x, y = np.broadcast_arrays(x, y)

    # Distance and bearing (bearing = CW from North)
    dist_km = np.hypot(x, y)
    bearing_deg = np.degrees(np.arctan2(x, y))  # atan2(East, North)

    # Flatten for simple geopy loop
    dist_flat = dist_km.ravel()
    bear_flat = bearing_deg.ravel()

    lat_flat = np.empty_like(dist_flat)
    lon_flat = np.empty_like(dist_flat)

    start = Point(lat0_deg, lon0_deg)

    for i, (d, brg) in enumerate(zip(dist_flat, bear_flat)):
        if np.isnan(d) or np.isnan(brg):
            lat_flat[i] = np.nan
            lon_flat[i] = np.nan
        else:
            dest = geo_distance(kilometers=float(d)).destination(start, float(brg))
            lat_flat[i] = dest.latitude
            lon_flat[i] = dest.longitude

    # Reshape and normalize lon to [-180, 180)
    lat_out = lat_flat.reshape(dist_km.shape)
    lon_out = ((lon_flat.reshape(dist_km.shape) + 180.0) % 360.0) - 180.0
    return lat_out, lon_out

def xy_from_zenith_az_height_flat(az_deg, zen_deg, h_km):
    """
    Flat ground plane (z=0). Returns local East (x), North (y) distances in km.

    Parameters
    ----------
    az_deg : array_like
        Azimuth clockwise from North, degrees.
    zen_deg : array_like
        Zenith angle (0 = vertical, 90 = horizon), degrees.
    h_km : float or array_like
        Sensor height above ground in km.

    Returns
    -------
    x_east_km, y_north_km, d_km : ndarray
        Ground intercept eastward offset, northward offset, and ground range (km).
        NaN where the ray does not intersect the ground in forward direction.
    """
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    zen = np.deg2rad(np.asarray(zen_deg, dtype=float))
    h  = np.asarray(h_km, dtype=float)

    # horizontal ground range: d = h / tan(zen)
    d_km = h / np.tan(zen)
    # only valid if zen > 90° (pointing downward); for zen < 90° it's above horizon
    d_km = np.where(zen > np.pi/2, np.nan, d_km)

    x_east_km  = d_km * np.sin(az)
    y_north_km = d_km * np.cos(az)
    return x_east_km, y_north_km, d_km

def create_skymaps_panels(
    files,
    nrows=2,
    ncols=2,
    font_size=15,
    figsize=(4, 4),
    fig_title="",
    fname="",
    date=None,
    date_lims=[],
    wl="of", 
    file_ext="_150km_193_1.nc",
    data_folder="data/2023/mask/",
):
    skyplot = SkySummaryPlots(
        fig_title="",
        nrows=nrows,
        ncols=ncols,
        font_size=font_size,
        figsize=figsize,
        date=date,
        date_lims=date_lims,
        subplot_kw=dict(projection="polar"),
        draw_local_time=False,
    )
    for i, f in enumerate(files):
        extractor = SkyExtractor(f, True, True,)
        extractor.extract()
        df = extractor.to_pandas()
        # get solar zenith angle and also eclipse
        p = 1-get_eclipse_contours_pyEclipse(
            [extractor.date], extractor.stn_info["LAT"],
            extractor.stn_info["LONG"],
            wl=wl, 
            file_ext=file_ext,
            data_folder=data_folder,
        )
        sza = solar.get_altitude(extractor.stn_info["LAT"],extractor.stn_info["LONG"], extractor.date.replace(tzinfo=timezone.utc))
        text=f"{extractor.date.strftime('%H:%M:%S UT')}\n" + r"$\mathcal{O}$=%.2f/"%p[0] + r"$\chi=%.1f^{\circ}$"%sza
        x_east_km, y_north_km = np.meshgrid(np.linspace(-10, 10.1, 300), np.linspace(-10, 10.1, 300))
        h = 200 * np.ones_like(x_east_km)
        lats, lons = latlon_from_xy_geopy(
            extractor.stn_info["LAT"],extractor.stn_info["LONG"],
            x_east_km, y_north_km
        )
        r, theta = np.sqrt(x_east_km**2+y_north_km**2), -np.arctan2(y_north_km, x_east_km)
        p = 1- get_fov_pyEclipse(
            [extractor.date], 
            lats, 
            lons,
            wl=wl, 
            file_ext=file_ext,
            data_folder=data_folder,
        )[0, :, :]
        skyplot.plot_skymap(
            df,
            zparam="spect_dop_freq",
            text=text,
            cmap="Spectral",
            clim=[-0.25, 0.25],
            rlim=6,
            cbar=i==len(files)-1,
        )
        ax = skyplot.axes[skyplot.n_sub_plots-1]
        ax.pcolormesh(theta, r, p, cmap="gray_r", vmax=1, vmin=0, alpha=0.4)
        ax.set_rmax(6)
        
    ax = skyplot.fig.get_axes()[0]
    ax.text(-0.1, 0.99, fig_title, ha="left", va="top", transform=ax.transAxes, rotation=90)
    skyplot.save(fname)
    skyplot.close()
    return


def create_skymap_overlay_tec(
    files,
    nrows=2,
    ncols=2,
    font_size=15,
    figsize=(4, 4),
    fig_title="",
    fname="",
    date=None,
    date_lims=[],
):
    skyplot = SkySummaryPlots(
        fig_title="",
        nrows=nrows,
        ncols=ncols,
        font_size=font_size,
        figsize=figsize,
        date=date,
        date_lims=date_lims,
        draw_local_time=False,
        subplot_kw=dict(projection=None),
    )

    for i, f in enumerate(files):
        extractor = SkyExtractor(f, True, True,)
        extractor.extract()
        df = extractor.to_pandas()

        p = create_eclipse_path_local([extractor.date], extractor.stn_info["LAT"],extractor.stn_info["LONG"])
        sza = solar.get_altitude(extractor.stn_info["LAT"],extractor.stn_info["LONG"], extractor.date.replace(tzinfo=timezone.utc))
        text=f"{extractor.date.strftime('%H:%M:%S UT')}\n" + r"$\mathcal{O}$=%.2f/"%p[0] + r"$\chi=%.1f^{\circ}$"%sza
        
        # x_east_km, y_north_km = np.meshgrid(df.x_coord, df.y_coord)
        lats, lons = latlon_from_xy_geopy(
            extractor.stn_info["LAT"],extractor.stn_info["LONG"],
            df.x_coord, df.y_coord
        )
        ax = skyplot.fig.get_axes()[skyplot.n_sub_plots-1]
        im = ax.scatter(
            lons, 
            lats, 
            c=df.spect_dop_freq,
            cmap="Spectral",
            s=1.5,
            marker="D",
            zorder=2,
            vmax=1,
            vmin=-1,
        )
        ax.axvline(extractor.stn_info["LONG"], ls="--", lw=0.8, color="k")
        ax.axhline(extractor.stn_info["LAT"], ls="--", lw=0.8, color="k")
        
        dtec = read_tec_observations_glonas(extractor.date)
        ax.scatter(
            dtec.ipplons, 
            dtec.ipplats, 
            c=dtec.dtec1,
            cmap="Blues",
            s=1.5,
            marker="D",
            zorder=2,
            alpha=0.5,
            vmax=0.1,
            vmin=-0.1,
        )
        ax.text(0.05, 0.95, text, ha="left", va="center", transform=ax.transAxes)

        # ax.set_xlim(extractor.stn_info["LONG"]+.05, extractor.stn_info["LONG"]-.05)
        # ax.set_ylim(extractor.stn_info["LAT"]-0.05, extractor.stn_info["LAT"]+0.05)
        if i==len(files)-1:
            skyplot._add_colorbar(im, skyplot.fig, ax, "Doppler, Hz", [0.05, 0.0125, 0.015, 0.5])
    ax = skyplot.fig.get_axes()[0]
    ax.text(-0.3, 0.99, fig_title, ha="left", va="top", transform=ax.transAxes, rotation=90)
    skyplot.save(fname)
    skyplot.close()
    return
