from pathlib import Path
import sys
sys.path.extend([
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2]),
])
import pandas as pd

import xarray as xr
import numpy as np
from pynasonde.vipir.ngi.plotlib import Ionogram

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from loguru import logger
import datetime as dt

class ScalingProcess:

    def __init__(
        self, 
        file_path,
        id_key="id_key",
        filter_params=dict(
            noise_level=10,
            range_top = 800,
            range_bot = 50,
            freq_top=15,
            freq_bot=2,
        ),
        scaling_params=dict(
            freq_axis_shrink=5,
            range_axis_shrink=1,
            eps=2, min_samples=5,
        )
    ):
        self.file_path = file_path
        self.filter_params = filter_params
        self.scaling_params = scaling_params
        self.load_ionosonde_xr(id_key)
        return

    def load_ionosonde_xr(self, id_key="id_dim"):
        self.ds = xr.open_dataset(self.file_path, decode_coords="all")
        # ID is a char array; convert to string
        self.ds = self.ds.assign(id=(id_key, ["".join(self.ds["ID"].values.astype(str))]))
        return

    def remove_vertical_bias(self, data):
        """
        data: 2D NumPy array (rows × columns), e.g. image or gridded measurements.
        """
        # compute median (or mean) per column, giving shape (columns,)
        col_offset = np.median(data, axis=1)
        # broadcast subtraction across rows to remove column-specific bias
        r_top = np.argmin(np.abs(
            self.ds.range.values-self.filter_params["range_top"]
        ))
        r_bot = np.argmin(np.abs(
            self.ds.range.values-self.filter_params["range_bot"]
        ))
        f_top = np.argmin(np.abs(
            self.ds.frequency.values-self.filter_params["freq_top"]
        ))
        f_bot = np.argmin(np.abs(
            self.ds.frequency.values-self.filter_params["freq_bot"]
        ))

        cleaned = data - col_offset[:, None]
        cleaned[cleaned < self.filter_params["noise_level"]] = 0
        cleaned[:, r_top:] = 0.
        cleaned[:, :r_bot] = 0.
        cleaned[f_top:, :] = 0.
        cleaned[:f_bot, :] = 0.
        return cleaned

    def identify_cluster(self):
        self.cleaned_data = self.remove_vertical_bias(
            np.array(self.ds.power.values).T
        )
        # scaled_points = StandardScaler().fit_transform(cleaned_points)
        mask = (self.cleaned_data != 0) & ~np.isnan(self.cleaned_data)
        rows, cols = np.nonzero(mask)
        coords = np.column_stack((rows, cols)).astype(float)
        coords[:, 0] /= self.scaling_params["freq_axis_shrink"]
        coords[:, 1] /= self.scaling_params["range_axis_shrink"]   

        # Run DBSCAN on these coordinates
        db = DBSCAN(
            eps=self.scaling_params["eps"], 
            min_samples=self.scaling_params["min_samples"]
        ).fit(coords)
        self.label_grid = np.full(self.cleaned_data.shape, -1, dtype=db.labels_.dtype)
        self.label_grid[rows, cols] = db.labels_
        self.find_params()
        return

    def find_params(self):
        from OIASA import (
            get_ground_distance,
            get_theta,
            get_curvature_correction,
            get_virtual_height,
            get_fv, get_phi
        )
        D = get_ground_distance(
            tuple(self.ds["receiveCoordinates"].values[:2]), 
            tuple(self.ds["transmitCoordinates"].values[:2])
        )
        theta = get_theta(D)
        c = get_curvature_correction(theta)
        logger.info(f"Ground Distance (km): {'%.2f'%D}" + f", Theta (deg): {'%.2f'%theta}" + f", Curvature Correction (km): {'%.2f'%c}")
        self.scaled_params = []
        time_str = self.file_path.split("/")[-1].replace(".nc", "").split("_")[-2:]
        self.time = dt.datetime.strptime(f"{time_str[0]} {time_str[1]}", "%Y-%m-%d %H%M%S")
        for u in np.unique(self.label_grid):
            logger.info(f"Cluster ID: {u}")
            data = self.cleaned_data[self.label_grid == u]
            if u >= 0:
                mask = self.label_grid == u
                f_idx, r_idx = np.where(mask)
                f_points = self.ds.frequency.values[f_idx]
                r_points = self.ds.range.values[r_idx]
                
                f_spread, r_spread = (
                    (np.max(f_points) - np.min(f_points)), 
                    (np.max(r_points) - np.min(r_points))
                )
                if f_spread > 1. and r_spread > 10:
                    logger.info(f"  Frequency Spread (MHz): {'%.2f'%f_spread}, Range Spread (km): {'%.2f'%r_spread}")
                    fo, rv = np.max(f_points), np.min(r_points)
                    phi = get_phi(D, rv)
                    fv, hv = get_fv(fo, phi), get_virtual_height(rv, c, phi)
                    logger.info(f"    fo (MHz): {'%.2f'%fo}, rv (km): {'%.2f'%rv}    phi (deg): {'%.2f'%phi}")
                    logger.info(f"    fv (MHz): {'%.2f'%fv}, hv (km): {'%.2f'%hv}")
                    self.scaled_params.append({
                        "cluster_id": u,
                        "fo": fo,
                        "rv": rv,
                        "phi": phi,
                        "fv": fv,
                        "hv": hv,
                        "time": self.time,
                    })
        return

    def to_pandas(self, need_fo=False):
        data = [dict(
            time=self.time,
        )]
        for j, param in enumerate(self.scaled_params):
            data[0][f"fv_{j}"] = param["fv"]
            data[0][f"hv_{j}"] = param["hv"]
            if need_fo:
                data[0][f"fo_{j}"] = param["fo"]
                data[0][f"rv_{j}"] = param["rv"]
                data[0][f"phi_{j}"] = param["phi"]
        df = pd.DataFrame(data)
        return df

    def draw_ionograms(
        self, 
        fig_title=f"Obliqe: WSMR-KR835, 14 Oct 2023",
        fig_dir="figures/",
        figsize=(5, 5), 
        font_size=20
    ):
        ionogram = Ionogram(fig_title=fig_title, nrows=1, ncols=3, figsize=figsize, font_size=font_size)
        time_str = self.file_path.split("/")[-1].replace(".nc", "").split("_")[-2:]
        time_str = f"{time_str[1][:2]}:{time_str[1][2:4]} UTC"
        
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            np.array(self.ds.power.values).T, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=False,
            xlabel="Frequency (MHz)",
            ylabel="Range (km)",
            text=f"{time_str} \n (A) Raw Ionogram",
            prange=[0, 30],
            cbar_label="Power (dB)"
        )
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            self.cleaned_data, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=False,
            xlabel="Frequency (MHz)",
            ylabel="",
            text=f"(B) Cleaned",
            prange=[0, 30],
            cbar_label="Power (dB)"
        )
        cluster_img = self.label_grid+1
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            cluster_img, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=True,
            xlabel="Frequency (MHz)",
            ylabel="",
            text=f"(C) Segmented",
            prange=[np.unique(cluster_img).min(), np.unique(cluster_img).max()],
            cbar_label="Power (dB)"
        )
        ax = ionogram.axes[2]
        df = self.to_pandas(need_fo=True)
        if "fv_1" in df.columns:
            fo = df.fo_1.values[0]
            print(f"fo: {fo} / {np.log10(fo)}", ax.get_xlim())
            ax.axvline(np.log10(fo), color="white", linestyle="--", linewidth=2, zorder=4)
        for ax in ionogram.axes:
            ax.set_xlim(np.log10([2, 15]))
        fname = fig_dir + f"{time_str}_ol.png"
        ionogram.save(fname)
        ionogram.close()
        return
    
    def close(self):
        self.ds.close()
        return


import glob
date_str = "2023-10-14"
files = glob.glob(f"/tmp/Oblique/*{date_str}*.nc")
files.sort()
datasets = pd.DataFrame()
for file in files:
    sp = ScalingProcess(file)
    sp.identify_cluster()
    sp.draw_ionograms()
    datasets = pd.concat([datasets, sp.to_pandas()], ignore_index=True)
    sp.close()
    # break
    
datasets.to_csv(f"data/Conway/oblique_scaling_{date_str}.csv", index=False, header=True)
# base_dir = "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/"
# sp = ScalingProcess(
#     f"{base_dir}KirtlandDPS4D-WSMRreceiver_2023-10-14_150000.nc",
# )
# sp.identify_cluster()
# sp.draw_ionograms()

