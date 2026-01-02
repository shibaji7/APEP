import xarray as xr
import numpy as np
from pynasonde.vipir.ngi.plotlib import Ionogram

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
        return

    def find_params(self):
        for u in np.unique(self.label_grid):
            mask = self.label_grid != u
            grid[self.label_grid!=u] = 
        return

    def draw_ionograms(
        self, 
        fig_title=f"Obliqe: WSMR-KR835, 14 Oct 2023",
        fig_dir="figures/",
        figsize=(5, 5), 
        font_size=20
    ):
        ionogram = Ionogram(fig_title=fig_title, nrows=2, ncols=2, figsize=figsize, font_size=font_size)
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
            xlabel="",
            ylabel="Range (km)",
            text=f"{time_str} \n (A) Raw Ionogram",
            prange=[0, 30]
        )
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            self.cleaned_data, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=True,
            xlabel="",
            ylabel="",
            text=f"(B) Cleaned",
            prange=[0, 30]
        )
        cluster_img = self.label_grid+1
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            cluster_img, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=False,
            xlabel="Frequency (MHz)",
            ylabel="Range (km)",
            text=f"(C) Segmented",
            prange=[np.unique(cluster_img).min(), np.unique(cluster_img).max()],
        )
        ionogram.add_ionogram(
            np.array(self.ds.frequency.values), 
            np.array(self.ds.range.values), 
            np.array(self.ds.power.values).T, 
            del_ticks=False,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=False,
            xlabel="Range (km)",
            ylabel="",
            text=f"(D) Scaled",
            prange=[0, 30]
        )
        for ax in ionogram.axes:
            ax.set_xlim(np.log10([2, 15]))

        fname = fig_dir + f"{time_str}_ol.png"
        ionogram.save(fname)
        ionogram.close()
        return
    
    def close(self):
        self.ds.close()
        return


# import glob
# files = glob.glob("/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/*.nc")
# for file in files:
#     print(file)
#     draw_ionograms(file, fname="figures/" + file.split("/")[-1].replace(".nc", ".png")) 
# file = "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/WSMR0-WSMRsorcer_2023-10-14_180218.nc"
# draw_ionograms(file)

base_dir = "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/"
sp = ScalingProcess(
    f"{base_dir}KirtlandDPS4D-WSMRreceiver_2023-10-14_150000.nc",
)
sp.identify_cluster()
sp.draw_ionograms()
# files = [
#     "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_150000.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_153600.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_160000.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_163600.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_170000.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_173600.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_180000.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_183600.nc",
#     # "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_190000.nc",
# ]
# draw_ionograms(files)