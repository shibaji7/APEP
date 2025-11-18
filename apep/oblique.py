import xarray as xr
import numpy as np
from pynasonde.vipir.ngi.plotlib import Ionogram

def load_ionosonde_xr(path):
    ds = xr.open_dataset(path, decode_coords="all")
    # ID is a char array; convert to string
    ds = ds.assign(id=("id_dim", ["".join(ds["ID"].values.astype(str))]))
    print(ds)
    return ds

def draw_ionograms(files, fname="figures/oblique_ionogram.png"):
    ionogram = Ionogram(fig_title=f"Obliqe: WSMR-KR835, 14 Oct 2023", nrows=3, ncols=3, figsize=(7, 7), font_size=20)
    for i, file in enumerate(files):
        ds = load_ionosonde_xr(file)
        time_str = file.split("/")[-1].replace(".nc", "").split("_")[-2:]
        time_str = f"{time_str[0]} {time_str[1][:2]}:{time_str[1][2:4]} UTC"
        ionogram.add_ionogram(
            np.array(ds.frequency.values), 
            np.array(ds.range.values), 
            np.array(ds.power.values).T, 
            del_ticks=i!=6,
            ylim=(50, 1000),
            xlim=(2, 15),
            add_cbar=i==8,
            xlabel="Frequency (MHz)" if i in [6] else "",
            ylabel="Virtual Height (km)" if i in [6] else "",
            text=f"({i}) {time_str}",
            prange=[5, 30]
        )
        ionogram.axes[i].set_xlim(np.log10([2, 15]))
    ionogram.save(fname)
    ionogram.close()
    ds.close()
    return


# import glob
# files = glob.glob("/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/*.nc")
# for file in files:
#     print(file)
#     draw_ionograms(file, fname="figures/" + file.split("/")[-1].replace(".nc", ".png")) 
# file = "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh/WSMR0-WSMRsorcer_2023-10-14_180218.nc"
# draw_ionograms(file)
files = [
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_150000.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_153600.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_160000.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_163600.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_170000.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_173600.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_180000.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_183600.nc",
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/receiver_files/for_Aroh//KirtlandDPS4D-WSMRreceiver_2023-10-14_190000.nc",
]
draw_ionograms(files)