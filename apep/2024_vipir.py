import sys
sys.path.extend(["apep/"])
import datetime as dt

from loguru import logger
from fetch import copy2local
import os

from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.utils import load_toml
from pynasonde.vipir.ngi.scale import AutoScaler, NoiseProfile

cfg = load_toml()


# Copy to local folder
folder = "/media/chakras4/Crucial X9/Solar_Eclipse_2024/public/WI937/individual/2024/099/ionogram/"
local = folder.replace("/media/", "/tmp/")
if not os.listdir(local):
    logger.info("Downloading.....")
    os.makedirs(local, exist_ok=True)
    copy2local(local, folder)

# Create a list of events for 2024/WI937 and later ERRIE
logger.info(f"Local folder:{local}")
ds = DataSource(source_folder=local)
ds.load_data_sets(60*14, 60*21)

for i, dx in enumerate(ds.datasets):
    scaler = AutoScaler(
        dx,
        noise_profile=NoiseProfile(constant=cfg.ngi.scaler.noise_constant),
        mode=cfg.ngi.scaler.mode,
        filter=dict(
            frequency=[cfg.ngi.scaler.frequency_min, cfg.ngi.scaler.frequency_max],
            height=[cfg.ngi.scaler.height_min, cfg.ngi.scaler.height_max],
        ),
        apply_filter=cfg.ngi.scaler.apply_filter,
        segmentation_method=cfg.ngi.scaler.segmentation_method,
    )
    scaler.mdeian_filter()
    scaler.image_segmentation()
    scaler.to_binary_traces(
        nbins=cfg.ngi.scaler.otsu.nbins,
        thresh=cfg.ngi.scaler.otsu.thresh,
        eps=cfg.ngi.scaler.dbscan.eps,
        min_samples=cfg.ngi.scaler.dbscan.min_samples,
    )
    scaler.draw_sanity_check_images(f"figures/2024/scan_{i}.png", font_size=15)
    del scaler
    break