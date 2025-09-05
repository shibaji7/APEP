import sys
sys.path.extend(["py/", "py/2024/"])
import datetime as dt

from loguru import logger
from fetch import copy2local
import os

from pynasonde.vipir.ngi.source import DataSource

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
ds.load_data_sets(60*16, 60*17)

# TODO ERRIE