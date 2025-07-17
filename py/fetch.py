import shutil
import os
import glob
import numpy as np
from loguru import logger
import datetime as dt

def create_local_folder(
    base:str="/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_13/",
    clean:bool=False
):
    local = base
    if clean: shutil.rmtree(local, ignore_errors=True)
    os.makedirs(
        local, 
        exist_ok=True
    )
    logger.info(f"Created local folder: {local}")
    remote = local.replace("/tmp/", "/media/")
    logger.info(f"Remote folder: {remote}")
    return local, remote

def copy2local(local:str, remote:str, ext: str="*", ):
    remote_files = glob.glob(os.path.join(remote, ext))
    for remote_file in remote_files:
        fname = remote_file.split("/")[-1]
        shutil.copy2(remote_file, os.path.join(local, fname))
    return

if __name__ == "__main__":
    local, remote = create_local_folder()
    copy2local(local, remote)
