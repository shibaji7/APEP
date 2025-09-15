import sys
sys.path.extend(["py/", "py/2024/"])
import datetime as dt
import os
from loguru import logger

from fetch import create_local_folder, copy2local
from skymap import create_skymap_overlay_tec

def copy(date, mode="SKYWAVE"):
    os.makedirs("figures/2024/", exist_ok=True)
    base = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/{mode}_DPS4D_{date.strftime('%Y_%m_%d')}/"
    local, remote = create_local_folder(base)
    if not os.listdir(local):
        logger.info("Downloading.....")
        copy2local(local, remote)
    return local, remote

copy(dt.datetime(2024, 4, 8))

create_skymap_overlay_tec(
    [
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099170913.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099172113.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099173313.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099174513.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099175713.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099180913.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099182113.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099183313.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099184513.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099185713.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099190913.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099192113.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099193313.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099194513.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099195713.SKY",
        # "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/KR835_2024099200913.SKY",
    ], 
    nrows=1, ncols=1,
    fig_title="08 April 2024 / KR835", 
    fname="figures/2024_sky_stack_KR835.png"
)


create_skymap_overlay_tec(
    [
                "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2024_04_08/WS833_2024099193313.SKY",
    ], 
    nrows=1, ncols=1,
    fig_title="08 April 2024 / WS833", 
    fname="figures/2024_sky_stack_WS833.png"
)