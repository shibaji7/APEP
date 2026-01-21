from pathlib import Path
import sys
sys.path.extend([
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2]),
])

from pathlib import Path
import datetime as dt
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.digi_utils import get_digisonde_info

def main(derived_param="scaled"):
    base_dir = Path(f"data/Conway/")
    if not base_dir.exists():
        base_dir.mkdir(exist_ok=True) 
    target_date = dt.date(2023, 10, 14)
    time_limits = (
        dt.datetime.combine(target_date, dt.time.min),
        dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min),
    )

    local = f"/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2023_10_14/"
    df = SaoExtractor.load_SAO_files(
        folders=[local],
        func_name=derived_param,
        n_procs=12,
    )
    if derived_param == "scaled":
        stn_info = get_digisonde_info("WS833")
        df = df[["datetime", "local_datetime", "foF2", "foE", "hmF2", "hEs"]]
        df.rename(columns={
            "datetime": "UTC",
            "local_datetime": "Local_Time",
            "foF2": "foF2(MHz)",
            "foE": "foE(MHz)",
            "hmF2": "hmF2(km)",
            "hEs": "hE(km)",
        }, inplace=True)
        for k in stn_info.keys():
            df[k] = stn_info[k]
    else:
        df.rename(columns={
            "datetime": "UTC",
            "local_datetime": "Local_Time",
            "pf": "plasma_density(MHz)",
            "th": "height(km)",
            "ed": "density(cc)",
        }, inplace=True)
    print(df.head())
    df.to_csv(base_dir / f"WSMR833_{derived_param}.csv", header=True, index=False, float_format="%g")



if __name__ == "__main__":
    main()
    main("height_profile")
