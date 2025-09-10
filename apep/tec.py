from xarray import open_dataset
import datetime as dt
import sys
sys.path.append("py/")

from geometry import *
import utils


def read_datasets(dates):
    dtec = []
    for d in dates:
        ds = open_dataset(f"data/{d.year}/{d.strftime('%Y%m%d%H')}_dtec.nc")
        i = int(d.minute/5)
        dtec.append(dict(
            time=d,
            data=ds.dtec.values[:, :, i],
            lat=ds.lat.values,
            lon=ds.lon.values,
        ))
        ds.close()
    return dtec