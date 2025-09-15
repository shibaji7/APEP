from xarray import open_dataset
import datetime as dt
import sys
import scipy.io as io
import h5py as hf
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

def read_tec_observations_glonas(
    date,
    fname="data/2024/fulltimedata_04_08_2024_32to36lat_-104to108lon.mat"
):
    timedt,ipplats,ipplons,dtec1,dtec2,el,az = read_tec_azimuth_fulltime(fname)

    timedt = pd.to_datetime(timedt).tolist()
    index = np.argmin([np.abs((date-d).total_seconds())for d in timedt])
    df = pd.DataFrame()
    (
        df["ipplats"], 
        df["ipplons"], 
        df["dtec1"], 
        df["dtec2"], 
        df["el"], 
        df["az"],
    ) = (
        ipplats[index], 
        ipplons[index], 
        dtec1[index], 
        dtec2[index], 
        el[index],
        az[index],
    )
    return df


def read_tec_azimuth_fulltime(ftec):
    alltec=hf.File(ftec,"r")
    timestamps=alltec["UTT"]
    times=np.concatenate(timestamps,axis=0)
    #######
    timedt=np.array([])
    ipplats=np.ones(len(times)-1,dtype=object)*np.NaN
    ipplons=np.ones(len(times)-1,dtype=object)*np.NaN
    dtec1=np.ones(len(times)-1,dtype=object)*np.NaN
    dtec2=np.ones(len(times)-1,dtype=object)*np.NaN
    el=np.ones(len(times)-1,dtype=object)*np.NaN
    az=np.ones(len(times)-1,dtype=object)*np.NaN
    arrs=alltec["fulltimedata"]
    for tidx,ts in enumerate(times[:-1]):
        tsdt=dt.datetime.fromordinal(int(ts)) + dt.timedelta(days=ts%1)-dt.timedelta(days=366)
        ###
        ref=arrs[tidx][0]
        dset=np.array(alltec[ref])
        try:
            nentries,nstat=np.shape(dset)
        except:
            continue
        ###
        ipplat=dset[3]
        ipplon=dset[4]
        filt1=dset[5]
        filt2=dset[6]
        elev=dset[1]
        azm=dset[2]
        # filt3=dset[6]
        ###
        timedt=np.append(timedt,tsdt)
        ipplats[tidx]=ipplat
        ipplons[tidx]=ipplon
        dtec1[tidx]=filt1
        dtec2[tidx]=filt2
        el[tidx]=elev
        az[tidx]=azm
    return timedt,ipplats,ipplons,dtec1,dtec2,el,az

if __name__ == "__main__":
    read_tec_observations_glonas(dt.datetime(2024,4,8,16))