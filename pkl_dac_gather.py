import os
import pandas as pd
import xarray as xr
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

MAIN_DIR = 'C:/Users/Rustam/Downloads/20250619_pkl/20250619/' #pkl files folder
FNS = np.sort(glob.glob(MAIN_DIR + '*pkl'))

folder_path = "C:/Users/Rustam/Desktop/test/20250612_1/" #wv file folder
abs_list = []
metadata = []
wvr = pd.read_csv("C:/Users/Rustam/Downloads/WV.csv").iloc[0, 1]
wv_cleaned = re.sub(r'[\[\]\n]', ' ', wvr)
wv = np.fromstring(wv_cleaned, sep=' ')

file_path = "C:/Users/Rustam/Downloads/20250619/CALIBRAT.DAT"  #dat file path

def mk_xr_dataset(FNS, wv):
    # initialize arrays for data
    I = np.empty([len(FNS), len(wv)])
    timestamp = np.empty([len(FNS), 1])
    T = np.empty([len(FNS), 1])
    Fcount = np.empty([len(FNS), 1])
    lR0 = np.empty([len(FNS), 1])
    lR1 = np.empty([len(FNS), 1])

    # read all files
    for ifn in range(FNS.size):
        obj = pd.read_pickle(FNS[ifn])
        timestamp[ifn] = obj.index.values  # time
        I[ifn, :] = obj.to_numpy()[0][1]  # intensity
        T[ifn] = obj.to_numpy()[0][3]  # temperature
        Fcount[ifn] = obj.to_numpy()[0][4][0]  # flash count
        lR1[ifn] = obj.to_numpy()[0][5]
        lR0[ifn] = obj.to_numpy()[0][6]

    timestamp = pd.to_datetime(timestamp.flatten())  # convert to datetime64 datatype

    # create xarray dataset from dataarrays
    da = xr.DataArray(
        data=I,
        dims=['time', 'wv'],
        coords={'time': timestamp,
                'wv': ('wv', wv),  # consistent naming
                },
        attrs=dict(
            description="Intensity corrected for dark counts",
            units="counts",
        ),
    )
    ds = da.to_dataset(name='Intensity')

    ds['wv'] = ds.coords['wv'].assign_attrs({'units': 'nm'})

    da = xr.DataArray(
        data=lR0[:, 0],
        dims="time",
        coords={'time': timestamp},
        attrs=dict(
            description="lampRef0",
            units="counts",
        ),
    )
    ds['lampR0'] = da

    da = xr.DataArray(
        data=lR1[:, 0],
        dims="time",
        coords={'time': timestamp},
        attrs=dict(
            description="lampRef1",
            units="counts",
        ),
    )
    ds['lampR1'] = da

    da = xr.DataArray(
        data=T[:, 0],
        dims="time",
        coords={'time': timestamp},
        attrs=dict(
            description="tempRefDiode",
            units="degC",
        ),
    )
    ds['T'] = da


    da = xr.DataArray(
        data=Fcount[:, 0],
        dims="time",
        coords={'time': timestamp},
        attrs=dict(
            description="Number of flashes",
            units="-",
        ),
    )
    ds['Fcount'] = da

    return ds

time = []
t1_ref = []
t2_spec = []
t3_cpu = []

with open(file_path, 'r') as file:
    for line in file:
        if 'DateTime' in line:
            time.append(line)
        if 'tempRefDiode' in line:
            t1_ref.append(line)
        if 'tempSpectrometer' in line:
            t2_spec.append(line)
        if 'tempMainCPU' in line:
            t3_cpu.append(line)

time_clean = [time[x][11:30] for x in range(len(time))]
t1_ref_clean = [float(t1_ref[x][15:22]) for x in range(len(t1_ref))]
t2_spec_clean = [float(t2_spec[x][19:24]) for x in range(len(t2_spec))]
t3_cpu_clean = [float(t3_cpu[x][14:18]) for x in range(len(t3_cpu))]

df_temp = pd.DataFrame([time_clean, t1_ref_clean, t2_spec_clean, t3_cpu_clean]).T
df_temp[0] = pd.to_datetime(df_temp[0])
#df_temp[0] = df_temp[0] + pd.Timedelta(hours=2, minutes=1, seconds=40)
#df_temp = df_temp[df_temp[0]>=('2025-06-12 12:58')]

a = mk_xr_dataset(FNS, wv)

t_diff = a.time.values[0] - df_temp[0][0]
components = t_diff.components
df_temp[0] = df_temp[0] + pd.Timedelta(hours=components.hours, minutes=components.minutes, seconds=components.seconds)

times_a = a['time'].values
ref_diode = []
ref_spectr = []
cpu = []

for t in times_a:
    # Calculate absolute time difference
    diffs = np.abs(df_temp[0] - pd.Timestamp(t))

    # Get index of closest match
    min_idx = diffs.idxmin()

    # Append matched values
    ref_diode.append(df_temp.loc[min_idx, 1])
    ref_spectr.append(df_temp.loc[min_idx, 2])
    cpu.append(df_temp.loc[min_idx, 3])

a = a.assign(
    RefDiode=('time', ref_diode),
    RefSpectr=('time', ref_spectr),
    CPU=('time', cpu)
)

plt.plot(a.time, a.T)
plt.plot(a.time, a.RefDiode)
plt.plot(a.time, a.RefSpectr)
plt.plot(a.time, a.CPU)
plt.show()

