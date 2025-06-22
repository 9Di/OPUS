import os
import pandas as pd
import xarray as xr
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as LR

# MAIN_DIR = 'C:/Users/Rustam/Downloads/20250619_pkl/20250619/' #pkl files folder
# FNS = np.sort(glob.glob(MAIN_DIR + '*pkl'))

# folder_path = "C:/Users/Rustam/Desktop/test/20250612_1/" #wv file folder
# abs_list = []
# metadata = []
# wvr = pd.read_csv("C:/Users/Rustam/Downloads/WV.csv").iloc[0, 1]
# wv_cleaned = re.sub(r'[\[\]\n]', ' ', wvr)
# wv = np.fromstring(wv_cleaned, sep=' ')

# file_path = "C:/Users/Rustam/Downloads/20250619/CALIBRAT.DAT"  #dat file path

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


def mk_xr_dataset_old(FNS, wv):

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
        
        timestamp[ifn] = obj.index.values[0]  # time
        
        I[ifn, :] = obj.to_numpy()[0][0]  # intensity
        T[ifn] = obj.to_numpy()[0][2]  # temperature
        Fcount[ifn] = obj.to_numpy()[0][3][0]  # flash count
        lR1[ifn] = obj.to_numpy()[0][4]
        lR0[ifn] = obj.to_numpy()[0][5]

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


def add_all_temps(file_path, a):

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
    
    Ts_opus = pd.DataFrame([time_clean, t1_ref_clean, t2_spec_clean, t3_cpu_clean]).T
    Ts_opus[0] = pd.to_datetime(Ts_opus[0])
    
    Ts_opus.set_index(0, drop=True, inplace=True)
    

    # shift time of opus temperatures  
    delta_time = Ts_opus.index[0] - a.time.values[0] # compute delta time
    Ts_opus.index = Ts_opus.index.shift(periods=1, freq=-delta_time) # shift time index
     
    Ts_opus.index.name = 'time' # add name to index
    Ts_opus.columns = ['T_RefDiode', 'T_RefSpectr', 'T_CPU'] # add names to columns


    # interpolate Ts_opus onto ds2 time axis
    for col in Ts_opus.columns:
    
        da = xr.DataArray(
            data=np.interp(a.time, Ts_opus.index, Ts_opus[col].astype(float)),
            dims="time",
            attrs=dict(
                description=col,
                units="degC",
            ),
        )
               
        a[col] = da
        
    return a, Ts_opus


def fit_linear(x, y):
    x = sm.add_constant(x)

    rlm = sm.RLM(y, x, sm.robust.norms.TrimmedMean(0.5))
    rlm_result = rlm.fit(maxiter=50,
                         tol=1e-08,
                         scale_est='mad',
                         init=None,
                         cov='H1',
                         update_scale=True,
                         conv='dev',
                         start_params=None
                         )
    rlm_result.summary()

    return rlm, rlm_result


def cmp_R_T_corr(ds, T_col, DIN):
    
    m_lRef = np.empty(ds['wv'].size) * np.nan
    m_T = np.empty(ds['wv'].size) * np.nan
    # q = np.empty(wv_i.size) * np.nan
        
    X = [   ds['lampR1'][:],
            ds[T_col][:]  ]
    X = np.asarray(X)
    
    for iiw,wv in enumerate(ds['wv'].values):
        
        
        lr = LR(fit_intercept=True)
        lr.fit(X.T, ds['Intensity'].values[:,iiw]);
        
        m_lRef[iiw], m_T[iiw] = lr.coef_       


    # save results
    da = xr.DataArray(
        data=m_lRef,
        dims='wv',
        coords={
                'wv': ('wv', ds['wv'].values),  # consistent naming
                },
        attrs=dict(
            description="slope for lampRef1 of multilinear fit of Intensity",
            units="-"
        ),
    )
    
    ds = da.to_dataset(name='m_lRef1')
    
    ds['m_' + T_col] = xr.DataArray(
        data=m_T,
        dims='wv',
        coords={
                'wv': ('wv', ds['wv'].values),  # consistent naming
                },
        attrs=dict(
            description="slope for T of multilinear fit of Intensity",
            units="counts/degC",
        ),
    )

    fn_out = DIN + '/Results/lampRef1_T_fit.nc'
    ds.to_netcdf(fn_out)

    return m_lRef, m_T


def cmp_R_T_corr_robust(ds, T_col, DIN):

    m_lRef = np.empty(ds['wv'].size) * np.nan
    m_T = np.empty(ds['wv'].size) * np.nan
    q = np.empty(ds['wv'].size) * np.nan
        
    X = [   ds['lampR1'][:],
            ds[T_col][:]  ]
    X = np.asarray(X)
    
    for iiw,wv in enumerate(ds['wv'].values):
        
        # robust fit
        rlm, rlm_result = fit_linear(X.T, ds['Intensity'].values[:,iiw])

        q[iiw], m_lRef[iiw], m_T[iiw]= rlm_result.params

    # save results
    da = xr.DataArray(
        data=m_lRef,
        dims='wv',
        coords={
                'wv': ('wv', ds['wv'].values),  # consistent naming
                },
        attrs=dict(
            description="slope for lampRef1 of multilinear robust fit of Intensity",
            units="-"
        ),
    )
    
    ds = da.to_dataset(name='m_lRef1')
    
    ds['m_' + T_col] = xr.DataArray(
        data=m_T,
        dims='wv',
        coords={
                'wv': ('wv', ds['wv'].values),  # consistent naming
                },
        attrs=dict(
            description="slope for T of multilinear robust fit of Intensity",
            units="counts/degC",
        ),
    )

    fn_out = DIN + '/Results/lampRef1_T_robust_fit.nc'
    ds.to_netcdf(fn_out)        

    return m_lRef, m_T, q









    