# Code for generating Structure Functions (SF) of second and third order.
# I followed Balwhada et al. (2022) to compute structure functions
from scipy.integrate import cumtrapz as ctrpz
from tqdm.notebook import tqdm
import math
import numpy as np
import xarray as xr
import gsw_xarray as gsw


def SF2_3(ull, utt):
    ''' Estimates 2nd and 3rd order structure function
        
        Input:
        
        ull, utt: Longitudinal and transversal structure function components''' 
    
    SF2 = ull**2 + utt**2
    SF3 = ull*SF2
    return SF2, SF3


def uv_ult(dU, dV, dx, dy):
    ''' Rotates velocity difference vector to longitudinal and transversal components (Balhwada 2022)
        
        Input:
        
        dU, dV: Zonal and meridional velocity difference component
        dx, dy: Zona and meridional distance vector '''     
    
    norm = (dx**2 + dy**2)**(1/2)
    rx = dx/norm
    ry = dy/norm
    utt = rx*dV - ry*dU # transversal: (u,v) dot [(0, 0, 1) x (dx, dy, 0)]
    ull = rx*dU + ry*dV # longitudinal: (u,v) dot (dx, dy)
    return ull, utt

def dist_ll2xy(xdata, ydata, kwargs, grid):
    ''' Calculates distance between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    

    lon1 = xdata.shift(**kwargs)
    lon0 = xdata
    lat1 = ydata.shift(**kwargs)
    lat0 = ydata
    
    if grid == "m":
        X = abs(lon0 - lon1)
        Y = abs(lat0 - lat1)
    elif grid =="deg":
        # Transforms from lon/lat to meters
        X = abs(lon0 - lon1)*np.cos(0.5*(lon1 + lon0) * np.pi/180.)* 111321
        Y = abs(lat0 - lat1)*111321
    return (X**2 + Y**2)**(1/2)

def distx_dlon(xdata, ydata, kwargs, grid):
    ''' Calculates X distance difference between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    
    lon1 = xdata.shift(**kwargs)
    lon0 = xdata
    lat1 = ydata.shift(**kwargs)
    lat0 = ydata
    
    if grid == "m":
        dlon = (lon0 - lon1)
        return dlon
    elif grid == "deg":
        dlon = (lon0 - lon1)*np.cos(0.5*(lat1 + lat0)*np.pi/180.)
        return (np.cos(np.pi/180*ydata) * np.pi/180.*6371e3*dlon)


def disty_dlat(ydata, kwargs, grid):
    ''' Calculates Y distance between pairs
    
        Input:
        xdata, ydata: X and Y vectors (xarray)
        kwargs: optional kwargs for shifting matrix 
        grid: "m" refers to metric, "deg" refers to lat/lon'''
    
    if grid =="m":
        dlat = (ydata.shift(**kwargs) - ydata)
    elif grid =="deg":
        dlat = (ydata.shift(**kwargs) - ydata)
        dlat = dnp.pi/180.* 6371e3 * dlat
    return dlat

def diff_vel(udata, vdata, kwargs):
    ''' Calculates velocity difference between pairs
    
        Input:
        udata, vdata: U and V velocity components (xarray)
        kwargs: optional kwargs for shifting matrix '''
    
    dU = udata.shift(**kwargs) - udata
    dV = vdata.shift(**kwargs) - vdata
    dU = dU
    dV = dV
    return dU, dV

def timescale(sf2, dr):
    ''' Calculates decorrelation time scale from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins '''
    
    return dr/sf2**(1/2)

def rossby_r(sf2, dr, fcor):
    ''' Calculates rossby number from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins 
        fcor: Coriolis frequency'''    
    
    return sf2**(1/2)/(dr*fcor)

def calculateSF(Uds, maxcorr, shiftdim, stackdim, grid):
    '''Calculates structure functions without ensemble averaging.
    
       Input:
       Uds: Xarray dataset with uvel and vvel as the zonal and meridional velocity components, 
            and X and Y as the zonal and meridional distances, and time as the time.'''
    
    SFdist = xr.Dataset()
    output_vars = {}
    dr_list, sf2_list, sf3_list = [], [], []
    ull_list, utt_list = [], []
    
    Udata = Uds.u
    Vdata = Uds.v
    
    # meshgrid of lon,lat
    Xg, Yg, tg = xr.broadcast(Uds.x, Uds.y, Uds.time) 
    
    # Shifting data
    for dcorr in tqdm(range(1, maxcorr)): #creates kwargs
        kwargs = {}
        for idx in shiftdim:
            kwargs[idx] = dcorr
            
        # Calculates velocity differences and distance
        dU, dV = diff_vel(Udata, Vdata, kwargs)
        dy = disty_dlat(Yg, kwargs, grid)
        dx = distx_dlon(Xg, Yg, kwargs, grid)
        dr = (dx**2 + dy**2)**(1/2)
        
        ull, utt = uv_ult(dU, dV, dx, dy) #converts to longitudinal and transversal component
        sf2, sf3 = SF2_3(ull, utt) # second and third order SF
    
        # Append results into lists
        dr_list.append(dr)        
        ull_list.append(ull)
        utt_list.append(utt)
        sf2_list.append(sf2)
        sf3_list.append(sf3)
    
    # Concatenate data 
    dr = xr.concat(dr_list, dim='dcorr').stack(r=stackdim)
    ull = xr.concat(ull_list, dim='dcorr').stack(r=stackdim)
    utt = xr.concat(utt_list, dim='dcorr').stack(r=stackdim)
    StFc2 = xr.concat(sf2_list, dim='dcorr').stack(r=stackdim)
    StFc3 = xr.concat(sf3_list, dim='dcorr').stack(r=stackdim)
    
    # Drops Nans for total
    dr = dr.dropna(dim="r", how="all")
    StFc2 = StFc2.dropna(dim="r", how="all")
    StFc3 = StFc3.dropna(dim="r", how="all")
    ull = ull.dropna(dim="r", how="all")
    utt = utt.dropna(dim="r", how="all")

    # Output variables in data array 
    output_vars['D2s'] = StFc2.compute()
    output_vars['D3s'] = StFc3.compute()
    output_vars['ulls'] = ull.compute()
    output_vars['utts'] = utt.compute()
    output_vars['utt2'] = utt**2
    output_vars['ull2'] = ull**2
    SFdist = SFdist.assign(**output_vars)
    SFdist['dr'] = dr.compute()
    
    SFdist = SFdist.reset_index('r')
    SFdist = SFdist.assign_coords(r=('r', SFdist.dr.isel(time=0).values))
    
    
    # Binned means and std dev
    #meanSF = averageSF(SFdist, rbins)
    #stdSF = stddevSF(SFdist, rbins)
    
    #return SFdist, meanSF, stdSF #returns xarrays
    return SFdist


def SFbin_xr(SFdist, rbins):
    '''Calculates ensemble averages of structure functions.
    
       Input:
       SFdist: Xarray dataset calculated from calculateSF, 
       rbins: Vector with separation distance'''

    # rbins = np.linspace(0, 2e5, 25)
    mid_rbins = 0.5*(rbins[0:-1] + rbins[1:])

    # Estimates structure function
    d3s_mean = np.zeros((len(SFdist.time.values), len(mid_rbins)))
    d2s_mean = np.copy(d3s_mean)*0.
    ul_mean = np.copy(d3s_mean)*0.
    ut_mean = np.copy(d3s_mean)*0.
    ul2_mean = np.copy(d3s_mean)*0.
    ut2_mean = np.copy(d3s_mean)*0.
    r_mean = np.copy(d3s_mean)*0.
    nobs = np.copy(d3s_mean)*0.

    d3s_std = np.copy(d3s_mean)*0.
    d2s_std = np.copy(d2s_mean)*0.
    ul_std = np.copy(ul_mean)*0.
    ut_std = np.copy(ut_mean)*0.
    ul2_std = np.copy(ul2_mean)*0.
    ut2_std = np.copy(ut2_mean)*0.
    r_std = np.copy(r_mean)*0.

    for ii in tqdm(np.arange(0, len(SFdist.time))):
        groupD2s = SFdist.isel(time=ii).groupby_bins('dr', rbins, labels=mid_rbins)
        mall = groupD2s.mean(...)
        nobs_SF = groupD2s.count(...)
        
        # Creates np arrays
        d3s_mean[ii, :] = mall.D3s.values
        d2s_mean[ii, :] = mall.D2s.values
        ul_mean[ii, :] = mall.ulls.values
        ut_mean[ii, :] = mall.utts.values
        ul2_mean[ii, :] = mall.ull2.values
        ut2_mean[ii, :] = mall.utt2.values
        r_mean[ii, :] = mall.dr.values
        nobs[ii, :] = nobs_SF.D2s.values
    
        sall = groupD2s.std(...)
        d3s_std[ii, :] = sall.D3s.values
        d2s_std[ii, :] = sall.D2s.values
        ul_std[ii, :] = sall.ulls.values
        ut_std[ii, :] = sall.utts.values
        ul2_std[ii, :] = sall.ull2.values
        ut2_std[ii, :] = sall.utt2.values
        r_std[ii, :] = sall.dr.values
        
    # Saves variables in Xarray dataset
    mSF1 = xr.Dataset(data_vars=dict(D2s=(["time", "rbins"], d2s_mean),
                                     D3s=(["time", "rbins"], d3s_mean),
                                     ulls=(["time", "rbins"], ul_mean),
                                     utts=(["time", "rbins"], ut_mean),
                                     ull2=(["time", "rbins"], ul2_mean),
                                     utt2=(["time", "rbins"], ut2_mean),
                                     dr=(["time", "rbins"], r_mean),
                                     nobs=(["time", "rbins"], nobs)),
                      coords=dict(time=SFdist.time, rbins=mid_rbins),
                      attrs=dict(description="Structure functions as a function of separation distance and time"),)
    
    sSF1 = xr.Dataset(data_vars=dict(D2s=(["time", "rbins"], d2s_std),
                                     D3s=(["time", "rbins"], d3s_std),
                                     ulls=(["time", "rbins"], ul_std),
                                     utts=(["time", "rbins"], ut_std),
                                     ull2=(["time", "rbins"], ul2_std),
                                     utt2=(["time", "rbins"], ut2_std),
                                     dr=(["time", "rbins"], r_std)),
                      coords=dict(time=SFdist.time, rbins=mid_rbins),
                     attrs=dict(description="Std. Dev. Structure functions as a function of separation distance and time"),)
    
    # Name and attributes
    mSF1.D2s.name = 'SF2'
    mSF1.D2s.attrs['long_name'] = 'Second Order Structure Function'
    mSF1.D2s.attrs['units'] = 'm^2/s^2'
    mSF1.D3s.name = 'SF3'
    mSF1.D3s.attrs['long_name'] = 'Third Order Structure Function'
    mSF1.D3s.attrs['units'] = 'm^3/s^3'
    mSF1.ulls.name = 'u_{ll}'
    mSF1.ulls.attrs['long_name'] = 'Averaged longitudinal velocity component'
    mSF1.ulls.attrs['units'] = 'm/s'
    mSF1.utts.name = 'u_{tt}'
    mSF1.utts.attrs['long_name'] = 'Averaged transversal velocity component'
    mSF1.utts.attrs['units'] = 'm/s'
    mSF1.utt2.name = 'u^2_{tt}'
    mSF1.utt2.attrs['long_name'] = 'Second Order Structure Function transveral velocity component'
    mSF1.utt2.attrs['units'] = 'm^2/s^2'
    mSF1.ull2.name = 'u^2_{ll}'
    mSF1.ull2.attrs['long_name'] = 'Second Order Structure Function longitudinal velocity component'
    mSF1.ull2.attrs['units'] = 'm^2/s^2'
    mSF1.dr.name = 'r'
    mSF1.dr.attrs['long_name'] = 'Mean Separation distance'
    mSF1.dr.attrs['units'] = 'm'
    mSF1.nobs.name = 'nobs'
    mSF1.nobs.attrs['long_name'] = 'Number of observations'



    # Name and attributes
    sSF1.D2s.name = 'SF2'
    sSF1.D2s.attrs['long_name'] = 'Std. Dev. Second Order Structure Function'
    sSF1.D2s.attrs['units'] = 'm^2/s^2'
    sSF1.D3s.name = 'SF3'
    sSF1.D3s.attrs['long_name'] = 'Std. Dev. Third Order Structure Function'
    sSF1.D3s.attrs['units'] = 'm^3/s^3'
    sSF1.ulls.name = 'u_{ll}'
    sSF1.ulls.attrs['long_name'] = 'Std. Dev. Averaged longitudinal velocity component'
    sSF1.ulls.attrs['units'] = 'm/s'
    sSF1.utts.name = 'u_{tt}'
    sSF1.utts.attrs['long_name'] = 'Std. Dev. Averaged transversal velocity component'
    sSF1.utts.attrs['units'] = 'm/s'
    sSF1.utt2.name = 'u^2_{tt}'
    sSF1.utt2.attrs['long_name'] = 'Std. Dev. Second Order Structure Function transveral velocity component'
    sSF1.utt2.attrs['units'] = 'm^2/s^2'
    sSF1.ull2.name = 'u^2_{ll}'
    sSF1.ull2.attrs['long_name'] = 'Std. Dev. Second Order Structure Function longitudinal velocity component'
    sSF1.ull2.attrs['units'] = 'm^2/s^2'
    sSF1.dr.name = 'r'
    sSF1.dr.attrs['long_name'] = 'Std. Dev. Mean Separation distance'
    sSF1.dr.attrs['units'] = 'm'
    
    return mSF1, sSF1

def timescale(sf2, dr):
    ''' Calculates decorrelation time scale from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins '''
    return dr/sf2**(1/2)

def rossby_r(sf2, dr, fcor):
    ''' Calculates rossby number from 2nd-order structure function
    
        Input:
        sf2: Second-order structure function
        dr: Separation distance bins 
        fcor: Coriolis frequency'''    
    
    return sf2**(1/2)/(dr*fcor)