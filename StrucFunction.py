# Code for generating Structure Functions (SF) of second and third order.
from scipy.integrate import cumulative_trapezoid as ctrpz
from tqdm.notebook import tqdm
import math
import numpy as np
import xarray as xr
import gsw_xarray as gsw
import time
from dask import delayed, compute  
from dask.diagnostics import ProgressBar
import dask.array as da
import pandas as pd


# ------------------------- Helper: shift or roll -------------------------
def shift_or_roll(data, kwargs, periodic=False):
    """Shift or roll an xarray.DataArray based on periodic flag."""
    if periodic:
        return data.roll({k: -v for k, v in kwargs.items()}, roll_coords=False)
    else:
        return data.shift(**kwargs)

# ------------------------- Velocity differences -------------------------
def diff_vel(udata, vdata, kwargs, periodic=False):
    """Compute velocity differences (shifted - original)."""
    dU = shift_or_roll(udata, kwargs, periodic=periodic) - udata
    dV = shift_or_roll(vdata, kwargs, periodic=periodic) - vdata
    return dU, dV

# ------------------------- Compute domain size automatically -------------------------
def compute_Lx_Ly(Xg, Yg, grid="deg"):
    """Estimate domain size from coordinates."""
    if grid == "deg":
        Lx = (Xg.max() - Xg.min()) * np.pi / 180. * 6371e3 * np.cos(0.5*(Yg.max()+Yg.min())*np.pi/180.)
        Ly = (Yg.max() - Yg.min()) * np.pi / 180. * 6371e3
    elif grid == "m":
        Lx = Xg.max() - Xg.min()
        Ly = Yg.max() - Yg.min()
    else:
        raise ValueError("grid must be 'deg' or 'm'")
    return float(Lx), float(Ly)

# ------------------------- Longitude distance -------------------------
def distx_dlon(xdata, ydata, kwargs, grid="deg", periodic=False, Lx=None):
    lon1 = shift_or_roll(xdata, kwargs, periodic=periodic)
    lon0 = xdata
    lat1 = shift_or_roll(ydata, kwargs, periodic=periodic)
    lat0 = ydata
    dlon = lon1 - lon0
    if periodic:
        if Lx is None:
            raise ValueError("Must provide Lx for periodic domain")
        dlon = (dlon + 0.5*Lx) % Lx - 0.5*Lx
    if grid == "deg":
        return (np.pi / 180.) * 6371e3 * dlon * np.cos(0.5 * (lat0 + lat1) * np.pi / 180.)
    elif grid == "m":
        return dlon

# ------------------------- Latitude distance -------------------------
def disty_dlat(ydata, kwargs, grid="deg", periodic=False, Ly=None):
    lat1 = shift_or_roll(ydata, kwargs, periodic=periodic)
    lat0 = ydata
    dlat = lat1 - lat0
    if periodic:
        if Ly is None:
            raise ValueError("Must provide Ly for periodic domain")
        dlat = (dlat + 0.5*Ly) % Ly - 0.5*Ly
    if grid == "deg":
        return (np.pi / 180.) * 6371e3 * dlat
    elif grid == "m":
        return dlat

# ------------------------- Total distance -------------------------
def dist_ll2xy(xdata, ydata, kwargs, grid="deg", periodic=False, Lx=None, Ly=None):
    if periodic and (Lx is None or Ly is None):
        Lx, Ly = compute_Lx_Ly(xdata, ydata, grid=grid)
    dx = distx_dlon(xdata, ydata, kwargs, grid=grid, periodic=periodic, Lx=Lx)
    dy = disty_dlat(ydata, kwargs, grid=grid, periodic=periodic, Ly=Ly)
    return np.sqrt(dx**2 + dy**2)

# ------------------------- Velocity rotation -------------------------
def uv_ult(dU, dV, dx, dy):
    norm = (dx**2 + dy**2) ** 0.5
    rx = dx / norm
    ry = dy / norm
    ull = rx * dU + ry * dV
    utt = -ry * dU + rx * dV
    return ull, utt

# ------------------------- Structure functions -------------------------
def SF2_3(ull, utt):
    SF2 = ull**2 + utt**2
    SF3 = ull * SF2
    return SF2, SF3

# ------------------------- Main processing (averaged) -------------------------
@delayed
def process_dcorr2(dcorr1, dcorr2, Udata, Vdata, Xg, Yg, grid="deg", periodic=False, Lx=None, Ly=None):
    if periodic and (Lx is None or Ly is None):
        Lx, Ly = compute_Lx_Ly(Xg, Yg, grid=grid)
    kwargs = {'XC': dcorr1, 'YC': dcorr2}
    dU, dV = diff_vel(Udata, Vdata, kwargs, periodic=periodic)
    dx = distx_dlon(Xg, Yg, kwargs, grid=grid, periodic=periodic, Lx=Lx)
    dy = disty_dlat(Yg, kwargs, grid=grid, periodic=periodic, Ly=Ly)
    dr = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    ull, utt = uv_ult(dU, dV, dx, dy)
    dul2, dut2 = ull**2, utt**2
    _, du3 = SF2_3(ull, utt)
    return (
        dr.mean(dim=('XC', 'YC')),
        ull.mean(dim=('XC', 'YC')),
        utt.mean(dim=('XC', 'YC')),
        theta.mean(dim=('XC', 'YC')),
        dul2.mean(dim=('XC', 'YC')),
        dut2.mean(dim=('XC', 'YC')),
        du3.mean(dim=('XC', 'YC')),
    )

# ------------------------- Main processing (full fields - no spatial averaging) -------------------------
@delayed
def process_dcorr(dcorr1, dcorr2, Udata, Vdata, Xg, Yg, grid="deg", periodic=False, Lx=None, Ly=None):
    if periodic and (Lx is None or Ly is None):
        Lx, Ly = compute_Lx_Ly(Xg, Yg, grid=grid)
    kwargs = {'XC': dcorr1, 'YC': dcorr2}
    dU, dV = diff_vel(Udata, Vdata, kwargs, periodic=periodic)
    dx = distx_dlon(Xg, Yg, kwargs, grid=grid, periodic=periodic, Lx=Lx)
    dy = disty_dlat(Yg, kwargs, grid=grid, periodic=periodic, Ly=Ly)
    dr = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    ull, utt = uv_ult(dU, dV, dx, dy)
    return dr, ull, utt, theta


# ------------------------- Derived scales -------------------------
def timescale(sf2, dr):
    """Calculate decorrelation timescale from 2nd-order structure function."""
    return dr / sf2**0.5

def rossby_r(sf2, dr, fcor):
    """Calculate Rossby number from 2nd-order structure function."""
    return sf2**0.5 / (dr * fcor)

def binSF_aver(ds, bins, dr_name="dr", time_name="time", dcorr_name="dcorr"):
    """
    Bin structure function dataset by separation distance and average over orientations.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dimensions (dcorr, time) and a separation coordinate (dr).
    bins : array-like
        Bin edges for separation distance.
    dr_name : str, optional
        Name of the separation coordinate (default "dr").
    time_name : str, optional
        Name of the time dimension (default "time").
    dcorr_name : str, optional
        Name of the decorrelation/orientation dimension (default "dcorr").
    
    Returns
    -------
    xr.Dataset
        Binned and averaged structure function dataset with dims (time, mid_rbins).
    """
    
    # Bin midpoints
    bin_labels = np.array(bins[:-1]) + np.diff(bins)/2
    
    # Assign bin labels for each (time, dcorr) pair
    dr_values = ds[dr_name].values.ravel()
    dr_bins = pd.cut(dr_values, bins=bins, labels=bin_labels)
    dr_bins = dr_bins.to_numpy().reshape(ds[dr_name].shape)
    
    # Add as coordinate
    ds = ds.assign_coords(dr_bin=( (dcorr_name, time_name), dr_bins ))
    
    # Define delayed grouping per time slice
    @delayed
    def groupby_rbins_time(t):
        ds_t = ds.sel({time_name: t})
        dr_bins_t = ds_t["dr_bin"].values
        ds_t = ds_t.assign_coords(mid_rbins=(dcorr_name, dr_bins_t))
        grouped = ds_t.groupby("mid_rbins").mean(dim=dcorr_name)
        grouped = grouped.expand_dims({time_name: [t]})
        return grouped
    
    # Create tasks
    tasks = [groupby_rbins_time(t) for t in ds[time_name].values]
    results = compute(*tasks)
    
    # Concatenate along time
    SF = xr.concat(results, dim=time_name)
    
    return SF

def calculateSF_2(u, v, maxcorr, shiftdim, max_dist, grid, periodic, aver_spat=True, sf2_3=True):
    '''Calculates structure functions without ensemble averaging.
    Input:
        Uds: Xarray dataset with uvel and vvel as the zonal and meridional velocity components, 
             and X and Y as the zonal and meridional distances, and time as the time.
        maxcorr: Maximum number of shifts.
        per: True if double periodic, False if not
        shiftdim: Dimensions to shift.
        grid: 'm' when is in meters and 'deg' when is lat/lon
        aver_spat=True: Averages over all positions X and Y
        sf2_3=True: Calculates du2 and du3
        
    Output:
        SF_dist: Xarray dataset with dimensions X, Y, time, dcorr (number of shifts) and variables:
            ulls: du_L longitudinal component
            utts: du_T transversal component
            dr: separation distance between pairs of observations
            theta: orientation angle [rad]'''
    
    output_vars = {}
    # tasks = []
    delayed_results = []
    
    dr_list = []
    ull_list, utt_list, theta_list = [], [], []
    
    Udata = u
    Vdata = v

    # Precompute meshgrid once
    Xg, Yg, tg = xr.broadcast(u.XC, u.YC, u.time)
    Xg = Xg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    Yg = Yg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    tg = tg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    
    # Grid spacing
    dx_mean = np.abs(Xg.diff('XC').min().values.item())
    dy_mean = np.abs(Yg.diff('YC').min().values.item())
    
    # If grid is in degrees, convert to meters approx (optional: refine with actual lat/lon calc)
    if grid == 'deg':
        dx_mean *= 111e3 * np.cos(np.deg2rad(u.YC.mean().item()))
        dy_mean *= 111e3

    for dcorr1 in range(-maxcorr + 1, maxcorr):
        for dcorr2 in range(-maxcorr + 1, maxcorr):
            if dcorr1 != 0 or dcorr2 != 0:
                # Estimate radial distance
                approx_dr = np.sqrt((dcorr1 * dx_mean)**2 + (dcorr2 * dy_mean)**2)
                
                # Apply constraint
                if approx_dr <= max_dist:
                    delayed_results.append(
                        process_dcorr(dcorr1, dcorr2, Udata, Vdata, 
                                      Xg, Yg, grid, periodic, Lx=None, Ly=None)
                    )

    # Compute the tasks in parallel using Dask
    print("Calculating pairwise velocity differences")
    results = compute(*delayed_results)  # <-- Use compute from Dask
    
    with ProgressBar():
        # Unpack results from Dask's delayed objects
        dr_list, ull_list, utt_list, theta_list = zip(*results)

    dr = xr.concat(dr_list, dim='dcorr')
    ull = xr.concat(ull_list, dim='dcorr')
    utt = xr.concat(utt_list, dim='dcorr')
    theta = xr.concat(theta_list, dim='dcorr')
    
    # Averages theta for all position s
    theta_mean = theta.mean(dim=('XC', 'YC', 'time'))
    dr_mean = dr.mean(dim=('XC', 'YC', 'time'))
    
    output_vars['dr'] = dr
    output_vars['dr'].name = '$r$'
    output_vars['dr'].attrs['long_name'] = 'Separation distance $r$'
    output_vars['dr'].attrs['units'] = 'meters'
    
    output_vars['ulls'] = ull
    output_vars['ulls'].name = '$\\delta u_L(\\mathbf{s}, \\mathbf{r}, t)$'
    output_vars['ulls'].attrs['long_name'] = 'Longitudinal velocity fluctuation component'
    output_vars['ulls'].attrs['units'] = 'm/s'
    
    output_vars['utts'] = utt
    output_vars['utts'].name = '$\\delta u_T(\\mathbf{s}, \\mathbf{r}, t)$'
    output_vars['utts'].attrs['long_name'] = 'Transversal velocity fluctuation component'
    output_vars['utts'].attrs['units'] = 'm/s'
    
    output_vars['theta'] = theta
    output_vars['theta'].name = '$\\Theta(\\mathbf{s}, \\mathbf{r}, t)$'
    output_vars['theta'].attrs['long_name'] = 'Orientation of shift direction ($\\pi$)'
    output_vars['theta'].attrs['units'] = 'radians'
    
    if sf2_3==True:
        # Calculates samples of second- and third-order structure function
        _, du3 = SF2_3(ull, utt)
        dul2, dut2 = ull**2, utt**2
        
        # output_vars['du2'] = du2
        # output_vars['du2'].name = '$\\delta u2$'
        # output_vars['du2'].attrs['long_name'] = 'Second-order velocity fluctuation'
        # output_vars['du2'].attrs['units'] = 'm^2/s^2'

        output_vars['dul2'] = dul2
        output_vars['dul2'].name = '$\\delta u2_L$'
        output_vars['dul2'].attrs['long_name'] = 'Second-order longitudinal velocity fluctuation'
        output_vars['dul2'].attrs['units'] = 'm^2/s^2'

        output_vars['dut2'] = dut2
        output_vars['dut2'].name = '$\\delta u2_T$'
        output_vars['dut2'].attrs['long_name'] = 'Second-order transversal velocity fluctuation'
        output_vars['dut2'].attrs['units'] = 'm^2/s^2'
        
        output_vars['du3'] = du3
        output_vars['du3'].name = '$\\delta u3$'
        output_vars['du3'].attrs['long_name'] = 'Third-order velocity fluctuation'
        output_vars['du3'].attrs['units'] = 'm^3/s^3'
            
    
    # Create the final xarray dataset
    SFdist = xr.Dataset(output_vars)
    SFdist = SFdist.assign_coords(dcorr=('dcorr', dr_mean.data))
    
    if aver_spat==True:
        
        # Estimates spatially-averaged fields
        SFdist = SFdist.mean(dim=('XC','YC'))
        
    
    return SFdist

def calculateSF_2aver_exc(u, v, maxcorr, shiftdim, max_dist, grid, periodic):
    '''Calculates structure functions without ensemble averaging.
    Input:
        Uds: Xarray dataset with uvel and vvel as the zonal and meridional velocity components, 
             and X and Y as the zonal and meridional distances, and time as the time.
        maxcorr: Maximum number of shifts.
        shiftdim: Dimensions to shift.
        grid: 'm' when is in meters and 'deg' when is lat/lon
        max_dist: 'm' maximum distance to calculate
        period: 'False' if not double-periodic.
        
    Output:
        SF_dist: Xarray dataset with dimensions X, Y, time, dcorr (number of shifts) and variables:
        ulls: du_L longitudinal component
        utts: du_T transversal component 
        dr: separation distance between pairs of observations [m]
        theta: orientation angle [rad]
        du2: second-order velocity fluctuations du_L**2 + du_T**2
        du3: third-order velocity fluctuations du_L**3 + du_L*du_T**2'''
    
    output_vars = {}
    delayed_results = []
    
    dr_list = []
    ull_list, utt_list, theta_list = [], [], []
    dul2_list, dut2_list, du3_list = [], [], []

    Udata, Vdata = u, v
    Xg, Yg, tg = xr.broadcast(u.XC, u.YC, u.time)
    Xg = Xg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    Yg = Yg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    tg = tg.chunk({'XC': len(u.XC), 'YC': len(u.YC), 'time': len(u.time)})
    
    # Get chunk sizes from u
    # Get dict of chunk sizes
    # u_chunks = dict(zip(u.dims, u.chunks))
    
    # Xg = Xg.chunk({dim: u_chunks[dim] for dim in Xg.dims if dim in u_chunks})
    # Yg = Yg.chunk({dim: u_chunks[dim] for dim in Yg.dims if dim in u_chunks})
    # tg = tg.chunk({dim: u_chunks[dim] for dim in tg.dims if dim in u_chunks})


    # Grid spacing
    dx_mean = np.abs(Xg.diff('XC').min().values.item())
    dy_mean = np.abs(Yg.diff('YC').min().values.item())
    
    # If grid is in degrees, convert to meters approx (optional: refine with actual lat/lon calc)
    if grid == 'deg':
        dx_mean *= 111e3 * np.cos(np.deg2rad(u.YC.mean().item()))
        dy_mean *= 111e3

    for dcorr1 in range(-maxcorr + 1, maxcorr):
        for dcorr2 in range(-maxcorr + 1, maxcorr):
            if dcorr1 != 0 or dcorr2 != 0:
                # Estimate radial distance
                approx_dr = np.sqrt((dcorr1 * dx_mean)**2 + (dcorr2 * dy_mean)**2)
                
                # Apply constraint
                if approx_dr <= max_dist:
                    delayed_results.append(
                        process_dcorr2(dcorr1, dcorr2, Udata, Vdata, 
                                       Xg, Yg, grid, periodic, Lx=None, Ly=None)
                    )

    # Use tqdm for description support
    print("Calculating pairwise velocity differences")
    computed = compute(*delayed_results)

    with ProgressBar():
        # Unpack results from Dask's delayed objects
        # dr_list, ull_list, utt_list, theta_list, du2_list, du3_list = zip(*computed)
        dr_list, ull_list, utt_list, theta_list, dul2_list, dut2_list, du3_list = zip(*computed)

    # Concatenate results
    print("Concatenate")
    dr = xr.concat(dr_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    ull = xr.concat(ull_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    utt = xr.concat(utt_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    theta = xr.concat(theta_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    # du2 = xr.concat(du2_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    dul2 = xr.concat(dul2_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    dut2 = xr.concat(dut2_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))
    du3 = xr.concat(du3_list, dim='dcorr')#.chunk(chunkdic).mean(dim=('XC', 'YC'))


    output_vars['dr'] = dr
    output_vars['dr'].attrs.update(name='$r$', long_name='Separation distance $r$', units='meters')

    output_vars['ulls'] = ull
    output_vars['ulls'].attrs.update(name='$\\delta u_L$', 
                                     long_name='Longitudinal velocity fluctuation', units='m/s')

    output_vars['utts'] = utt
    output_vars['utts'].attrs.update(name='$\\delta u_T$', 
                                     long_name='Transversal velocity fluctuation', units='m/s')

    output_vars['theta'] = theta
    output_vars['theta'].attrs.update(name='$\\Theta$', 
                                      long_name='Orientation of shift direction ($\\pi$)', units='radians')
    
    # output_vars['dul2'] = du2
    # output_vars['du2'].attrs.update(name='$\\delta u2$', 
    #                                 long_name='Second-order fluctuation', units='m^2/s^2')
    
    output_vars['dul2'] = dul2
    output_vars['dul2'].attrs.update(name='$\\delta u2_L$', 
                                    long_name='Second-order longitudinal fluctuation', units='m^2/s^2')
    
    output_vars['dut2'] = dut2
    output_vars['dut2'].attrs.update(name='$\\delta u2_T$', 
                                    long_name='Second-order transversal fluctuation', units='m^2/s^2')
    
    output_vars['du3'] = du3
    output_vars['du3'].attrs.update(name='$\\delta u3$', 
                                    long_name='Third-order fluctuation', units='m^3/s^3')

    SFdist = xr.Dataset(output_vars)
    theta_mean = theta.mean(dim=('time'))
    SFdist = SFdist.assign_coords(dcorr=('dcorr', theta_mean.data))
    print('Done')    
    
    return SFdist




