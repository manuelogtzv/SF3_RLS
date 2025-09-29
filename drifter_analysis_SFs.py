import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm import trange
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed

# ------------------------
# Top-level function for pickling
# ------------------------
def _process_group(g):
    name, gdf = g
    return {
        "name": name,
        "time": gdf["datetime"].to_numpy(),
        "lat": gdf["lat"].astype(float).to_numpy(),
        "lon": gdf["lon"].astype(float).to_numpy(),
        "u": gdf["u"].astype(float).to_numpy(),
        "v": gdf["v"].astype(float).to_numpy()
    }

def load_glad_trajectories(input_file, output_file, max_workers=6):
    output_file = Path(output_file)

    # ------------------------
    # Read the data
    # ------------------------
    df = pd.read_csv(
        input_file,
        sep=r'\s+',  # regex for whitespace
        comment='%',
        header=None,
        names=["name", "date", "time", "lat", "lon", "col6", "u", "v", "col9"],
        dtype={"date": str, "time": str}
    )

    # Strip string columns only
    df = df.astype(str).apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].str[:8])

    # Drop unused columns
    df = df.drop(columns=["col6", "date", "time", "col9"])

    # ------------------------
    # Group by drifter
    # ------------------------
    groups = list(df.groupby("name"))
    print(f"Found {len(groups)} drifters. Processing...")

    drifter_data = []

    # ------------------------
    # Parallel processing
    # ------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_group, g) for g in groups]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing drifters"):
            drifter_data.append(f.result())

    # ------------------------
    # Save to pkl
    # ------------------------
    with open(output_file, "wb") as f:
        pickle.dump(drifter_data, f)

    print(f"✅ Saved drifter data to {output_file}")
    return drifter_data



def _process_group_dict(group_dict):
    """Worker function: simply returns the input dict (already safe for multiprocessing)."""
    return group_dict

def load_laser_trajectories(input_file, output_file, max_workers=4):
    """
    Load LASER drifter data, separate by type (L, M, U, V),
    process in parallel safely, and save to pickle.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    # Read file as strings
    df = pd.read_csv(
        input_file,
        sep=r"\s+",
        header=None,
        comment="%",
        dtype=str,
        engine="python"
    )

    # Rename columns
    df.columns = ["name", "date", "time", "lat", "lon", "col6", "u", "v", "col9"]

    # Strip whitespace per column
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Convert numeric columns
    for col in ["lat","lon","u","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"].str[:8],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Determine drifter type
    df["type"] = df["name"].str[0]

    drifter_types = ["L","M","U","V"]
    drifter_data = {}

    for dtype in drifter_types:
        df_type = df[df["type"] == dtype]
        groups = list(df_type.groupby("name", sort=False))
        print(f"Found {len(groups)} {dtype}-type drifters. Processing...")

        # Prepare group dicts for multiprocessing
        group_dicts = []
        for name, g in groups:
            group_dicts.append({
                "name": name,
                "time": g["datetime"].tolist(),
                "lat": g["lat"].tolist(),
                "lon": g["lon"].tolist(),
                "u": g["u"].tolist(),
                "v": g["v"].tolist()
            })

        drifters = []
        # Parallel processing safely
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_group_dict, gd) for gd in group_dicts]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Processing {dtype}-type drifters"):
                drifters.append(f.result())

        drifter_data[dtype] = drifters

    # Save pickle
    with output_file.open("wb") as f:
        pickle.dump(drifter_data, f)

    print(f"✅ Saved LASER drifter data to {output_file}")
    return drifter_data

# ------------------------
# Top-level helper functions
# ------------------------

def compute_times(d):
    """Compute min, max, and dt stats for a single drifter."""
    times = np.array(d["time"])
    dt = np.diff(times)
    return {
        "tmin": times.min(),
        "tmax": times.max(),
        "dt_mean": dt.mean() if len(dt) > 0 else np.nan,
    }

def fill_traj(args):
    """Fill trajectory arrays for a single drifter."""
    i, d, T_axis = args
    traj_X = np.full(len(T_axis), np.nan)
    traj_Y = np.full(len(T_axis), np.nan)
    traj_U = np.full(len(T_axis), np.nan)
    traj_V = np.full(len(T_axis), np.nan)

    times = np.array(d["time"])
    idx_start = np.searchsorted(T_axis, times[0], side='left')
    idx_end = idx_start + len(times)

    traj_X[idx_start:idx_end] = d["lon"]
    traj_Y[idx_start:idx_end] = d["lat"]
    traj_U[idx_start:idx_end] = d["u"]
    traj_V[idx_start:idx_end] = d["v"]

    return i, traj_X, traj_Y, traj_U, traj_V

# ------------------------
# Main function
# ------------------------

def convert_drifter_struct_to_array(
    pkl_file_path,
    sel_data='LASER',
    output_dir='./',
    make_plots=False,
    max_workers=6
):
    # Load drifter data
    with open(pkl_file_path, 'rb') as f:
        drifter_data = pickle.load(f)

    # Select appropriate drifter list
    if sel_data == 'GLAD':
        drifters = drifter_data
    elif sel_data == 'LASER':
        drifters = drifter_data["L"]  # drogued type
    else:
        raise ValueError("sel_data must be 'GLAD' or 'LASER'")

    ndrifts = len(drifters)
    print(f"Processing {ndrifts} drifters from {sel_data}...")

    # ------------------------
    # Compute time stats in parallel
    # ------------------------
    time_stats = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_times, d) for d in drifters]
        for f in tqdm(as_completed(futures), total=ndrifts, desc="Time stats"):
            time_stats.append(f.result())

    tmin_all = np.array([s["tmin"] for s in time_stats])
    tmax_all = np.array([s["tmax"] for s in time_stats])
    dt_mean_all = np.array([s["dt_mean"] for s in time_stats])

    tmin_min = tmin_all.min()
    tmax_max = tmax_all.max()
    dt_mean_overall = np.nanmean(dt_mean_all)

    # Create common time axis
    T_axis = np.arange(tmin_min, tmax_max + dt_mean_overall, dt_mean_overall)

    # ------------------------
    # Allocate trajectory matrices
    # ------------------------
    trajmat_X = np.full((len(T_axis), ndrifts), np.nan)
    trajmat_Y = np.full((len(T_axis), ndrifts), np.nan)
    trajmat_U = np.full((len(T_axis), ndrifts), np.nan)
    trajmat_V = np.full((len(T_axis), ndrifts), np.nan)

    # ------------------------
    # Fill trajectories in parallel
    # ------------------------
    args_list = [(i, d, T_axis) for i, d in enumerate(drifters)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, X, Y, U, V in tqdm(executor.map(fill_traj, args_list),
                                  total=ndrifts, desc="Aligning"):
            trajmat_X[:, i] = X
            trajmat_Y[:, i] = Y
            trajmat_U[:, i] = U
            trajmat_V[:, i] = V

    # ------------------------
    # Save to pickle
    # ------------------------
    output_file = os.path.join(output_dir,
                               f"traj_mat_{sel_data}_15min_04_May_2021.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({
            "T_axis": T_axis,
            "trajmat_X": trajmat_X,
            "trajmat_Y": trajmat_Y,
            "trajmat_U": trajmat_U,
            "trajmat_V": trajmat_V
        }, f)

    print(f"Saved output to {output_file}")

    # ------------------------
    # Optional plots
    # ------------------------
    if make_plots:
        plt.figure()
        plt.subplot(211)
        plt.plot(trajmat_X)
        plt.title('Longitude')
        plt.subplot(212)
        plt.plot(trajmat_Y)
        plt.title('Latitude')
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(trajmat_U)
        plt.title('U Velocity')
        plt.subplot(212)
        plt.plot(trajmat_V)
        plt.title('V Velocity')
        plt.show()

        plt.figure()
        plt.plot(trajmat_X, trajmat_Y)
        plt.title('Trajectories')
        plt.show()

    return T_axis, trajmat_X, trajmat_Y, trajmat_U, trajmat_V


def estimate_drifter_depth(
    traj_pkl_file,       # input .pkl file with trajectory matrices (trajmat_X, trajmat_Y)
    output_pkl_file,     # output .pkl file to save depth array
    sandwell_nc_path,    # path to Sandwell/ETOPO bathymetry .nc
    make_plots=False,    # optional plotting
    max_workers=6        # threads for parallel computation
):
    """
    Estimate depth at drifter positions from a trajectory .pkl and Sandwell/ETOPO bathymetry.
    Saves depth array to output .pkl file.
    """

    # ------------------------
    # Load trajectory data
    # ------------------------
    with open(traj_pkl_file, "rb") as f:
        traj_data = pickle.load(f)

    traj_X = traj_data['trajmat_X']
    traj_Y = traj_data['trajmat_Y']

    # ------------------------
    # Load bathymetry
    # ------------------------
    ds = xr.open_dataset(sandwell_nc_path)
    lon = ds['lon'].values
    lat = ds['lat'].values
    elev = ds['Z'].values  # adjust if variable name differs

    # lon = lon - 360  # convert to [-180,180] if needed

    # ------------------------
    # Compute depth in parallel
    # ------------------------
    def depth_column(j):
        col_depth = np.full(traj_X.shape[0], np.nan)
        x_col = traj_X[:, j]
        y_col = traj_Y[:, j]

        valid_idx = ~np.isnan(x_col)
        x_valid = x_col[valid_idx]
        y_valid = y_col[valid_idx]

        lon_inds = np.searchsorted(lon, x_valid, side='right') - 1
        lat_inds = np.searchsorted(lat, y_valid, side='right') - 1

        col_depth[valid_idx] = elev[lat_inds, lon_inds]
        return j, col_depth

    Htraj = np.full(traj_X.shape, np.nan)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(depth_column, j) for j in range(traj_X.shape[1])]
        for f in tqdm(as_completed(futures), total=traj_X.shape[1], desc="Estimating depths"):
            j, col_depth = f.result()
            Htraj[:, j] = col_depth

    # ------------------------
    # Save to .pkl
    # ------------------------
    with open(output_pkl_file, "wb") as f:
        pickle.dump(Htraj, f)

    print(f"Saved trajectory depth matrix to {output_pkl_file}")

    # ------------------------
    # Optional plots
    # ------------------------
    if make_plots:
        plt.figure(figsize=(6,5))
        plt.subplot(211)
        plt.contourf(lon, lat, elev, cmap='terrain')
        plt.plot(traj_X[:,0], traj_Y[:,0], 'r-', label='Example drifter')
        plt.xlabel('Longitude'); plt.ylabel('Latitude')
        plt.legend()
        plt.title('Trajectories over bathymetry')

        plt.subplot(212)
        plt.plot(Htraj[:,0], 'b-')
        plt.xlabel('Time index'); plt.ylabel('Depth (m)')
        plt.title('Drifter depth')
        plt.tight_layout()
        plt.show()

    return Htraj


# -------------------
# Top-level functions
# -------------------
def geo_dist(a, b):
    """Geodesic distance in meters."""
    dx = (a[0]-b[0])*np.cos(np.deg2rad(0.5*(a[1]+b[1])))*111321
    dy = (a[1]-b[1])*111321
    return np.sqrt(dx**2 + dy**2)

def dist_rx(a, b):
    return (a[0]-b[0])*np.cos(np.deg2rad(0.5*(a[1]+b[1])))

def dist_ry(a, b):
    return a[1]-b[1]

def dist_du(a, b):
    return a - b

def process_time(i, trajmat_X, trajmat_Y, trajmat_U, trajmat_V, Htraj, depth_thresh):
    mask = ~np.isnan(trajmat_X[i,:]) & (Htraj[i,:] < depth_thresh)
    ids = np.where(mask)[0]

    if len(ids) <= 1:
        return {'dist': np.nan, 'dul': np.nan, 'dut': np.nan}

    X = trajmat_X[i, ids]
    Y = trajmat_Y[i, ids]
    U = trajmat_U[i, ids]
    V = trajmat_V[i, ids]

    Xvec = np.column_stack((X, Y))

    dist = pdist(Xvec, geo_dist)

    rx = pdist(Xvec, dist_rx)
    ry = pdist(Xvec, dist_ry)
    magr = np.sqrt(rx**2 + ry**2)
    rx /= magr
    ry /= magr

    dux = pdist(U.reshape(-1,1), dist_du)
    duy = pdist(V.reshape(-1,1), dist_du)

    dul = dux*rx + duy*ry
    dut = duy*rx - dux*ry

    return {'dist': dist, 'dul': dul, 'dut': dut}

# -------------------
# Main function
# -------------------
def compute_structure_pairs(traj_pkl, depth_pkl, output_pkl, depth_thresh=-500, n_jobs=6):
    with open(traj_pkl, 'rb') as f:
        traj = pickle.load(f)

    with open(depth_pkl, 'rb') as f:
        Htraj = pickle.load(f)

    T_axis = traj['T_axis']
    trajmat_X = traj['trajmat_X']
    trajmat_Y = traj['trajmat_Y']
    trajmat_U = traj['trajmat_U']
    trajmat_V = traj['trajmat_V']

    n_times = len(T_axis)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_time)(i, trajmat_X, trajmat_Y, trajmat_U, trajmat_V, Htraj, depth_thresh)
        for i in tqdm(range(n_times), desc="Processing pairs")
    )

    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)

    return results


# --------------------------
# Function to compute bins
# --------------------------
def process_bin(i, dist, dul, dut, dist_bin):
    idx = np.where((dist >= dist_bin[i]) & (dist < dist_bin[i+1]))[0]
    return {'dul': dul[idx], 'dut': dut[idx], 'n_pairs': len(idx)}

def save_binned_velocity_differences(input_pkl, output_pkl, dist_bin, plot=True, plot_bins=None, n_jobs=6):
    # Load structure pairs
    with open(input_pkl, 'rb') as f:
        pairs_time = pickle.load(f)
    
    # Align all pairs into single vectors
    dul_list, dut_list, dist_list = [], [], []
    for pair in pairs_time:
        if pair['dul'] is not None and pair['dut'] is not None:
            dul_list.append(np.atleast_1d(pair['dul']))
            dut_list.append(np.atleast_1d(pair['dut']))
            dist_list.append(np.atleast_1d(pair['dist']))
    
    dul = np.concatenate(dul_list) if dul_list else np.array([])
    dut = np.concatenate(dut_list) if dut_list else np.array([])
    dist = np.concatenate(dist_list) if dist_list else np.array([])
    
    dist_axis = 0.5*(dist_bin[:-1] + dist_bin[1:])
    
    # --------------------------
    # Compute binned pairs
    # --------------------------
    if n_jobs == 1:
        pairs_sep = [process_bin(i, dist, dul, dut, dist_bin) for i in trange(len(dist_axis), desc='Binning pairs')]
    else:
        pairs_sep = Parallel(n_jobs=n_jobs)(
            delayed(process_bin)(i, dist, dul, dut, dist_bin) for i in trange(len(dist_axis), desc='Binning pairs')
        )
    
    # Convert list of dicts into separate dict of arrays
    dul_binned = [p['dul'] for p in pairs_sep]
    dut_binned = [p['dut'] for p in pairs_sep]
    n_pairs = np.array([p['n_pairs'] for p in pairs_sep])
    
    # Save results directly
    result = {
        'dist_axis': dist_axis,
        'dul': dul_binned,
        'dut': dut_binned,
        'n_pairs': n_pairs
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(result, f)
    
    if not plot:
        return result

    
    # --------------------------
    # Plot: Number of pairs per bin
    # --------------------------
    pairs_per_bin = np.array([p['n_pairs'] for p in pairs_sep])
    plt.figure()
    plt.loglog(dist_axis, pairs_per_bin, 'o-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Number of pairs')
    plt.title('Number of pairs per distance bin')
    plt.grid(True)
    plt.show()
    
    # --------------------------
    # 2D histogram of normalized velocity differences (PDF, log scale)
    # --------------------------
    norm_vel_bins = np.linspace(-50, 50, 301)
    norm_vel_axis = 0.5*(norm_vel_bins[:-1] + norm_vel_bins[1:])
    Norm_vel_hist_dul = np.zeros((len(norm_vel_axis), len(dist_axis)))
    Norm_vel_hist_dut = np.zeros((len(norm_vel_axis), len(dist_axis)))
    
    for i, p in enumerate(pairs_sep):
        if p['n_pairs'] > 0:
            sigma_l = np.std(p['dul']) if np.std(p['dul']) > 0 else 1
            sigma_t = np.std(p['dut']) if np.std(p['dut']) > 0 else 1
            Norm_vel_hist_dul[:, i], _ = np.histogram(p['dul']/sigma_l, bins=norm_vel_bins, density=True)
            Norm_vel_hist_dut[:, i], _ = np.histogram(p['dut']/sigma_t, bins=norm_vel_bins, density=True)
    
    # Avoid log(0) issues
    Norm_vel_hist_dul[Norm_vel_hist_dul==0] = 1e-10
    Norm_vel_hist_dut[Norm_vel_hist_dut==0] = 1e-10
    
    # Plot 2D PDFs in log scale (linear x-axis)
    plt.figure(figsize=(8,5))
    plt.pcolormesh(dist_axis/1e3, norm_vel_axis, np.log10(Norm_vel_hist_dul), shading='auto', cmap='viridis')
    cbar = plt.colorbar(label='log10(PDF)')
    plt.xscale('log')
    plt.xlabel('Distance [km]')
    plt.ylabel('delta u_l / sigma')
    plt.title('2D PDF of longitudinal velocity differences')
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.pcolormesh(dist_axis/1e3, norm_vel_axis, np.log10(Norm_vel_hist_dut), shading='auto', cmap='viridis')
    cbar = plt.colorbar(label='log10(PDF)')
    plt.xscale('log')
    plt.xlabel('Distance [km]')
    plt.ylabel('delta u_t / sigma')
    plt.title('2D PDF of transverse velocity differences')
    plt.show()

    
    # --------------------------
    # Row of 1D PDFs for selected distance bins
    # --------------------------
    if plot_bins is not None:
        n_bins = len(plot_bins)
        fig, axes = plt.subplots(n_bins, 2, figsize=(10, 3*n_bins), sharex=True)
        if n_bins == 1:
            axes = axes[np.newaxis, :]
        
        for i, bin_idx in enumerate(plot_bins):
            if bin_idx >= len(pairs_sep):
                continue
            p = pairs_sep[bin_idx]
            if p['n_pairs'] == 0:
                continue
            
            sigma_l = np.std(p['dul']) if np.std(p['dul'])>0 else 1
            sigma_t = np.std(p['dut']) if np.std(p['dut'])>0 else 1

            pdf_l, _ = np.histogram(p['dul']/sigma_l, bins=norm_vel_bins, density=True)
            pdf_t, _ = np.histogram(p['dut']/sigma_t, bins=norm_vel_bins, density=True)

            # Gaussian reference
            f_normal = np.exp(-norm_vel_axis**2 / 2) / np.sqrt(2*np.pi)

            # Plot dul
            axes[i,0].bar(norm_vel_axis, pdf_l, width=norm_vel_axis[1]-norm_vel_axis[0], 
                          color='C0', alpha=0.6, label='dul')
            axes[i,0].plot(norm_vel_axis, f_normal, 'k--', label='Gaussian N(0,1)')
            axes[i,0].set_yscale('log')
            axes[i,0].set_ylabel('Probability Density')
            axes[i,0].set_title(f'dul, Bin {bin_idx} ({dist_axis[bin_idx]/1e3:.1f} km)')
            axes[i,0].grid(True, which='both', linestyle='--', linewidth=0.5)
            axes[i,0].legend()
            axes[i,0].set_ylim(1e-8, 2)
            
            # Plot dut
            axes[i,1].bar(norm_vel_axis, pdf_t, width=norm_vel_axis[1]-norm_vel_axis[0], 
                          color='C1', alpha=0.6, label='dut')
            axes[i,1].plot(norm_vel_axis, f_normal, 'k--', label='Gaussian N(0,1)')
            axes[i,1].set_yscale('log')
            axes[i,1].set_ylabel('Probability Density')
            axes[i,1].set_title(f'dut, Bin {bin_idx} ({dist_axis[bin_idx]/1e3:.1f} km)')
            axes[i,1].grid(True, which='both', linestyle='--', linewidth=0.5)
            axes[i,1].legend()
            axes[i,1].set_ylim(1e-8, 2)
        
        axes[-1,0].set_xlabel('delta u_l / sigma')
        axes[-1,1].set_xlabel('delta u_t / sigma')
        plt.tight_layout()
        plt.show()
    
    return {'dist_axis': dist_axis, 'pairs_sep': pairs_sep}
