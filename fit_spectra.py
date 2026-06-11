"""
fit_spectra.py
Performs nested sampling to fit cached grid spectra to observed MIRI data.
Usage: mpiexec -n 4 python fit_spectra.py <path_to_obs_data> <distance_to_source>
"""
import os
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
import ultranest
from spectres import spectres
from utils import load_config, get_grid_axes_from_index
from scipy.constants import astronomical_unit as au
from scipy.constants import parsec as pc

tic = time.time()
c_kms = 299792.458  

# ==========================================
# 1. LOAD CONFIGURATION
# ==========================================
config_data = load_config("config.yaml")

CORRECT_VELOCITY   = config_data["CORRECT_VELOCITY"]
R_MAX_LIMIT        = 10**config_data.get("LOG_RAD_MAX", 5.0) # Safe fallback
N_LIVE_POINTS      = config_data["N_LIVE_POINTS"]
EVIDENCE_TOLERANCE = config_data["EVIDENCE_TOLERANCE"]
FRAC_REMAIN        = config_data["FRAC_REMAIN"]
MOLECULE_CONFIG    = config_data["MOLECULES"]

# Load H2O template ONLY if dynamic velocity correction is True
if isinstance(CORRECT_VELOCITY, bool) and CORRECT_VELOCITY is True:
    try:
        H2O_template_spectra_for_vel_correction = np.loadtxt('./h2o_template.dat')
        h2o_template_1d_flux = H2O_template_spectra_for_vel_correction[:,1]
        h2o_template_1d_wave = H2O_template_spectra_for_vel_correction[:,0]
        correction_w_region = [[13.287, 13.300],[14.200, 14.220],[14.507, 14.524],[15.160, 15.190],[16.102, 16.129]]
    except Exception as e:
        print(f"Error loading H2O velocity template: {e}")
        sys.exit(1)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def compile_mcmc_meta(wave_rest, active_mol_names):
    combined_mask = np.zeros_like(wave_rest, dtype=bool)
    for name in active_mol_names:
        for wl_range in MOLECULE_CONFIG[name]["masks"]:
            combined_mask |= (wave_rest >= wl_range[0]) & (wave_rest <= wl_range[1])
    return combined_mask

def rebin_grid_for_source(source_wave, intermediate_wave, cached_grid_2d, n_T, n_N):
    """Now accepts n_T and n_N as arguments since they are molecule-specific."""
    rebinned_2d = spectres(source_wave, intermediate_wave, cached_grid_2d, verbose=False, fill=0.0)
    return rebinned_2d.reshape((n_T, n_N, len(source_wave)))

def prior_transform(cube):
    params = np.array(cube, copy=True)
    for idx, name in enumerate(active_molecules):
        p_offset = idx * 2
        t_bounds = MOLECULE_CONFIG[name]["bounds"]["T"]
        n_bounds = MOLECULE_CONFIG[name]["bounds"]["logN"]
        params[p_offset]     = t_bounds[0] + cube[p_offset] * (t_bounds[1] - t_bounds[0])
        params[p_offset + 1] = n_bounds[0] + cube[p_offset + 1] * (n_bounds[1] - n_bounds[0])
    return params

def log_likelihood(theta):
    try:
        n_mol = len(active_molecules)
        mol_templates = []
        for idx, mol_name in enumerate(active_molecules):
            p_offset = idx * 2
            T, N = theta[p_offset : p_offset + 2]
            
            # Use the MOLECULE-SPECIFIC grid boundaries extracted from its index.txt!
            meta = grid_meta[mol_name]
            if not (meta['T_MIN'] <= T <= meta['T_MAX'] and meta['N_MIN'] <= N <= meta['N_MAX']): 
                return -np.inf
                
            mol_templates.append(interps[idx]((T, N)) * FLUX_SCALE)
            
        y_data = source_flux[fit_mask]
        weights = 1.0 / (source_err[fit_mask]**2)
        
        A_mat = np.zeros((n_mol, n_mol))
        b_vec = np.zeros(n_mol)
        for j in range(n_mol):
            M_j = mol_templates[j][fit_mask]
            b_vec[j] = np.sum(y_data * M_j * weights)
            for k in range(n_mol):
                A_mat[j, k] = np.sum(M_j * mol_templates[k][fit_mask] * weights)
                
        try:
            best_area_factors = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            return -np.inf
            
        best_area_factors = np.clip(best_area_factors, 0.0, R_MAX_LIMIT**2)
        full_model = sum(mol_templates[idx] * best_area_factors[idx] for idx in range(n_mol))
        
    except ValueError:
        return -np.inf
        
    chi2 = np.sum(((source_flux[fit_mask] - full_model[fit_mask]) / source_err[fit_mask])**2)
    return -0.5 * chi2

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fit_spectra.py <path_to_data_file> <distance_to_source (optional)>")
        sys.exit(1)
        
    obs_file = sys.argv[1]
    name = os.path.basename(obs_file).split('.')[0]

    try:
        distance = float(sys.argv[2])
    except:
        print('Distance to source not provided, assuming 150 pc')
        distance = 150.0 # pc
        
    FLUX_SCALE = np.pi * (au / distance / pc)**2 * 1e23
    
    # Load Observation
    data = np.loadtxt(obs_file)
    w, f, e, continuum = data[:,0], data[:,1], data[:,2], data[:,3]
    
    inds = np.argsort(w)
    source_wave_raw, flux, source_err, continuum = w[inds], f[inds], e[inds], continuum[inds]
    source_flux = flux - continuum

    # --- Velocity Correction Router ---
    if isinstance(CORRECT_VELOCITY, bool):
        if CORRECT_VELOCITY is True:
            h2o_mask = np.zeros_like(source_wave_raw, dtype=bool)
            for wl_range in correction_w_region:
                h2o_mask |= (source_wave_raw >= wl_range[0]) & (source_wave_raw <= wl_range[1])
                
            if np.max(np.abs(source_flux[h2o_mask])) > 3.0 * np.median(source_err[h2o_mask]):
                v_mask = (source_wave_raw >= 17.0) & (source_wave_raw <= 17.5)
                def velocity_objective(v_kms):
                    wave_rest = source_wave_raw[v_mask] / (1.0 + v_kms / c_kms)
                    template_rebinned = spectres(wave_rest, h2o_template_1d_wave, h2o_template_1d_flux)
                    num = np.sum((source_flux[v_mask] * template_rebinned) / source_err[v_mask]**2)
                    den = np.sum((template_rebinned**2) / source_err[v_mask]**2)
                    if den == 0: return np.inf
                    return np.sum(((source_flux[v_mask] - (num/den) * template_rebinned) / source_err[v_mask])**2)
                
                best_v_kms = minimize_scalar(velocity_objective, bounds=(-50, 50), method='bounded').x
                print(f"   --> [Velocity Correction] Located optimal radial velocity offset: {best_v_kms:.2f} km/s")
            else:
                best_v_kms = 0.0
        else:
            best_v_kms = 0.0
    elif isinstance(CORRECT_VELOCITY, (int, float)):
        best_v_kms = float(CORRECT_VELOCITY)
        print(f"   --> [Velocity Correction] Applied hardcoded offset: {best_v_kms:.2f} km/s")
    else:
        best_v_kms = 0.0
        
    source_wave_rest = source_wave_raw / (1.0 + best_v_kms / c_kms)

    # Initialize Active Targets
    active_molecules = list(MOLECULE_CONFIG.keys()) 
    
    # ---------------------------------------------------------
    # DYNAMIC GRID META EXTRACTOR
    # ---------------------------------------------------------
    interps, param_names = [], []
    grid_meta = {}  # Store the specific grid limits for each molecule
    
    for mol_name in active_molecules:
        print(f"Loading matrix and grid axes for: {mol_name}...")
        
        if "index_path" in MOLECULE_CONFIG[mol_name]:
            idx_path = MOLECULE_CONFIG[mol_name]["index_path"]
        else:
            idx_path = os.path.join(os.path.dirname(MOLECULE_CONFIG[mol_name]["cache_path"]), f"{mol_name}_index.txt")
            
        # 2. Extract specific T and N axes
        mol_Tg, mol_Ntot, mol_nT, mol_nN = get_grid_axes_from_index(idx_path)
        
        # 3. Store metadata for the log_likelihood function
        grid_meta[mol_name] = {
            'Tg': mol_Tg,
            'Ntot': mol_Ntot,
            'T_MIN': mol_Tg.min(),
            'T_MAX': mol_Tg.max(),
            'N_MIN': mol_Ntot.min(),
            'N_MAX': mol_Ntot.max()
        }
        
        # 4. Load Cache and Rebin
        raw_cache = np.load(MOLECULE_CONFIG[mol_name]["cache_path"])
        intermediate_wave = np.load(os.path.join(os.path.dirname(MOLECULE_CONFIG[mol_name]["cache_path"]), "intermediate_wave.npy"))
        
        # Pass the specific n_T and n_N to the rebinning function
        grid_3d = rebin_grid_for_source(source_wave_rest, intermediate_wave, raw_cache, mol_nT, mol_nN)
        
        # 5. Build Interpolator using specific axes
        interps.append(RegularGridInterpolator((mol_Tg, mol_Ntot), grid_3d, bounds_error=False, fill_value=np.inf))
        param_names.extend([f"{mol_name}_T", f"{mol_name}_logN"])
        
        del raw_cache, grid_3d

    fit_mask = compile_mcmc_meta(source_wave_rest, active_molecules)

    # --- ULTRANEST SAMPLING ---
    print(f"Initializing UltraNest Reactive Sampler...")
    log_dir = f"run_ultranest_{name}"
    
    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_likelihood, prior_transform, log_dir=log_dir, resume=True
    )
    
    result = sampler.run(
        min_num_live_points=N_LIVE_POINTS,
        Lepsilon=EVIDENCE_TOLERANCE,
        frac_remain=FRAC_REMAIN,
        viz_callback=False,
        max_iters=50000, 
        max_ncalls=1500000
    )
    
    sampler.print_results()
    print(f"Fit complete. Time taken = {time.time()-tic:.2f} seconds.")