"""
build_cache.py
Convolves prodimopy slab models and rebins them onto a standard wavelength grid, 
saving them as a .npy cache matrix.
"""

import numpy as np
import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import get_grid_axes_from_index
from scipy.constants import c
import prodimopy.read_slab as rs
from spectres import spectres

# ==========================================
# SETUP
# ==========================================
SLAB_DIR = './models/CO2/'                 # Directory containing the FITS models and index file
index_file = './models/CO2/CO2_index.txt'  # path to the file containing grid points of the slab grid
slab_filename_prefix = 'CO2'               # Prefix used for the slab models 
R_MIRI = 3000                              # Resolving power for convolution 
wavelength_range = [4.9,28]                # Wavelength range for the slab spectra
number_of_cores_to_use = 40                #'max', 'half', 'quarter', or an integer

# ==========================================
# ==========================================
# ==========================================
def _worker_process_model(args):
    """
    Isolated worker function to read, convolve, and rebin a single model.
    Must be top-level so it can be pickled by multiprocessing.
    """
    idx, model_path, r_miri, intermediate_wave = args
    try:
        # Read and convolve
        slab_data = rs.read_slab([model_path], verbose=False)
        slab_data.convolve(R=r_miri, lambda_0=wavelength_range[0], lambda_n=wavelength_range[1], overlap=True, verbose=False)
        
        # ProDiMo arrays need to be reversed to be strictly increasing for spectres
        highres_wave = slab_data.convOverlapWave[::-1]
        convolved_flux = slab_data.convOverlapLTE[::-1]
        
        # Rebin
        rebinned_flux = spectres(intermediate_wave, highres_wave, convolved_flux, verbose=False, fill=0.0)
        return idx, rebinned_flux
        
    except Exception as e:
        print(f"Error processing model at index {idx} ({model_path}): {e}")
        return idx, None

# ==========================================
# MAIN CACHE BUILDER
# ==========================================
def create_cache_parallel(mol_name, file_list, Tg, Ntot, output_dir, r_miri, cores_setting):
    """Dispatches cache building across multiple CPU cores."""
    n_T, n_N = len(Tg), len(Ntot)
    total_models = n_T * n_N
    
    # Establish common intermediate wave grid (in microns)
    intermediate_wave = c / rs.generate_grid(lambda_0=wavelength_range[0], lambda_n=wavelength_range[1], R=1e4) * 1e-3
    cached_grid = np.zeros((total_models, len(intermediate_wave)))
    
    # Determine Core Count
    total_sys_cores = multiprocessing.cpu_count()
    if cores_setting == 'max':
        n_workers = total_sys_cores
    elif cores_setting == 'half':
        n_workers = max(1, total_sys_cores // 2)
    elif cores_setting == 'quarter':
        n_workers = max(1, total_sys_cores // 4)
    else:
        try:
            n_workers = int(cores_setting)
        except ValueError:
            n_workers = 1 
            
    print(f"Building Cache for:  {mol_name}")
    print(f"Total Models:        {total_models}")
    print(f"Resolving Power:     R~{r_miri}")
    print(f"Parallelizing over:  {n_workers} CPU cores")
    print("-" * 40)
    
    # Package arguments for the worker pool
    tasks = [(i, file_list[i], r_miri, intermediate_wave) for i in range(total_models)]
    
    # Execute Multiprocessing
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_process_model, task): task for task in tasks}
        
        for future in as_completed(futures):
            idx, rebinned_flux = future.result()
            
            if rebinned_flux is not None:
                cached_grid[idx, :] = rebinned_flux
                
            completed += 1
            if completed % 100 == 0 or completed == total_models:
                print(f"  --> Processed {completed}/{total_models} models...")
                
    # Save the matrices
    cache_filename = os.path.join(output_dir, f"{mol_name}_R{int(r_miri)}_cache.npy")
    wave_filename = os.path.join(output_dir, "intermediate_wave.npy")
    
    np.save(cache_filename, cached_grid)
    np.save(wave_filename, intermediate_wave)
    print("\n[SUCCESS] Cache build complete.")
    print(f"Saved Matrix: {cache_filename}")
    print(f"Saved Wavelength Grid: {wave_filename}")

# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == '__main__':
   
    Tg, Ntot, n_T, n_N = get_grid_axes_from_index(index_file)
    
    # Grab the target files
    fits_files = sorted(glob.glob(os.path.join(SLAB_DIR, f"{slab_filename_prefix}_*.fits*")))
    
    if len(fits_files) != (n_T * n_N):
        print(f"WARNING: Found {len(fits_files)} files, but expected {n_T * n_N} based on the index.txt file!")
        print("The cache might misalign with the grid axes.")
        
    create_cache_parallel(
        mol_name=slab_filename_prefix,
        file_list=fits_files,
        Tg=Tg, 
        Ntot=Ntot,
        output_dir=SLAB_DIR,
        r_miri=R_MIRI,
        cores_setting=number_of_cores_to_use
    )