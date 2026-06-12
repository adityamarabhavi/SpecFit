"""
plot_results.py
Generates corner plots and spectral fits from UltraNest posteriors.
Usage: python plot_results.py <path_to_data_file> <distance_to_source (optional)>
"""
import os
import sys
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
from utils import load_config, get_grid_axes_from_index
from spectres import spectres
from scipy.constants import astronomical_unit as au
from scipy.constants import parsec as pc

# ==========================================
# 1. COMMAND LINE INPUTS & CONFIGURATION
# ==========================================
if len(sys.argv) < 2:
    print("Usage: python plot_results.py <path_to_data_file> <distance_to_source (optional)>")
    sys.exit(1)

obs_file = sys.argv[1]
name = os.path.basename(obs_file).split('.')[0]
log_dir = f"run_ultranest_{name}"
chain_file = os.path.join(log_dir, "chains", "equal_weighted_post.txt")

try:
    distance = float(sys.argv[2])
except IndexError:
    print('Distance to source not provided, assuming 150 pc')
    distance = 150.0 # pc
    
FLUX_SCALE = np.pi * (au / distance / pc)**2 * 1e23

config = load_config("config.yaml")
MOLECULE_CONFIG = config["MOLECULES"]
CORRECT_VELOCITY = config["CORRECT_VELOCITY"]
R_MAX_LIMIT = 10**config.get("LOG_RAD_MAX", 5.0)

# ==========================================
# 2. AUTO-SCREENER
# ==========================================
def auto_screen_molecules(source_wave, source_flux, source_err):
    """Dynamically screens molecules based on the noise floor."""
    detected = []
    for mol_name in MOLECULE_CONFIG.keys():
        mol_mask = np.zeros_like(source_wave, dtype=bool)
        for wl_range in MOLECULE_CONFIG[mol_name]["masks"]:
            mol_mask |= (source_wave >= wl_range[0]) & (source_wave <= wl_range[1])
        if not np.any(mol_mask): continue
        if np.max(np.abs(source_flux[mol_mask])) >= 2.5 * np.median(source_err[mol_mask]):
            detected.append(mol_name)
    return detected

# ==========================================
# 3. LOAD DATA & DYNAMICALLY FILTER LABELS
# ==========================================
data = np.loadtxt(obs_file)
w, f, e, continuum = data[:,0], data[:,1], data[:,2], data[:,3]
inds = np.argsort(w)
source_wave_raw = w[inds]
flux = f[inds]
source_err = e[inds]
continuum = continuum[inds]
source_flux = flux - continuum

if isinstance(CORRECT_VELOCITY, (int, float)) and not isinstance(CORRECT_VELOCITY, bool):
    best_v_kms = float(CORRECT_VELOCITY)
else:
    best_v_kms = 0.0 
    
c_kms = 299792.458
source_wave_rest = source_wave_raw / (1.0 + best_v_kms / c_kms)

active_molecules = list(MOLECULE_CONFIG.keys()) 

fit_mask = np.zeros_like(source_wave_rest, dtype=bool)
for mol in active_molecules:
    for wl_range in MOLECULE_CONFIG[mol]["masks"]:
        fit_mask |= (source_wave_rest >= wl_range[0]) & (source_wave_rest <= wl_range[1])

# ---------------------------------------------------------
# DYNAMIC GRID META EXTRACTOR & MATRIX BUILDER
# ---------------------------------------------------------
interps, labels = [], []

for mol_name in active_molecules:
    cache_path = MOLECULE_CONFIG[mol_name]["cache_path"]
    idx_path = MOLECULE_CONFIG[mol_name]["index_path"]
    mol_dir = os.path.dirname(cache_path)
    wave_path = os.path.join(mol_dir, "intermediate_wave.npy")
    mol_Tg, mol_Ntot, mol_nT, mol_nN = get_grid_axes_from_index(idx_path)
    raw_cache = np.load(cache_path)
    intermediate_wave = np.load(wave_path)
    rebinned_2d = spectres(source_wave_rest, intermediate_wave, raw_cache, verbose=False, fill=0.0)
    grid_3d = rebinned_2d.reshape((mol_nT, mol_nN, len(source_wave_rest)))
    interps.append(RegularGridInterpolator((mol_Tg, mol_Ntot), grid_3d, bounds_error=False, fill_value=np.inf))
    labels.extend([rf"$T_{{\mathrm{{{mol_name}}}}}$ (K)", rf"$\log N_{{\mathrm{{{mol_name}}}}}$"])
    del raw_cache, grid_3d

if not os.path.exists(chain_file):
    raise FileNotFoundError(f"Missing sampling file: '{chain_file}'. Did the UltraNest run complete?")

samples = np.loadtxt(chain_file, skiprows=1)
best_params = np.percentile(samples, 50, axis=0)

# ==========================================
# 4. INDEPENDENT RADIUS MATRIX SOLVER
# ==========================================
def get_individual_radii_multi(theta_vector):
    n_mol = len(active_molecules)
    mol_templates = [interps[i]((theta_vector[i*2], theta_vector[i*2+1])) * FLUX_SCALE for i in range(n_mol)]
    y_data, weights = source_flux[fit_mask], 1.0 / (source_err[fit_mask]**2)
    A_mat = np.zeros((n_mol, n_mol))
    b_vec = np.zeros(n_mol)
    for j in range(n_mol):
        M_j = mol_templates[j][fit_mask]
        b_vec[j] = np.sum(y_data * M_j * weights)
        for k in range(n_mol):
            A_mat[j, k] = np.sum(M_j * mol_templates[k][fit_mask] * weights)
    try:
        best_areas = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        return np.zeros(n_mol)
    return np.sqrt(np.clip(best_areas, 0.0, R_MAX_LIMIT**2))

derived_radii = get_individual_radii_multi(best_params)
title_summary = ", ".join([f"R({name})={rad:.3f}au" for name, rad in zip(active_molecules, derived_radii)])

# ==========================================
# 5. EXECUTING VISUAL DIAGNOSTIC OUTPUT
# ==========================================
output_pdf = f"corner_plot_report_{name}.pdf"
print(f"Compiling plots into: {output_pdf}")

with PdfPages(output_pdf) as pdf:
    fig_corner = corner.corner(
        samples, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True, 
        title_fmt=".2f", color="darkblue", range=[0.999] * len(labels)
    )
    fig_corner.suptitle(f"Posteriors: {name}\nDerived Radii: {title_summary}", y=1.05, fontsize=10)
    pdf.savefig(fig_corner, bbox_inches='tight')
    plt.close(fig_corner)

    fig_spec, ax = plt.subplots(2, figsize=(18, 8), sharex=True)
    all_masks = [wl for mol in active_molecules for r in MOLECULE_CONFIG[mol]["masks"] for wl in r]
    w_min, w_max = min(all_masks) - 0.2, max(all_masks) + 0.2
    mask_plot = (source_wave_rest >= w_min) & (source_wave_rest <= w_max)
    
    ax[0].step(source_wave_rest[mask_plot], source_flux[mask_plot], c='k', where='mid', label='Data (Subtracted)', lw=1)
    ax[1].step(source_wave_rest[mask_plot], flux[mask_plot], c='k', where='mid', label='Total Raw Data', lw=1)
    ax[1].fill_between(source_wave_rest[mask_plot], 0, continuum[mask_plot], step='mid', color='gray', alpha=0.2, label='Provided Continuum')

    running_model = np.zeros_like(source_wave_rest)
    for idx, mol_name in enumerate(active_molecules):
        T, N = best_params[idx*2 : idx*2+2]
        mol_model = interps[idx]((T, N)) * (derived_radii[idx]**2) * FLUX_SCALE
        color = MOLECULE_CONFIG[mol_name].get("plot_color", "r")
        
        ax[0].fill_between(source_wave_rest[mask_plot], running_model[mask_plot], (running_model + mol_model)[mask_plot], step='mid', color=color, alpha=0.35, label=f'{mol_name}')
        ax[1].fill_between(source_wave_rest[mask_plot], (continuum + running_model)[mask_plot], (continuum + running_model + mol_model)[mask_plot], step='mid', color=color, alpha=0.35)
        running_model += mol_model
        
        for wl in MOLECULE_CONFIG[mol_name]["masks"]:
            ax[0].axvspan(wl[0], wl[1], alpha=0.08, color=color, zorder=-10)

    ax[0].step(source_wave_rest[mask_plot], running_model[mask_plot], c='r', where='mid', lw=1.2, label='Total Model')
    ax[1].step(source_wave_rest[mask_plot], (continuum + running_model)[mask_plot], c='r', where='mid', lw=1.2)

    ax[0].set_ylabel('Flux Density')
    ax[0].legend(loc='upper right', frameon=True)
    ax[0].set_title(f'Fit: {name}')
    ax[1].set_ylabel('Total Flux')
    ax[1].set_xlabel(r'Wavelength ($\mu$m)')
    ax[1].set_xlim(w_min, w_max)
    
    plt.tight_layout()
    pdf.savefig(fig_spec)
    plt.close(fig_spec)

print("Diagnostic reports saved successfully.")