"""
generate_grid.py
Generates a grid of 0D slab models using prodimopy and writes them to disk.
"""
import numpy as np
import os
from itertools import product as it_product
import prodimopy.hitran as ht
import prodimopy.run_slab as runs 

#===========================================================================================
# GRID PARAMETERS (Temperature in K, Column Density in cm-2, Turbulent broadening in km.s-1)
#===========================================================================================

Ttot = np.linspace(25,1500,60)
Ntot = np.logspace(14, 24.5, 64, endpoint=True)
vturb = 2 

params = np.asarray(list(it_product(Ttot, Ntot)))
T_grid = params[:,0]
N_grid = params[:,1]
vturb = np.full_like(T_grid,vturb)

#===========================================================================================
# SETUP
#===========================================================================================
output_dir = './models/CO2/'                             # Directory to store slab model output files
list_of_molecules = ['CO2','CO2']                        # for isotopologues input the main molecule here and iso number below
isotopolog_numbers = [1,2]                               # following HITRAN definitions
molecular_mass = [44,45]                                 # in amu
paths_to_hitran_files = ['/path/to/CO2/HITRAN/file.par',
                         '/path/to/CO2/HITRAN/file.par'] # downloaded from HITRAN
column_density_fractions = [1,1/70]                      # column density for each species will be scaled by their respective entries
wavelength_range = [4.9,28]                              # Wavelength range for the slab spectra
slab_filename_prefix = 'CO2'                             # Will result in slab models written out as CO2_00001.fits.gz, CO2_00002.fits.gz, ...
number_of_cores_to_use = 40                              # Can be an integer or 'max' or 'half' or 'quarter'

#===========================================================================================
#===========================================================================================
#===========================================================================================

os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    print(f"Generating {len(params)} slab models...")
    
    slabs = runs.run_0D_slab_grid(
        Ng=params[:, 1],  
        Tg=params[:, 0], 
        vturb=vturb, 
        molecule=list_of_molecules, 
        mol_mass=molecular_mass, 
        HITRANfile=paths_to_hitran_files, 
        line_selection_file='', 
        ESelection=None, 
        bandSelection=None, 
        hitran_min_strength=None, 
        cdn_scale_fac=column_density_fractions, 
        custom_partition_sum_file='', 
        isotopolog=isotopolog_numbers, 
        waveSelection=wavelength_range, 
        wave_spec=wavelength_range, 
        R_grid=100000.0, 
        output='file', 
        output_directory=output_dir, 
        output_filename_prefix=slab_filename_prefix, 
        mode='overlap', 
        convolve_R=None, 
        write_index_file=True, 
        number_of_cores=number_of_cores_to_use, 
        verbose=False
    )
    
    print("Grid generation complete.")