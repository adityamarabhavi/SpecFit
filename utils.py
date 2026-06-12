"""
utils.py
Helper functions for managing configurations and grid axes across the MIRI fitting pipeline.
"""
import numpy as np
import yaml
import os

def load_config(path="config.yaml"):
    """Loads the pipeline configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file '{path}' not found. Please ensure it exists in the working directory.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_grid_axes_from_index(index_file_path):
    """
    Parses the prodimopy index.txt file to extract unique T and N grid axes.
    This ensures the interpolators always match the simulated grid dimensions.
    
    Returns:
        Tg_unique (1D array): Unique temperature values (K).
        logN_unique (1D array): Unique column density values (log10 cm^-2).
        n_T (int): Number of temperature points.
        n_N (int): Number of column density points.
    """
    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Grid index file '{index_file_path}' not found. Did you run the grid generation step?")
    data = np.loadtxt(index_file_path)
    Ng_all = data[:, 1]
    Tg_all = data[:, 2]
    Tg_unique = np.unique(Tg_all)
    Ng_unique = np.unique(Ng_all)
    logN_unique = np.log10(Ng_unique)
    return Tg_unique, logN_unique, len(Tg_unique), len(logN_unique)
