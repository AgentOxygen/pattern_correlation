"""
pattern_corr.py

Contains all relevant pattern correlation functions written using only the numpy and xarray packages.

"""
import numpy as np
import xarray


def pattern_corr(grid_a: np.ndarray, grid_b: np.ndarray, weights: np.ndarray=None, centered: bool=True) -> np.ndarray:
    """
    
    """
    if weights is None:
        # If weights are unspecified, assume uniform weighting.
        weights = np.ones(grid_a.shape) / grid_a.size

    assert grid_a.shape == grid_b.shape, "Arrays A and B need to be the same shape." 
    assert weights.shape == grid_a.shape, "Weights array needs to be same shape as array A/B."

    if centered:
        grid_mean_a = np.nansum(grid_a * weights)
        grid_mean_b = np.nansum(grid_b * weights)
        
        grid_a -= grid_mean_a
        grid_b -= grid_mean_b

    numerator = np.nansum(grid_a*grid_b*weights)

    grid_squared_a = (grid_a**2)*weights
    grid_squared_b = (grid_b**2)*weights

    denominator = np.sqrt(np.nansum(grid_squared_a)*np.nansum(grid_squared_b))

    return numerator / denominator
