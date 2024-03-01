#!/usr/bin/env python
"""
pattern_corr.py

Pattern Correlation Functions

Contains primary functions for computing non-transient
and transient pattern correlations with spatial weighting.

Developers: Cameron Cummins (1), Kayla White (2)
Contacts: cameron.cummins@utexas.edu (1), kaylaw@utexas.edu (2)
3/1/24
"""
import numpy as np
import xarray


def pattern_corr(grid_a: np.ndarray, grid_b: np.ndarray, weights: np.ndarray=None, centered: bool=True) -> np.ndarray:
    """
    Computes Pearson Correlation coefficient between two spatial grids.
    May be centered (mean subtracted) or uncentered (see https://archive.ipcc.ch/ipccreports/tar/wg1/458.htm).

    A value of 1 indicates a perfectly linear relationship between grids A and B.
        (as A increases, B increases)
    A value of 0 indicates no linear relationship between the grids.
    A value of -1 indicates a perfectly negative linear relationship between the grids.
        (as A increases, B decreases)

    Parameters
    ----------
    grid_a : np.array
        Grid A to compare
    grid_b : np.array
        Grid B to compare
    weights : np.array, optional
        Spatial weights, such as latiduinal weights, to be used in mean
        calculation.
    centered : bool, optional
        Whether or not to subtract the means of grids A and B before
        computing the correlation coefficient.
    """
    if weights is None:
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


def spatial_weights(latitudes: np.ndarray, longitudes_size: int) -> np.ndarray:
    """
    Computes normalized latitudinal weights for a spatial grid.

    Parameters
    ----------
    latitudes : np.array
        Latiduinal values for grid coordinates.
    longitudes_size : np.array
        Size of longitudinal dimension for grid.
    """
    assert longitudes_size >= 0

    lat_weights = np.cos(np.deg2rad(latitudes))
    lat_lon_weights = np.tile(lat_weights, (longitudes_size, 1)).T
    lat_lon_weights = lat_lon_weights / np.nansum(lat_lon_weights)

    return lat_lon_weights


def xarray_pattern_corr(ds_a: xarray.DataArray, ds_b: xarray.DataArray, centered: bool, lat_dim: str="lat", lon_dim: str="lon", time_dim: str="time") -> xarray.DataArray:
    """
    Computes Pearson Correlation coefficient between two timeseries of spatial grids.
    May be centered (mean subtracted) or uncentered (see https://archive.ipcc.ch/ipccreports/tar/wg1/458.htm).

    A value of 1 indicates a perfectly linear relationship between grids A and B.
        (as A increases, B increases)
    A value of 0 indicates no linear relationship between the grids.
    A value of -1 indicates a perfectly negative linear relationship between the grids.
        (as A increases, B decreases)

    Parameters
    ----------
    ds_a : xarray.DataArray
        DataArray timeseries A to compare
    ds_b : xarray.DataArray
        DataArray timeseries B to compare
    centered : bool
        Whether or not to subtract the means of grids A and B before
        computing the correlation coefficient.
    lat_dim : str, optional
        Label for latitude dimension
    lon_dim : str, optional
        Label for longitude dimension
    time_dim : str, optional
        Label for time dimension
    """
    assert ds_a.shape == ds_b.shape
    weights = spatial_weights(ds_a[lat_dim].values, ds_a[lon_dim].size)
    combined_attrs = {}
    for key in ds_a.attrs:
        if key in ds_b.attrs:
            if ds_a.attrs[key] == ds_b.attrs[key]:
                combined_attrs[key] = ds_b.attrs[key]

    corr_ds = xarray.DataArray(data=np.zeros(ds_a[time_dim].size), coords={"time": ds_a[time_dim]}, attrs=combined_attrs)
    corr_ds.attrs["description"] = "transient pattern correlation with latitudinal weights applied"
    corr_ds.attrs["units"] = "[-1, 1]"

    for t in range(corr_ds[time_dim].size):
        corr_ds.values[t] = pattern_corr(ds_a.sel(time=corr_ds[time_dim].values[t]), ds_b.sel(time=corr_ds[time_dim].values[t]), weights, centered=centered)

    return corr_ds