#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for climate data processing

This script demonstrates how to use the climate_processor package
to process climate data and create visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import os

# Import our custom package
from climate_processor import ClimateDataProcessor

def main():
    """
    Main function to demonstrate the usage of ClimateDataProcessor.
    """
    # Create a processor instance with default parameters
    processor = ClimateDataProcessor(
        data_dir="data/climate_variables",
        lon_slice=slice(187.5, 192.5),
        lat_slice=slice(-12.5, -17.5),
        time_slice=slice("1958-01-01", "2024-12-31"),
        target_lons=np.array([187.5, 190.0, 192.5]),
        target_lats=np.array([-12.5, -15.0, -17.5])
    )
    
    # Process all variables
    print("Processing all climate variables...")
    processor.process_all_variables()
    
    # Validate grid consistency
    processor.validate_grid_consistency()
    
    # Save the processed data to a NetCDF file
    output_path = "AS_climate_var_ds.nc"
    processor.save_to_netcdf(output_path)
    
    print(f"Processed climate data saved to {output_path}")
    print("Available variables:")
    for var_name in processor.climate_data.keys():
        print(f"  - {var_name}")

if __name__ == "__main__":
    main()
