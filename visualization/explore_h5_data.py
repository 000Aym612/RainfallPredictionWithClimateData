"""
Script to explore and visualize the contents of the rainfall prediction H5 data file.

This script allows you to:
1. View the structure of the H5 file
2. Extract and visualize specific components (DEM patches, climate variables, rainfall)
3. Explore data for specific dates
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

def view_h5_structure(h5_path):
    """
    Display the structure of the H5 file.
    """
    with h5py.File(h5_path, 'r') as h5_file:
        print("\nH5 File Structure:")
        print("==================")
        
        def print_group(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                if name != '':  # Skip root group attributes
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{indent}  Attribute: {attr_name} = {attr_value}")
            elif isinstance(obj, h5py.Dataset):
                shape_str = f"shape={obj.shape}, dtype={obj.dtype}"
                print(f"{indent}Dataset: {name} ({shape_str})")
                if len(obj.attrs) > 0:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{indent}  Attribute: {attr_name} = {attr_value}")
        
        h5_file.visititems(print_group)
        
        # Print metadata
        if 'metadata' in h5_file:
            print("\nMetadata:")
            print("=========")
            for attr_name, attr_value in h5_file['metadata'].attrs.items():
                print(f"  {attr_name}: {attr_value}")
        
        # Print available dates
        dates = [key for key in h5_file.keys() if key.startswith('date_')]
        print(f"\nAvailable Dates ({len(dates)}):")
        print("================")
        for i, date_key in enumerate(sorted(dates)):
            print(f"  {i+1}. {date_key}")
            if i >= 9:  # Show only first 10 dates
                print(f"  ... and {len(dates) - 10} more")
                break

def visualize_date_data(h5_path, date_idx=0, grid_point_idx=0):
    """
    Visualize data for a specific date and grid point.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    date_idx : int
        Index of the date to visualize (0-based)
    grid_point_idx : int
        Index of the grid point to visualize (0-based, default 0)
    """
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        if date_idx >= len(date_keys):
            print(f"Error: Date index {date_idx} out of range. Max index is {len(date_keys)-1}")
            return
        
        date_key = date_keys[date_idx]
        date_group = h5_file[date_key]
        date_str = date_key.split('_')[-1]
        
        print(f"\nVisualizing data for {date_key} ({date_str})")
        
        # Get metadata
        grid_size = 5  # Default
        if 'metadata' in h5_file and 'grid_size' in h5_file['metadata'].attrs:
            grid_size = h5_file['metadata'].attrs['grid_size']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Get grid points for visualization
        grid_points = date_group['grid_points'][:] if 'grid_points' in date_group else None
        
        # Check if grid_point_idx is valid
        num_grid_points = len(grid_points) if grid_points is not None else 0
        if grid_point_idx >= num_grid_points:
            print(f"Error: Grid point index {grid_point_idx} out of range. Max index is {num_grid_points-1}")
            grid_point_idx = 0
        
        # 1. Plot local DEM patch for the selected grid point
        if 'local_patches' in date_group:
            ax1 = fig.add_subplot(231)
            local_patches = date_group['local_patches'][:]
            
            # Get the patch for the selected grid point
            local_patch = local_patches[grid_point_idx]
            
            # Check if we have valid data in the patch
            if np.all(local_patch == 0):
                print(f"Warning: Local patch for grid point {grid_point_idx} is all zeros")
                # Try to find a non-zero patch
                for i in range(len(local_patches)):
                    if not np.all(local_patches[i] == 0):
                        print(f"Using local patch {i} instead of {grid_point_idx} (which was all zeros)")
                        local_patch = local_patches[i]
                        break
            
            # Display the patch
            im1 = ax1.imshow(local_patch, cmap='terrain')
            ax1.set_title(f'Local DEM Patch (12km) - Grid Point {grid_point_idx}')
            plt.colorbar(im1, ax=ax1)
            
            # Add grid lines to show the 3x3 structure
            ax1.set_xticks(np.arange(-0.5, local_patch.shape[1], 1), minor=True)
            ax1.set_yticks(np.arange(-0.5, local_patch.shape[0], 1), minor=True)
            ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # 2. Plot regional DEM patch for the selected grid point
        if 'regional_patches' in date_group:
            ax2 = fig.add_subplot(232)
            regional_patches = date_group['regional_patches'][:]
            
            # Get the patch for the selected grid point
            regional_patch = regional_patches[grid_point_idx]
            
            # Check if we have valid data in the patch
            if np.all(regional_patch == 0):
                print(f"Warning: Regional patch for grid point {grid_point_idx} is all zeros")
                # Try to find a non-zero patch
                for i in range(len(regional_patches)):
                    if not np.all(regional_patches[i] == 0):
                        print(f"Using regional patch {i} instead of {grid_point_idx} (which was all zeros)")
                        regional_patch = regional_patches[i]
                        break
            
            # Display the patch
            im2 = ax2.imshow(regional_patch, cmap='terrain')
            ax2.set_title(f'Regional DEM Patch (60km) - Grid Point {grid_point_idx}')
            plt.colorbar(im2, ax=ax2)
            
            # Add grid lines to show the 9x9 structure
            ax2.set_xticks(np.arange(-0.5, regional_patch.shape[1], 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, regional_patch.shape[0], 1), minor=True)
            ax2.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # 3. Plot month encoding
        if 'month_one_hot' in date_group:
            ax3 = fig.add_subplot(233)
            month_one_hot = date_group['month_one_hot'][:]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_idx = np.argmax(month_one_hot)
            
            ax3.bar(range(len(month_one_hot)), month_one_hot)
            ax3.set_xticks(range(len(month_one_hot)))
            ax3.set_xticklabels(month_names, rotation=45)
            ax3.set_title(f'Month: {month_names[month_idx]}')
        
        # 4. Plot rainfall
        if 'rainfall' in date_group:
            ax4 = fig.add_subplot(234)
            rainfall = date_group['rainfall'][:]
            
            # Check if rainfall is already a grid or needs reshaping
            if len(rainfall.shape) == 1:
                # Reshape to grid (assuming square grid)
                grid_size = int(np.sqrt(len(rainfall)))
                rainfall_grid = rainfall.reshape(grid_size, grid_size)
            else:
                rainfall_grid = rainfall
            
            # Check if rainfall has any non-zero values
            if np.all(rainfall_grid == 0):
                print("WARNING: All rainfall values are zero for this date.")
            
            im4 = ax4.imshow(rainfall_grid, cmap='Blues')
            ax4.set_title(f'Rainfall (Max: {np.max(rainfall_grid):.2f}, Mean: {np.mean(rainfall_grid):.2f})')
            plt.colorbar(im4, ax=ax4)
            
            # Mark the selected grid point
            if grid_point_idx < len(rainfall):
                row = grid_point_idx // grid_size
                col = grid_point_idx % grid_size
                ax4.plot(col, row, 'rx', markersize=10)
        
        # 5-6. Plot climate variables (first 2)
        if 'climate_vars' in date_group:
            climate_vars = date_group['climate_vars']
            var_names = list(climate_vars.keys())
            
            if len(var_names) > 0:
                ax5 = fig.add_subplot(235)
                var_name = var_names[0]
                var_data = climate_vars[var_name][:]
                
                # Check if data needs reshaping
                if len(var_data.shape) == 1:
                    # Reshape to grid (assuming square grid)
                    grid_size = int(np.sqrt(len(var_data)))
                    var_data = var_data.reshape(grid_size, grid_size)
                
                im5 = ax5.imshow(var_data, cmap='viridis')
                ax5.set_title(f'Climate: {var_name}')
                plt.colorbar(im5, ax=ax5)
                
                # Mark the selected grid point
                if grid_point_idx < len(var_data.flatten()):
                    row = grid_point_idx // grid_size
                    col = grid_point_idx % grid_size
                    ax5.plot(col, row, 'rx', markersize=10)
            
            if len(var_names) > 1:
                ax6 = fig.add_subplot(236)
                var_name = var_names[1]
                var_data = climate_vars[var_name][:]
                
                # Check if data needs reshaping
                if len(var_data.shape) == 1:
                    # Reshape to grid (assuming square grid)
                    grid_size = int(np.sqrt(len(var_data)))
                    var_data = var_data.reshape(grid_size, grid_size)
                
                im6 = ax6.imshow(var_data, cmap='viridis')
                ax6.set_title(f'Climate: {var_name}')
                plt.colorbar(im6, ax=ax6)
                
                # Mark the selected grid point
                if grid_point_idx < len(var_data.flatten()):
                    row = grid_point_idx // grid_size
                    col = grid_point_idx % grid_size
                    ax6.plot(col, row, 'rx', markersize=10)
        
        plt.tight_layout()
        plt.savefig(f"data_visualization_{date_str}_grid{grid_point_idx}.png")
        plt.show()
        print(f"Visualization saved to data_visualization_{date_str}_grid{grid_point_idx}.png")

def visualize_climate_variables(h5_path, date_idx=0):
    """
    Visualize all climate variables for a specific date.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    date_idx : int
        Index of the date to visualize (0-based)
    """
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        if date_idx >= len(date_keys):
            print(f"Error: Date index {date_idx} out of range. Max index is {len(date_keys)-1}")
            return
        
        date_key = date_keys[date_idx]
        date_group = h5_file[date_key]
        date_str = date_key.split('_')[-1]
        
        if 'climate_vars' not in date_group:
            print(f"No climate variables found for {date_key}")
            return
        
        climate_vars = date_group['climate_vars']
        var_names = list(climate_vars.keys())
        
        print(f"\nVisualizing climate variables for {date_key} ({date_str})")
        print(f"Available variables: {var_names}")
        
        # Determine grid layout
        n_vars = len(var_names)
        n_cols = min(4, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Plot each climate variable
        for i, var_name in enumerate(var_names):
            if i < len(axes):
                var_data = climate_vars[var_name][:]
                
                # Check if data needs reshaping
                if len(var_data.shape) == 1:
                    # Reshape to grid (assuming square grid)
                    grid_size = int(np.sqrt(len(var_data)))
                    var_data = var_data.reshape(grid_size, grid_size)
                
                im = axes[i].imshow(var_data, cmap='viridis')
                axes[i].set_title(var_name)
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"climate_vars_{date_str}.png")
        plt.show()
        print(f"Climate variables visualization saved to climate_vars_{date_str}.png")

def compare_dates(h5_path, date_indices):
    """
    Compare rainfall patterns across multiple dates.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    date_indices : list
        List of date indices to compare
    """
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        # Filter valid indices
        valid_indices = [idx for idx in date_indices if idx < len(date_keys)]
        if len(valid_indices) == 0:
            print("No valid date indices provided")
            return
        
        # Create figure with subplots
        n_dates = len(valid_indices)
        n_cols = min(3, n_dates)
        n_rows = (n_dates + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Plot rainfall for each date
        for i, date_idx in enumerate(valid_indices):
            if i < len(axes):
                date_key = date_keys[date_idx]
                date_group = h5_file[date_key]
                date_str = date_key.split('_')[-1]
                
                if 'rainfall' in date_group:
                    rainfall = date_group['rainfall'][:]
                    
                    # Check if rainfall is already a grid or needs reshaping
                    if len(rainfall.shape) == 1:
                        # Reshape to grid (assuming square grid)
                        grid_size = int(np.sqrt(len(rainfall)))
                        rainfall_grid = rainfall.reshape(grid_size, grid_size)
                    else:
                        rainfall_grid = rainfall
                    
                    im = axes[i].imshow(rainfall_grid, cmap='Blues')
                    axes[i].set_title(f'Rainfall: {date_str}')
                    plt.colorbar(im, ax=axes[i])
                else:
                    axes[i].text(0.5, 0.5, f"No rainfall data for {date_str}", 
                                ha='center', va='center')
                    axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(valid_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        date_strs = [date_keys[idx].split('_')[-1] for idx in valid_indices]
        filename = f"rainfall_comparison_{'_'.join(date_strs)}.png"
        plt.savefig(filename)
        plt.show()
        print(f"Rainfall comparison saved to {filename}")

def visualize_all_patches(h5_path, date_idx=0):
    """
    Visualize all DEM patches for a specific date.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    date_idx : int
        Index of the date to visualize (0-based)
    """
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        if date_idx >= len(date_keys):
            print(f"Error: Date index {date_idx} out of range. Max index is {len(date_keys)-1}")
            return
        
        date_key = date_keys[date_idx]
        date_group = h5_file[date_key]
        date_str = date_key.split('_')[-1]
        
        # Get local and regional patches
        if 'local_patches' not in date_group or 'regional_patches' not in date_group:
            print(f"Error: No patch data found for {date_key}")
            return
        
        local_patches = date_group['local_patches'][:]
        regional_patches = date_group['regional_patches'][:]
        grid_points = date_group['grid_points'][:] if 'grid_points' in date_group else None
        
        # Create a figure to display all patches
        num_points = len(local_patches)
        grid_size = int(np.sqrt(num_points))
        
        # Create a figure for local patches
        fig1, axes1 = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig1.suptitle(f'Local DEM Patches (12km) for {date_str}', fontsize=16)
        
        # Create a figure for regional patches
        fig2, axes2 = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig2.suptitle(f'Regional DEM Patches (60km) for {date_str}', fontsize=16)
        
        # Plot each patch
        for i in range(num_points):
            row = i // grid_size
            col = i % grid_size
            
            # Plot local patch
            ax1 = axes1[row, col]
            local_patch = local_patches[i]
            im1 = ax1.imshow(local_patch, cmap='terrain')
            ax1.set_title(f'Point {i}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Add grid lines to show the 3x3 structure
            ax1.set_xticks(np.arange(-0.5, local_patch.shape[1], 1), minor=True)
            ax1.set_yticks(np.arange(-0.5, local_patch.shape[0], 1), minor=True)
            ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
            
            # Plot regional patch
            ax2 = axes2[row, col]
            regional_patch = regional_patches[i]
            im2 = ax2.imshow(regional_patch, cmap='terrain')
            ax2.set_title(f'Point {i}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Add grid lines to show the 9x9 structure
            ax2.set_xticks(np.arange(-0.5, regional_patch.shape[1], 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, regional_patch.shape[0], 1), minor=True)
            ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Add colorbars
        fig1.subplots_adjust(right=0.9)
        cbar_ax1 = fig1.add_axes([0.92, 0.15, 0.02, 0.7])
        fig1.colorbar(im1, cax=cbar_ax1)
        
        fig2.subplots_adjust(right=0.9)
        cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
        fig2.colorbar(im2, cax=cbar_ax2)
        
        # Save figures
        fig1.savefig(f"local_patches_{date_str}.png")
        fig2.savefig(f"regional_patches_{date_str}.png")
        
        plt.show()
        print(f"Visualizations saved to local_patches_{date_str}.png and regional_patches_{date_str}.png")

def visualize_single_point(h5_path, date_idx=0, grid_point_idx=12):
    """
    Create a focused visualization for a single grid point.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    date_idx : int
        Index of the date to visualize (0-based)
    grid_point_idx : int
        Index of the grid point to visualize (0-based)
    """
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        if date_idx >= len(date_keys):
            print(f"Error: Date index {date_idx} out of range. Max index is {len(date_keys)-1}")
            return
        
        date_key = date_keys[date_idx]
        date_group = h5_file[date_key]
        date_str = date_key.split('_')[-1]
        
        # Get metadata
        grid_size = 5  # Default
        if 'metadata' in h5_file and 'grid_size' in h5_file['metadata'].attrs:
            grid_size = h5_file['metadata'].attrs['grid_size']
        
        # Check if grid_point_idx is valid
        grid_points = date_group['grid_points'][:] if 'grid_points' in date_group else None
        num_grid_points = len(grid_points) if grid_points is not None else 0
        
        if grid_point_idx >= num_grid_points:
            print(f"Error: Grid point index {grid_point_idx} out of range. Max index is {num_grid_points-1}")
            grid_point_idx = 0
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Create a grid layout
        gs = fig.add_gridspec(2, 3)
        
        # 1. Plot local DEM patch
        if 'local_patches' in date_group:
            ax1 = fig.add_subplot(gs[0, 0])
            local_patches = date_group['local_patches'][:]
            local_patch = local_patches[grid_point_idx]
            
            im1 = ax1.imshow(local_patch, cmap='terrain')
            ax1.set_title(f'Local DEM Patch (12km)')
            plt.colorbar(im1, ax=ax1)
            
            # Add grid lines
            ax1.set_xticks(np.arange(-0.5, local_patch.shape[1], 1), minor=True)
            ax1.set_yticks(np.arange(-0.5, local_patch.shape[0], 1), minor=True)
            ax1.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # 2. Plot regional DEM patch
        if 'regional_patches' in date_group:
            ax2 = fig.add_subplot(gs[0, 1])
            regional_patches = date_group['regional_patches'][:]
            regional_patch = regional_patches[grid_point_idx]
            
            im2 = ax2.imshow(regional_patch, cmap='terrain')
            ax2.set_title(f'Regional DEM Patch (60km)')
            plt.colorbar(im2, ax=ax2)
            
            # Add grid lines
            ax2.set_xticks(np.arange(-0.5, regional_patch.shape[1], 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, regional_patch.shape[0], 1), minor=True)
            ax2.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # 3. Plot month encoding
        if 'month_one_hot' in date_group:
            ax3 = fig.add_subplot(gs[0, 2])
            month_one_hot = date_group['month_one_hot'][:]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_idx = np.argmax(month_one_hot)
            
            ax3.bar(range(len(month_one_hot)), month_one_hot)
            ax3.set_xticks(range(len(month_one_hot)))
            ax3.set_xticklabels(month_names, rotation=45)
            ax3.set_title(f'Month: {month_names[month_idx]}')
        
        # 4. Plot rainfall grid with the selected point highlighted
        if 'rainfall' in date_group:
            ax4 = fig.add_subplot(gs[1, 0])
            rainfall = date_group['rainfall'][:]
            
            # Reshape to grid
            if len(rainfall.shape) == 1:
                grid_size = int(np.sqrt(len(rainfall)))
                rainfall_grid = rainfall.reshape(grid_size, grid_size)
            else:
                rainfall_grid = rainfall
            
            im4 = ax4.imshow(rainfall_grid, cmap='Blues')
            ax4.set_title(f'Rainfall Grid (inches)')
            plt.colorbar(im4, ax=ax4)
            
            # Mark the selected grid point
            row = grid_point_idx // grid_size
            col = grid_point_idx % grid_size
            ax4.plot(col, row, 'rx', markersize=10)
            
            # Add text with the actual rainfall value
            rainfall_value = rainfall[grid_point_idx]
            ax4.text(col, row + 0.3, f"{rainfall_value:.2f}", 
                    color='red', fontweight='bold', ha='center')
        
        # 5-6. Plot two key climate variables
        if 'climate_vars' in date_group:
            climate_vars = date_group['climate_vars']
            var_names = list(climate_vars.keys())
            
            # Select two interesting climate variables
            selected_vars = ['air_2m', 'pr_wtr'] if all(v in var_names for v in ['air_2m', 'pr_wtr']) else var_names[:2]
            
            for i, var_name in enumerate(selected_vars[:2]):
                ax = fig.add_subplot(gs[1, i+1])
                var_data = climate_vars[var_name][:]
                
                # Get the value for this grid point
                point_value = var_data[grid_point_idx]
                
                # Reshape to grid for visualization
                if len(var_data.shape) == 1:
                    var_grid = var_data.reshape(grid_size, grid_size)
                else:
                    var_grid = var_data
                
                im = ax.imshow(var_grid, cmap='viridis')
                ax.set_title(f'Climate: {var_name}')
                plt.colorbar(im, ax=ax)
                
                # Mark the selected grid point
                row = grid_point_idx // grid_size
                col = grid_point_idx % grid_size
                ax.plot(col, row, 'rx', markersize=10)
                
                # Add text with the actual value
                ax.text(col, row + 0.3, f"{point_value:.2f}", 
                       color='red', fontweight='bold', ha='center')
        
        # Add a title for the entire figure
        fig.suptitle(f'Data for Grid Point {grid_point_idx} on {date_str}', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
        
        # Save the figure
        output_path = f"single_point_{date_str}_grid{grid_point_idx}.png"
        plt.savefig(output_path)
        plt.show()
        print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Explore and visualize rainfall prediction H5 data')
    parser.add_argument('--file', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file')
    parser.add_argument('--action', type=str, default='structure',
                        choices=['structure', 'visualize', 'climate', 'compare', 'grid_view', 'all_patches', 'single_point'],
                        help='Action to perform')
    parser.add_argument('--date', type=int, default=0,
                        help='Date index to visualize (0-based)')
    parser.add_argument('--compare', type=int, nargs='+',
                        help='Date indices to compare')
    parser.add_argument('--grid_point', type=int, default=12,
                        help='Grid point index to visualize (0-based)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return
    
    if args.action == 'structure':
        view_h5_structure(args.file)
    elif args.action == 'visualize':
        visualize_date_data(args.file, args.date, args.grid_point)
    elif args.action == 'climate':
        visualize_climate_variables(args.file, args.date)
    elif args.action == 'compare':
        if args.compare is None:
            print("Error: Must specify date indices to compare with --compare")
            return
        compare_dates(args.file, args.compare)
    elif args.action == 'grid_view':
        # Add a new action to visualize all grid points for a date
        print("Grid view not implemented yet")
    elif args.action == 'all_patches':
        visualize_all_patches(args.file, args.date)
    elif args.action == 'single_point':
        visualize_single_point(args.file, args.date, args.grid_point)

if __name__ == "__main__":
    main()
