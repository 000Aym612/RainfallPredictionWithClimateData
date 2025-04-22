#!/usr/bin/env python3
"""
Evaluate the best model on test data and generate predictions.

This script loads the best model trained with optimal hyperparameters,
evaluates its performance on test data, and generates rainfall predictions
with visualizations.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from src.deep_learning.model import RainfallModel
from scripts.hyperparameter_tuning import load_data

def load_model(model_dir):
    """
    Load the best model from the specified directory.
    
    Parameters
    ----------
    model_dir : str
        Directory containing the model files
        
    Returns
    -------
    model : tf.keras.Model
        Loaded model
    """
    # Load model architecture
    with open(os.path.join(model_dir, 'model_architecture.json'), 'r') as f:
        model_json = f.read()
    
    # Create model from JSON
    model = tf.keras.models.model_from_json(model_json)
    
    # Load weights
    model.load_weights(os.path.join(model_dir, 'best_model.weights.h5'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_rainfall_map(rainfall, predictions, dates, output_dir, num_plots=5):
    """
    Plot maps of actual vs predicted rainfall for selected dates.
    
    Parameters
    ----------
    rainfall : array
        Actual rainfall values
    predictions : array
        Predicted rainfall values
    dates : array
        Date indices
    output_dir : str
        Directory to save plots
    num_plots : int, optional
        Number of dates to plot
    """
    # Create directory for maps
    maps_dir = os.path.join(output_dir, 'rainfall_maps')
    os.makedirs(maps_dir, exist_ok=True)
    
    # Randomly select dates to plot
    if len(dates) > num_plots:
        plot_indices = np.random.choice(len(dates), num_plots, replace=False)
    else:
        plot_indices = range(len(dates))
    
    for i in plot_indices:
        date_idx = dates[i]
        
        # Get actual and predicted rainfall for this date
        actual = rainfall[i]
        pred = predictions[i]
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot actual rainfall
        im1 = ax1.imshow(actual, cmap='Blues', vmin=0, vmax=max(np.max(actual), np.max(pred)))
        ax1.set_title(f'Actual Rainfall - Date {date_idx}')
        plt.colorbar(im1, ax=ax1, label='Rainfall (mm)')
        
        # Plot predicted rainfall
        im2 = ax2.imshow(pred, cmap='Blues', vmin=0, vmax=max(np.max(actual), np.max(pred)))
        ax2.set_title(f'Predicted Rainfall - Date {date_idx}')
        plt.colorbar(im2, ax=ax2, label='Rainfall (mm)')
        
        # Plot difference
        diff = pred - actual
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-max(abs(diff)), vmax=max(abs(diff)))
        ax3.set_title('Difference (Predicted - Actual)')
        plt.colorbar(im3, ax=ax3, label='Difference (mm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(maps_dir, f'rainfall_map_date_{date_idx}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate best model and generate predictions')
    parser.add_argument('--data', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--model_dir', type=str, default='best_model',
                        help='Directory with the best model')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data using the same function as in training
    print(f"Loading data from {args.data}...")
    _, _, X_val, y_val, _ = load_data(
        args.data, 
        validation_split=args.test_split,
        random_seed=args.random_seed
    )
    
    print(f"Loaded {len(y_val)} test samples")
    
    # Load the best model
    print(f"\nLoading best model from {args.model_dir}...")
    model = load_model(args.model_dir)
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    test_loss, test_mae = model.evaluate(X_val, y_val, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f} mm")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Test R²: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f} mm")
    print(f"Test MAE: {mae:.4f} mm")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test MAE: {test_mae:.4f} mm\n")
        f.write(f"Test R²: {r2:.4f}\n")
        f.write(f"Test RMSE: {rmse:.4f} mm\n")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([0, np.max(y_val)], [0, np.max(y_val)], 'r--')
    plt.xlabel('Actual Rainfall (mm)')
    plt.ylabel('Predicted Rainfall (mm)')
    plt.title('Predicted vs Actual Rainfall (Test Set)')
    plt.savefig(os.path.join(args.output_dir, 'test_predictions_vs_actual.png'))
    
    # Calculate error distribution
    errors = y_pred - y_val
    
    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (mm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(os.path.join(args.output_dir, 'error_distribution.png'))
    
    # Plot error vs actual rainfall
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Rainfall (mm)')
    plt.ylabel('Prediction Error (mm)')
    plt.title('Prediction Error vs Actual Rainfall')
    plt.savefig(os.path.join(args.output_dir, 'error_vs_actual.png'))
    
    print(f"\nEvaluation results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
