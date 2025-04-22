"""
Deep learning model for rainfall prediction.

This module implements a neural network model similar to the LAND (Learning Across Non-uniform Domains)
approach described in research for Hawaii. The model takes both climate variables and DEM data as input
to predict rainfall.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd


class RainfallModel:
    """
    Neural network model for rainfall prediction using climate variables and DEM data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the rainfall prediction model.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with model parameters
        """
        # Default configuration
        self.config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'model_dir': 'models',
            'random_seed': 42
        }
        
        # Update with user-provided config
        if config is not None:
            self.config.update(config)
            
        # Set random seeds for reproducibility
        tf.random.set_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Initialize model
        self.model = None
        
    def build_model(self, input_shapes):
        """
        Build the neural network model.
        
        Parameters
        ----------
        input_shapes : dict
            Dictionary containing shapes of input data
            {
                'climate_vars': (num_climate_vars,),
                'local_dem': (height, width),
                'regional_dem': (height, width),
                'month_encoding': (12,)
            }
        """
        # Input layers
        climate_input = layers.Input(shape=input_shapes['climate_vars'], name='climate_vars')
        local_dem_input = layers.Input(shape=input_shapes['local_dem'], name='local_dem')
        regional_dem_input = layers.Input(shape=input_shapes['regional_dem'], name='regional_dem')
        month_input = layers.Input(shape=input_shapes['month_encoding'], name='month_encoding')
        
        # Process climate variables
        climate_features = layers.Dense(64, activation='relu')(climate_input)
        climate_features = layers.Dropout(0.3)(climate_features)
        climate_features = layers.Dense(32, activation='relu')(climate_features)
        
        # Process local DEM with CNN
        local_dem_features = layers.Reshape(input_shapes['local_dem'] + (1,))(local_dem_input)
        local_dem_features = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(local_dem_features)
        local_dem_features = layers.MaxPooling2D((2, 2), padding='same')(local_dem_features)
        local_dem_features = layers.Conv2D(8, (2, 2), activation='relu', padding='same')(local_dem_features)
        local_dem_features = layers.Flatten()(local_dem_features)
        local_dem_features = layers.Dense(32, activation='relu')(local_dem_features)
        
        # Process regional DEM with CNN
        regional_dem_features = layers.Reshape(input_shapes['regional_dem'] + (1,))(regional_dem_input)
        regional_dem_features = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(regional_dem_features)
        regional_dem_features = layers.MaxPooling2D((2, 2), padding='same')(regional_dem_features)
        regional_dem_features = layers.Conv2D(8, (2, 2), activation='relu', padding='same')(regional_dem_features)
        regional_dem_features = layers.Flatten()(regional_dem_features)
        regional_dem_features = layers.Dense(32, activation='relu')(regional_dem_features)
        
        # Process month encoding
        month_features = layers.Dense(8, activation='relu')(month_input)
        
        # Combine all features
        combined_features = layers.Concatenate()([
            climate_features, 
            local_dem_features, 
            regional_dem_features, 
            month_features
        ])
        
        # Dense layers for prediction
        x = layers.Dense(64, activation='relu')(combined_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, name='rainfall')(x)
        
        # Create model
        model = models.Model(
            inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def load_data(self, h5_path):
        """
        Load data from H5 file and prepare for training.
        
        Parameters
        ----------
        h5_path : str
            Path to H5 file containing the processed data
            
        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val) - Training and validation data
        """
        # Load data from H5 file
        with h5py.File(h5_path, 'r') as h5_file:
            # Get all date keys
            date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
            
            # Initialize lists to store data
            climate_vars_list = []
            local_patches_list = []
            regional_patches_list = []
            month_encodings_list = []
            rainfall_list = []
            
            # Extract data for each date
            for date_key in date_keys:
                date_group = h5_file[date_key]
                
                # Skip if any required data is missing
                if not all(key in date_group for key in 
                           ['climate_vars', 'local_patches', 'regional_patches', 'month_one_hot', 'rainfall']):
                    continue
                
                # Get number of grid points
                n_points = len(date_group['rainfall'][:])
                
                # Extract climate variables
                climate_data = []
                for var_name in date_group['climate_vars']:
                    var_data = date_group['climate_vars'][var_name][:]
                    climate_data.append(var_data)
                
                # Stack climate variables
                climate_vars = np.column_stack(climate_data)
                
                # Extract other data
                local_patches = date_group['local_patches'][:]
                regional_patches = date_group['regional_patches'][:]
                month_encoding = date_group['month_one_hot'][:]
                rainfall = date_group['rainfall'][:]
                
                # Repeat month encoding for each grid point
                month_encodings = np.tile(month_encoding, (n_points, 1))
                
                # Append to lists
                climate_vars_list.append(climate_vars)
                local_patches_list.append(local_patches)
                regional_patches_list.append(regional_patches)
                month_encodings_list.append(month_encodings)
                rainfall_list.append(rainfall)
            
            # Concatenate data
            climate_vars = np.vstack(climate_vars_list)
            local_patches = np.vstack(local_patches_list)
            regional_patches = np.vstack(regional_patches_list)
            month_encodings = np.vstack(month_encodings_list)
            rainfall = np.concatenate(rainfall_list)
            
            # Reshape rainfall to match model output
            rainfall = rainfall.reshape(-1, 1)
            
            # Split data into training and validation sets
            # Note: We're not splitting by time as mentioned in your requirements
            # Instead, we're doing a random split to ensure good representation
            indices = np.arange(len(rainfall))
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=self.config['validation_split'],
                random_state=self.config['random_seed']
            )
            
            # Create training and validation sets
            X_train = {
                'climate_vars': climate_vars[train_idx],
                'local_dem': local_patches[train_idx],
                'regional_dem': regional_patches[train_idx],
                'month_encoding': month_encodings[train_idx]
            }
            
            y_train = rainfall[train_idx]
            
            X_val = {
                'climate_vars': climate_vars[val_idx],
                'local_dem': local_patches[val_idx],
                'regional_dem': regional_patches[val_idx],
                'month_encoding': month_encodings[val_idx]
            }
            
            y_val = rainfall[val_idx]
            
            # Get input shapes for model building
            self.input_shapes = {
                'climate_vars': climate_vars.shape[1:],
                'local_dem': local_patches.shape[1:],
                'regional_dem': regional_patches.shape[1:],
                'month_encoding': month_encodings.shape[1:]
            }
            
            return X_train, y_train, X_val, y_val
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Parameters
        ----------
        X_train : dict
            Training features
        y_train : array
            Training targets
        X_val : dict, optional
            Validation features
        y_val : array, optional
            Validation targets
            
        Returns
        -------
        history : History object
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(self.input_shapes)
            
        # Create model directory if it doesn't exist
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['model_dir'], 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        X_test : dict
            Test features
        y_test : array
            Test targets
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate normalized metrics
        mean_rainfall = np.mean(y_test)
        rmae = mae / mean_rainfall
        rrmse = rmse / mean_rainfall
        
        # Calculate median absolute deviation
        mad = np.median(np.abs(y_test - y_pred))
        rmad = mad / mean_rainfall
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mad': mad,
            'r2': r2,
            'rmae': rmae,
            'rrmse': rrmse,
            'rmad': rmad,
            'mean_rainfall': mean_rainfall
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : dict
            Input features
            
        Returns
        -------
        array
            Predicted rainfall values
        """
        return self.model.predict(X)
    
    def save(self, filepath):
        """
        Save the model.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        self.model.save(filepath)
        
    def load(self, filepath):
        """
        Load a saved model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        self.model = models.load_model(filepath)
        
    def plot_history(self, history):
        """
        Plot training history.
        
        Parameters
        ----------
        history : History object
            Training history
            
        Returns
        -------
        fig : Figure
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.config['model_dir'], 'training_history.png'))
        
        return fig
    
    def plot_predictions(self, y_true, y_pred, title='Predicted vs Actual Rainfall'):
        """
        Plot predicted vs actual rainfall.
        
        Parameters
        ----------
        y_true : array
            Actual rainfall values
        y_pred : array
            Predicted rainfall values
        title : str, optional
            Plot title
            
        Returns
        -------
        fig : Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot scatter
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Plot perfect prediction line
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate metrics directly here instead of calling self.evaluate
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        # Add metrics to plot
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.90, f"RMSE = {rmse:.3f} mm", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.85, f"MAE = {mae:.3f} mm", transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel('Actual Rainfall (mm)')
        ax.set_ylabel('Predicted Rainfall (mm)')
        ax.set_title(title)
        
        # Save figure
        plt.savefig(os.path.join(self.config['model_dir'], 'prediction_scatter.png'))
        
        return fig
