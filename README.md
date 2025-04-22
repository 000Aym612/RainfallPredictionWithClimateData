# RainfallPredictionWithClimateData

A deep learning framework for predicting rainfall patterns using climate variables and digital elevation models (DEM). This project implements the LAND (Learning Across Non-uniform Domains) methodology to leverage both climate data and topographical information for improved rainfall predictions.

## Project Overview

This project aims to predict rainfall patterns by combining climate variables (temperature, humidity, wind, etc.) with topographical information from digital elevation models (DEM). The approach follows these key steps:

1. **Data Processing**: Processes climate variables, DEM data, and historical rainfall data
2. **Feature Engineering**: Creates local and regional DEM patches to capture topographical influences
3. **Deep Learning**: Implements a neural network with CNN layers for DEM processing and dense layers for climate variables
4. **Hyperparameter Tuning**: Optimizes model architecture and parameters using Keras Tuner
5. **Evaluation**: Assesses model performance with metrics like R², RMSE, and MAE

## Key Features

- **Climate Data Integration**: Processes and incorporates various climate variables
- **Multi-scale DEM Analysis**: Uses both local and regional DEM patches to capture topographical influences
- **Neural Network Architecture**: Combines CNN layers for DEM processing with dense layers for climate variables
- **Hyperparameter Optimization**: Implements systematic tuning using Keras Tuner with Hyperband algorithm
- **Comprehensive Evaluation**: Generates detailed metrics and visualizations of model performance

## Environment Setup

### Prerequisites
- Python 3.11.9 or newer
- pip (Python package installer)

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RainfallPredictionWithClimateData.git
   cd RainfallPredictionWithClimateData
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. You're all set! You can now run the project scripts.

### Deactivating the Environment
When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

## Usage

The project provides a unified interface through `main.py` with various actions:

### Data Pipeline

Process raw climate and DEM data to create the dataset for deep learning:

```bash
python main.py --action pipeline --h5_file output/rainfall_prediction_data.h5
```

### Data Preprocessing

Before running the pipeline, you need to prepare the climate and rainfall data:

#### Climate Data Preprocessing

The climate data file `processed_data/AS_climate_var_ds_updated.nc` is created by:

1. Downloading data files from https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
2. Processing the raw NetCDF files to extract relevant climate variables
3. Regridding to a consistent spatial resolution
4. Combining variables into a single NetCDF file

You can regenerate the climate data using:

```bash
python main.py --action regenerate_climate
```

#### Rainfall Data Preprocessing

The monthly rainfall data in `processed_data/mon_rainfall` is created by:

1. Collecting station-based rainfall measurements
2. Performing quality control and gap-filling
3. Converting to a gridded format through interpolation
4. Saving as monthly files with consistent naming conventions

You can process raw rainfall data into monthly aggregates using:

```bash
python main.py --action process_rainfall --raw_rainfall_dir data/raw_rainfall --mon_rainfall_dir processed_data/mon_rainfall
```

This script:
- Reads raw rainfall CSV files with datetime and precipitation columns
- Aggregates the data into monthly totals
- Handles missing values appropriately
- Saves the processed data as CSV files with a consistent format

These preprocessing steps are essential for ensuring data quality and consistency before feeding into the deep learning pipeline.

### Model Training

Train the deep learning model with default parameters:

```bash
python main.py --action train_model --h5_file output/rainfall_prediction_data.h5 --model_dir model_output --epochs 100
```

### Hyperparameter Tuning

Optimize model hyperparameters using Keras Tuner:

```bash
python main.py --action tune_hyperparams --h5_file output/rainfall_prediction_data.h5 --max_trials 20 --epochs 50 --min_epochs_per_trial 15
```

### Training with Best Hyperparameters

Train a model using the best hyperparameters found during tuning:

```bash
python main.py --action train_best_model --h5_file output/rainfall_prediction_data.h5 --tuner_dir tuner_results --best_model_dir best_model --epochs 1000
```

### Model Evaluation

Evaluate the trained model and generate performance visualizations:

```bash
python main.py --action evaluate_best_model --h5_file output/rainfall_prediction_data.h5 --best_model_dir best_model --eval_dir evaluation_results
```

### Rainfall Prediction

Generate rainfall predictions for specific dates:

```bash
python main.py --action predict --h5_file output/rainfall_prediction_data.h5 --model_dir best_model --output_dir predictions
```

## Model Architecture

The neural network architecture consists of:

1. **Climate Variables Branch**: Dense layers processing climate features
2. **Local DEM Branch**: CNN layers processing local topography (3x3 patches)
3. **Regional DEM Branch**: CNN layers processing regional topography (larger patches)
4. **Month Encoding Branch**: Dense layer processing temporal information
5. **Combined Layers**: Merged features from all branches for final prediction

## Performance

The optimized model achieves:
- **R²**: 0.695 (explains ~70% of rainfall variance)
- **RMSE**: 53.81 mm
- **MAE**: 16.00 mm

## Data Requirements

The project requires several data sources to function properly:

### Digital Elevation Model (DEM)
- File: `data/DEM/DEM_Tut1.tif`
- Format: GeoTIFF
- Resolution: 1km
- Source: USGS or similar topographical data provider

### Climate Variables
- File: `processed_data/AS_climate_var_ds_updated.nc`
- Format: NetCDF4
- Variables: Temperature, humidity, wind speed, pressure, etc.
- Source: ERA5 reanalysis data from Copernicus Climate Data Store

### Rainfall Measurements
- Directory: `processed_data/mon_rainfall`
- Format: CSV or NetCDF files with monthly data
- Variables: Precipitation amounts
- Source: Local weather stations or global precipitation datasets

### Station Locations
- File: `data/as_raingage_list2.csv`
- Format: CSV with latitude, longitude coordinates
- Purpose: Defines locations for rainfall measurements

These data files should be placed in their respective directories before running the pipeline.

## Project Structure

```
RainfallPredictionWithClimateData/
├── main.py                      # Main interface
├── src/
│   ├── utils/                   # Utility functions
│   ├── data_processing/         # Data processing modules
│   └── deep_learning/           # Deep learning model
├── scripts/
│   ├── rainfall_prediction_pipeline.py  # Data pipeline
│   ├── train_model.py                   # Model training
│   ├── predict_rainfall.py              # Prediction generation
│   ├── hyperparameter_tuning.py         # Hyperparameter optimization
│   ├── train_best_model.py              # Training with best parameters
│   └── evaluate_best_model.py           # Model evaluation
├── output/                      # Output data directory
├── model_output/                # Trained model directory
├── tuner_results/               # Hyperparameter tuning results
├── best_model/                  # Best model from tuning
└── evaluation_results/          # Evaluation metrics and visualizations