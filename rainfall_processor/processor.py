import pandas as pd
import os
from pathlib import Path

class RainfallProcessor:
    """
    A class to process rainfall data from CSV files and aggregate it to monthly totals.
    """
    
    def __init__(self, input_dir, output_dir="processed_data/mon_rainfall"):
        """
        Initialize the RainfallProcessor.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input CSV files
        output_dir : str
            Directory to save processed output files
        """
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True)
        self.processed_files = []
        self.error_files = []
    
    def process_file(self, file_path):
        """
        Process a single CSV file to calculate monthly rainfall totals.
        
        Parameters
        ----------
        file_path : Path
            Path to the CSV file to process
            
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['year_month'] = df['datetime'].dt.to_period('M')
            grouped = df.groupby('year_month')

            def sum_or_nan(series):
                return pd.NA if series.isna().any() else series.sum()

            # Apply only to the 'precip_in' column to avoid deprecation warning
            monthly_precip = grouped['precip_in'].apply(sum_or_nan).reset_index(name='monthly_total_precip_in')

            output_file = self.output_path / f"{file_path.stem}_monthly.csv"
            monthly_precip.to_csv(output_file, index=False)
            print(f"Processed: {file_path.name} -> {output_file.name}")
            self.processed_files.append(file_path.name)
            return True
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            self.error_files.append(file_path.name)
            return False
    
    def process_all_files(self):
        """
        Process all CSV files in the input directory.
        
        Returns
        -------
        tuple
            (number of successfully processed files, number of files with errors)
        """
        self.processed_files = []
        self.error_files = []
        
        for file in self.input_path.glob("*.csv"):
            self.process_file(file)
        
        return len(self.processed_files), len(self.error_files)
    
    def get_summary(self):
        """
        Get a summary of the processing results.
        
        Returns
        -------
        dict
            Dictionary with summary information
        """
        return {
            "total_files": len(self.processed_files) + len(self.error_files),
            "successful_files": len(self.processed_files),
            "error_files": len(self.error_files),
            "processed_file_names": self.processed_files,
            "error_file_names": self.error_files
        }