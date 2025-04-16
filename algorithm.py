import numpy as np
import pandas as pd

def read_csv_file(file_path):
    # First read all columns to inspect them
    df = pd.read_csv(file_path)

        # Find the ECG column (case insensitive and handling quotes)
    ecg_col = None
    for col in df.columns:
        # Strip quotes and whitespace and check case-insensitive match
        clean_col = col.strip("'\"").strip()
        if clean_col.lower() == 'ecg':
            ecg_col = col
            break
        
    if ecg_col is not None:
        # Keep only the ECG column and reset the column name if needed
        df = df[[ecg_col]].copy()
        if ecg_col != 'ECG':
            df.columns = ['ECG']
        return df