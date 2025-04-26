import streamlit as st
import pandas as pd
import numpy as np

def LOG_INFO(untukApa, data, content = None):
    """
    Function to log information in Streamlit app.
    Args:
        untukApa (str): Description of the data.
        data (any): Data to be logged.
        content (str, optional): Type of content. Defaults to None.
    """
    print(f"{untukApa}, {data}")
    if content == None:
        st.info(f"{untukApa}, {data}")
    if content == "dataframe":
        st.dataframe(data)

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame with the ECG column.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing only the ECG column.
    """
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

def tableDisplay(title, data):
    st.subheader(title)
    st.dataframe(data)

def plotLine(title,data):
    st.subheader(title)
    st.line_chart(data)

def plotDFT(title, data, absolute=False):
    st.subheader(title)
    if absolute:
        st.line_chart(np.abs(data))
    else:
        # Plot real part for IDFT results (or other non-absolute cases)
        # Streamlit charts cannot directly handle complex numbers
        st.line_chart(np.real(data))

def plotFilter(title, data):
    st.subheader(title)
    st.line_chart(data)

# def buttonEvent(whatTodo, execute):
#     st.button(f"{whatTodo}", on_click=execute)

# def complexDataFrame(data):
#     """
#     Convert complex data to a DataFrame with real and imaginary parts.
#     Args:
#         data (np.ndarray): Complex data to be converted.
#     Returns:
#         pd.DataFrame: DataFrame with real, imaginary, and absolute values.
#     """
#     # Ensure data is a numpy array
#     if not isinstance(data, np.ndarray):
#         data = np.array(data)
#     # Convert complex data to a DataFrame with real and imaginary parts
#     df = pd.DataFrame({
#         'Real': data.real,
#         'Imaginary': data.imag,
#         'abs': np.abs(data)
#     })
#     return df

def averageBPM(data, fs):
    total_segment = len(data) // fs 
    if total_segment> 1:
        intervalEachSegment = np.diff(total_segment) / fs
        bpm = 60 / intervalEachSegment
        return bpm
    else:
        return 0     