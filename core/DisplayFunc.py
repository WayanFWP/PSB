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
    # Skip the second row only (unit description), but use the first row as header
    df = pd.read_csv(file_path, skiprows=[1])

    # Clean column names (strip quotes and whitespace)
    df.columns = df.columns.str.strip("'\" ").str.strip()

    # Convert the columns to numeric
    if 'sample interval' in df.columns:
        df['sample interval'] = pd.to_numeric(df['sample interval'], errors='coerce')
    elif 'sample #' in df.columns:
        df['sample #'] = pd.to_numeric(df['sample #'], errors='coerce')
    else:
        raise KeyError("Neither 'sample interval' nor 'sample #' columns are present in the CSV file.")
        
    df['ECG'] = pd.to_numeric(df['ECG'], errors='coerce')

    # Keep only the ECG column
    return df[['ECG']]


def tableDisplay(title, data):
    st.subheader(title)
    st.dataframe(data)

def plotLine(title,data, compare=None, title2=None):
    st.subheader(title)
    if compare is None:
        st.line_chart(data)
    else:
        plotdata = pd.DataFrame({
            title: data,
            title2: compare
        })
        st.line_chart(plotdata)

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