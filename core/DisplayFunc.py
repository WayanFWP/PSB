import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

    # # Convert the columns to numeric
    # if 'sample interval' in df.columns:
    #     df['sample interval'] = pd.to_numeric(df['sample interval'], errors='coerce')
    # elif 'sample #' in df.columns:
    #     df['sample #'] = pd.to_numeric(df['sample #'], errors='coerce')
    # else:
    #     raise KeyError("Neither 'sample interval' nor 'sample #' columns are present in the CSV file.")
        
    df['ECG'] = pd.to_numeric(df['ECG'], errors='coerce')

    # Keep only the ECG column
    return df[['ECG']]


def tableDisplay(title, data):
    st.subheader(title)
    st.dataframe(data)

def plotLine(title, data, x_label="Time", y_label="Amplitude"):
    st.subheader(title)
    # Create Altair chart
    if isinstance(data, pd.DataFrame):
        # If data is a DataFrame, use the first column as x-axis
        x_col = data.columns[0]
        # Use remaining columns for y values
        chart = alt.Chart(data.reset_index() if data.index.name else data).mark_line().encode(
            x=alt.X(x_col, title=x_label),
            y=alt.Y(':Q', title=y_label),
            color='variable:N'
        ).transform_fold(
            [col for col in data.columns if col != x_col],
            ['variable', 'value']
        ).properties(
            width=800,
            height=400
        ).interactive()
    else:
        # If data is a Series, create a DataFrame with index as x-axis
        df = pd.DataFrame({y_label: data})
        df.reset_index(inplace=True)
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('index:Q', title=x_label),
            y=alt.Y(f'{y_label}:Q', title=y_label)
        ).properties(
            width=800,
            height=400
        ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

def plotData(titleChart="Data", data=None):
    """
    Function to plot data using Altair.
    Args:
        titleChart (str): Title of the chart.
        data (dict): Dictionary with format {label: data_array}
    """
    if data is None or not data:
        st.warning("No data to plot.")
        return
    
    # Temukan panjang maksimum dari semua array data
    max_len = max(len(signal) for signal in data.values())
    
    # Buat DataFrame
    df = pd.DataFrame()
    df["Time"] = np.arange(max_len)
    
    # Tambahkan setiap sinyal ke DataFrame
    for label, signal in data.items():
        # Pad sinyal yang lebih pendek dari max_len
        if len(signal) < max_len:
            padded_signal = np.zeros(max_len)
            padded_signal[:len(signal)] = signal
            df[label] = padded_signal
        else:
            df[label] = signal[:max_len]
    
    df_long = df.melt(id_vars=["Time"], var_name="Signal", value_name="Value")
    chart = alt.Chart(df_long).mark_line().encode(
        x='Time',
        y='Value',
        color='Signal',
        tooltip=['Time', 'Signal', 'Value']
    ).properties(
        title=titleChart,
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    return df

def plotDFTs(title="comparasion", dft_data=None, fs=None, absolute=False):
    if dft_data is None or not dft_data:
        st.warning("No DFT data to plot.")
        return
    
    max_len = max(len(signal) for signal in dft_data.values())
    df_dft = pd.DataFrame()

    max_idx = max_len // 2

    freq = np.arange(max_idx) * fs / max_len
    df_dft["Frequency"] = freq[:max_idx]

    for name , dft_array in dft_data.items():
        if absolute: 
            if len(dft_array) >= max_idx:
                df_dft[name] = np.abs(dft_array[:max_idx])
            else:
                padded_dft = np.zeros(max_len)
                padded_dft[:len(dft_array)] = dft_array
                df_dft[name] = np.abs(padded_dft[:max_idx])
        else:
            if len(dft_array) >= max_idx:
                df_dft[name] = np.real(dft_array[:max_idx])
            else:
                padded_dft = np.zeros(max_len)
                padded_dft[:len(dft_array)] = dft_array
                df_dft[name] = np.real(padded_dft[:max_idx])
    
    df_dft_long = df_dft.melt(id_vars=["Frequency"], var_name="Signal", value_name="Value")
    chart = alt.Chart(df_dft_long).mark_line().encode(
        x='Frequency',
        y='Value',
        color='Signal',
        tooltip=['Frequency', 'Signal', 'Value']
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    return df_dft