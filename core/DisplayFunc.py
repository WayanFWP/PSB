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
    df["Samples"] = np.arange(max_len)
    
    # Tambahkan setiap sinyal ke DataFrame
    for label, signal in data.items():
        # Pad sinyal yang lebih pendek dari max_len
        if len(signal) < max_len:
            padded_signal = np.zeros(max_len)
            padded_signal[:len(signal)] = signal
            df[label] = padded_signal
        else:
            df[label] = signal[:max_len]
    
    df_long = df.melt(id_vars=["Samples"], var_name="Signal", value_name="Amplitude")
    chart = alt.Chart(df_long).mark_line().encode(
        x='Samples',
        y='Amplitude',
        color='Signal',
        tooltip=['Samples', 'Signal', 'Amplitude']
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

def visualize_pqrst_altair(ecg_signal, segments):
    if not segments or ecg_signal is None:
        st.warning("No segments to visualize")
        return
    
    # Create a time array
    time = np.arange(len(ecg_signal)) * 0.01
    
    # Create a DataFrame for the ECG signal
    df_ecg = pd.DataFrame({
        'Time': time,
        'Amplitude': ecg_signal,
        'Type': 'ECG Signal'
    })
    
    # Create DataFrames for each wave type
    wave_data = []
    wave_colors = {
        "P": "#4CAF50",  # Green
        "Q": "#F44336",  # Red
        "R": "#2196F3",  # Blue
        "S": "#FF9800",  # Orange
        "T": "#9C27B0"   # Purple
    }
    
    for wave_type, indices in segments.items():
        if indices:  
            for idx in indices:
                wave_data.append({
                    'Time': time[idx],
                    'Amplitude': ecg_signal[idx],
                    'Type': wave_type
                })
    
    df_waves = pd.DataFrame(wave_data)
    
    # Create the ECG line chart
    ecg_chart = alt.Chart(df_ecg).mark_line(color='gray').encode(
        x=alt.X('Time:Q', title='samples'),
        y=alt.Y('Amplitude:Q', title='Amplitude'),
        tooltip=['Time', 'Amplitude']
    )
    
    # Create scatter plots for each wave type
    wave_chart = alt.Chart(df_waves).mark_circle(size=100).encode(
        x=alt.X('Time:Q'),
        y=alt.Y('Amplitude:Q'),
        color=alt.Color('Type:N', scale=alt.Scale(domain=list(wave_colors.keys()), 
                                                  range=list(wave_colors.values()))),
        tooltip=['Type', 'Time', 'Amplitude']
    )
    
    # Combine the charts
    chart = (ecg_chart + wave_chart).properties(
        title='ECG Signal with PQRST Segmentation',
        width=800,
        height=400
    ).interactive()
    
    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

def visualize_heart_rate(mav, r_peaks, r_values, threshold):
    """
    Visualize heart rate data with R-peaks and threshold
    
    Parameters:
    -----------
    mav : ndarray
        Moving average values
    r_peaks : list
        Indices of detected R-peaks
    r_values : list
        Amplitude values at R-peaks
    threshold : float
        Threshold used for peak detection
    """
    # Create visualization dataframes
    mav_df = pd.DataFrame({"Sample": np.arange(len(mav)), "Value": mav})
    peak_df = pd.DataFrame({"Sample": r_peaks, "Value": r_values})
    threshold_df = pd.DataFrame({'threshold': [threshold] * len(mav_df)})
    
    # Create chart
    base_chart = alt.Chart(mav_df).mark_line().encode(
        x=alt.X('Sample:Q', title='Sample'),
        y=alt.Y('Value:Q', title='Amplitude')
    )

    threshold_line = alt.Chart(threshold_df).mark_rule(color='red', strokeDash=[3, 3]).encode(
        y=alt.Y('threshold:Q'), 
        tooltip=alt.value(f"Threshold: {threshold:.4f}")
    )

    peak_chart = alt.Chart(peak_df).mark_circle(color='red', size=100).encode(
        x=alt.X('Sample:Q', title='Sample'),
        y=alt.Y('Value:Q'),
        tooltip=['Sample:Q', 'Value:Q']
    )

    chart = (base_chart + threshold_line + peak_chart).properties(
        title="MAV with R-peaks",
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)