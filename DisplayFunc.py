import streamlit as st
import pandas as pd
import numpy as np

def LOG_INFO(untukApa, data, content = None):
    print(f"{untukApa}, {data}")
    if content == None:
        st.info(f"{untukApa}, {data}")
    if content == "dataframe":
        st.dataframe(data)

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
        st.line_chart(data)

def buttonEvent(whatTodo, execute):
    st.button(f"{whatTodo}", on_click=execute)
        