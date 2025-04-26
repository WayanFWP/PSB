import streamlit as st
import numpy as np
from .DisplayFunc import *
from .FilterLogic import *
from traceback import print_exc

class Logic:
    def __init__(self, variable):
        self.var = variable
        self.file_path = "data/samples_10sec.csv"
        self.fs = None # Default sampling frequency

    def process_data(self):
        # step 1
        file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        self.loadDisplayData(file_path=self.file_path)
        # self.loadDisplayData(file_path)
        # LOG_INFO("Load Data", self.var.dataECG, content="dataframe")
        # self.fs = np.max(self.var.dataECG)

        if self.var is not None:
            st.write("Performing DFT on the data...")
            # Perform DFT on the raw ECG data
            self.loadDFT(self.var.dataECG, absolute=True, transformType="DFT")

        
        if self.var.dataECG is not None:
            st.subheader("Plot filtered data...")
            fc_low = st.slider("Low Cutoff for LPF", min_value=1, max_value=500, value=100)
            orde = st.slider("Filter Order", min_value=1, max_value=10, value=4)
            fs_lowpass = 2*np.max(self.var.dataECG)  # Sampling frequency

            # st.write(f"Parameters: fc_low={fc_low}, orde={orde}, fs={fs}")

            amplitude = self.var.dataECG - np.mean(self.var.dataECG)
            self.applyFilter(filter_type="LPF", data_input=amplitude, absolute=True, fcl=fc_low, fch = 1, orde=1, fs=fs_lowpass)       
            # self.applyFilter(filter_type="LPF", data_input=self.var.dft, absolute=False, fcl=fc_low, fch = 1, orde=1, fs=self.fs)       

       
        else: st.write("Please upload data first on the 'Data' page.")


    def loadDisplayData(self, file_path=None, data_title="Raw ECG"):
        # Read the CSV file and extract the ECG column
        if file_path is not None:
            self.var.dataECG = read_csv_file(file_path)
            try:
                data = self.var.dataECG

                if data is not None:
                    plotLine(f"{data_title} Data", data)
                    tableDisplay(f"Display {data_title} Data", data)
                    self.var.dataECG = data
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        
        elif self.var.dataECG is not None: plotLine(f"{data_title} Data", self.var.dataECG)
        # else: st.error("Please upload a file first.")

    def loadDFT(self, data_input, absolute=False, transformType="DFT"):
        if data_input is None:
            st.warning("No data to process.")
            return
        
        try:
            if transformType == "DFT":
                # Perform DFT on the data
                dft_result = DFT(data_input)
                plotDFT("DFT Result", dft_result, absolute)
            elif transformType == "IDFT":
                # Perform IDFT on the DFT data
                idft_result = IDFT(data_input)
                plotDFT("IDFT Result", idft_result, absolute)

            self.var.dft = dft_result

        except Exception as e:
            st.error(f"Error during {transformType}: {e}")
            LOG_INFO("Error", self.var.dft, content="dataframe")
            self.var.dft = None

    def applyFilter(self, filter_type="LPF", data_input=None, absolute=False,fcl=None, fch=None, orde=None, fs=None):
        if data_input is None:
            st.warning("No data to filter.")
            return

        import pandas as pd
        if isinstance(data_input, pd.DataFrame):
            data_input = data_input.values.flatten()  # Convert DataFrame to 1D array

        # Validasi parameter
        if fcl is None or fcl <= 0:
            st.error("Invalid Low Cutoff Frequency (fcl). It must be greater than 0.")
            return
        if orde is None or orde <= 0:
            st.error("Invalid Filter Order (orde). It must be greater than 0.")
            return
        if fs is None or fs <= 0:
            st.error("Invalid Sampling Frequency (fs). It must be greater than 0.")
            return

        try:
            if filter_type == "LPF":
                h_i = LPF(fcl, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)
                
                proceed = DFT(filtered_data)
                plotDFT("LPF Result", proceed, absolute)
            elif filter_type == "HPF":
                h_i = HPF(fcl, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)

                proceed = DFT(filtered_data)
                plotDFT("HPF Result", proceed, absolute)
            elif filter_type == "BPF":
                h_i = BPF(fcl, fch, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)

                proceed = DFT(filtered_data)
                plotDFT("BPF Result", proceed, absolute)               

            self.var.filtered_data = np.real(filtered_data)

        except Exception as e:
            print_exc()
            st.error(f"Error during {filter_type}: {e}")
            self.var.filtered_data = None