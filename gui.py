import streamlit as st
import pandas as pd
from algorithm import *  # Assuming algorithm.py contains the necessary functions
from DisplayFunc import * # Assuming DisplayFunc.py contains the necessary functions

class App:
    def __init__(self):
        self.page = None
        # Initialize variable
        self.fs = 100 # Sampling frequency
        self.amplitude = 1 # Amplitude

        # Initialize session state for data if it doesn't exist
        if 'dataECG' not in st.session_state:
            st.session_state.dataECG = None
        if 'dft' not in st.session_state:
            st.session_state.dft = None     
        if  'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None

    def process_data(self):
        if st.session_state is not None:

            # Load and display the data by hardcoding the file path
            self.loadDisplayData("data/samples_10sec.csv", action="hardcode")
            self.plotData()
            
            # Perform DFT on the raw ECG data
            if st.session_state.dataECG is not None:
                st.write("Performing DFT on the data...")
                # Perform DFT on the raw ECG data
                self.loadDFT(st.session_state.dataECG, absolute=True, transformType="DFT")

            # Perform IDFT on the DFT data
            # if st.session_state.dft is not None:
            #     try:
            #         st.write("Performing IDFT on the DFT data...")
            #         # Perform IDFT on the DFT data
            #         self.loadDFT(st.session_state.dft, absolute=False, transformType="IDFT")
            #     except Exception as e:
            #         st.dataframe(st.session_state.dft)
            #         st.session_state.dft = None

            # Apply filters to the DFT data
            if st.session_state.dft is not None:
                st.write("Applying filters to the DFT data...")
                # Apply LPF
                self.aplyFilter(st.session_state.dft, filter_type="LPF", fc=100, orde=5)
                # Apply BPF
                # self.aplyFilter(st.session_state.dft, filter_type="BPF", fc=100, orde=5)                    
        else:
            st.write("Please upload data first on the 'Data' page.")
    
    def loadDisplayData(self, file_path=None, data_title="Raw ECG", action="upload"):
        if action == "upload":
            # File uploader for CSV files
            uploaded_file_path = st.file_uploader("Choose a CSV file", type="csv")

        if action == "hardcode":
            # For testing purposes, we will use a hardcoded file path
            uploaded_file_path = file_path

        if uploaded_file_path is not None:
            try:
                # Display the uploaded file name
                data = read_csv_file(uploaded_file_path)

                if data is not None:
                    st.session_state.dataECG = data
                    tableDisplay(f"Display {data_title} Data", data)
                    plotLine(f"{data_title} Data", data)
                    st.session_state.dataECG = data
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
                    st.session_state.dataECG = None

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.dataECG = None

        # Display existing ECG data if already loaded
        elif st.session_state.dataECG is not None:
            plotLine("Raw ECG Data", st.session_state.dataECG)
        else: st.info("Upload a CSV file containing an 'ECG' column.")
    
    def plotData(self, data_input=None, plot_title="default title"):
        if data_input is not None:
            # Ensure data is suitable for line chart (e.g., numeric columns)
            try:
                # Attempt to display the line chart
                plotLine(plot_title, data_input)
            except Exception as e:
                st.error(f"Could not display chart. Ensure data is numeric and properly formatted. Error: {e}")
                st.dataframe(data_input) 

    def loadDFT(self, data_input=None, absolute=False, transformType="DFT"):
            if data_input is None:
                st.warning("Tidak ada data input untuk DFT/IDFT.")
                return

            try:
                # Correctly check the transformType parameter
                if transformType == "DFT":
                    # Make sure the DFT function is defined/imported and handles data_input correctly
                    data = DFT(data_input)
                    plotDFT("DFT Result", data, absolute)
                    st.session_state.dft = data

                elif transformType == "IDFT":
                     # Make sure the IDFT function is defined/imported
                    data = IDFT(data_input)
                    plotDFT("IDFT Result", data, absolute)
                    st.session_state.dft = data # Should this be idft? Or keep dft? Decide based on your needs.

            except Exception as e:
                st.error(f"An error occurred while calculating {transformType}: {e}")
                st.session_state.dft = None

    def aplyFilter(self, data_input=None , filter_type="LPF", fc=0, orde=0):
        if data_input is None:
            st.warning("Tidak ada data input untuk filter.")
            return

        try:
            if filter_type == "LPF":
                h_i = LPF(fc, orde)
                # apply the filter to the data 
                data = forward_filter(h_i, data_input)
                data = backward_filter(h_i, data)
                # plot the filtered data
                plotDFT("LPF Result", data)
                st.session_state.filtered_data = data

            elif filter_type == "HPF":
                h_i = HPF(fc, orde)
                # apply the filter to the data
                data = forward_filter(h_i, data_input)
                data = backward_filter(h_i, data)
                # plot the filtered data
                plotDFT("HPF Result", data)
                st.session_state.filtered_data = data

            elif filter_type == "BPF":
                h_i = BPF(fc, orde)
                # apply the filter to the data
                data = forward_filter(h_i, data_input)
                data = backward_filter(h_i, data)
                # plot the filtered data
                plotDFT("BPF Result", data)
                st.session_state.filtered_data = data
            else:
                st.error("Invalid filter type. Choose 'LPF', 'HPF', or 'BPF'.")

        except Exception as e:
            st.error(f"An error occurred while applying {filter_type}: {e}")

    def run(self):
        self.sidebar()
        self.content()
    
    def sidebar(self):
        st.sidebar.title("Navigation")
        self.page = st.sidebar.radio("Go to", ("Chart"))

    def content(self):
        if self.page == "Home":
            st.write("Welcome to the home page!")
        elif self.page == "Data":
            self.loadDisplayData()
        elif self.page == "Chart":
            self.process_data()


if __name__ == "__main__":
    app = App()
    app.run()
