import streamlit as st
import pandas as pd
from algorithm import *  # Assuming algorithm.py contains the necessary functions
from DisplayFunc import * # Assuming DisplayFunc.py contains the necessary functions

class App:
    def __init__(self):
        self.page = None
        # Initialize variable
        self.fs = 100 # Sampling frequency

        # Initialize session state for data if it doesn't exist
        if 'rawECG' not in st.session_state:
            st.session_state.rawECG = None
        if 'dft' not in st.session_state:
            st.session_state.dft = None            
        

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
            self.display_chart()

    def display_chart(self):
        if st.session_state is not None:
            self.loadDisplayData("data/samples_10sec.csv")
            self.plotData()
            
            # Perform DFT on the raw ECG data
            if st.session_state.rawECG is not None:
                st.write("Performing DFT on the data...")
                # Perform DFT on the raw ECG data
                self.loadDFT(st.session_state.rawECG, absolute=True, transformType="DFT")
        
        else:
            st.write("Please upload data first on the 'Data' page.")
    
    def loadDisplayData(self, file_path=None, data_title="Raw ECG"):
        # File uploader for CSV files
        # uploaded_file_path = st.file_uploader("Choose a CSV file", type="csv")

        # For testing purposes, we will use a hardcoded file path
        uploaded_file_path = file_path

        if uploaded_file_path is not None:
            try:
                # Display the uploaded file name
                data = read_csv_file(uploaded_file_path)

                if data is not None:
                    st.session_state.rawECG = data
                    tableDisplay(f"Display {data_title} Data", data)
                    plotLine(f"{data_title} Data", data)
                    st.session_state.rawECG = data
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
                    st.session_state.rawECG = None

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.rawECG = None

        # Display existing ECG data if already loaded
        elif st.session_state.rawECG is not None:
            plotLine("Raw ECG Data", st.session_state.rawECG)
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

if __name__ == "__main__":
    app = App()
    app.run()
