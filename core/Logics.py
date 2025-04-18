import streamlit as st
from .DisplayFunc import *
from .FilterLogic import *

class Logic:
    def __init__(self, variable):
        self.var = variable

    def process_data(self):
        if self.var is not None:
            self.loadDisplayData("data/samples_10sec.csv", action="hardcode")
            LOG_INFO("Load Data", self.var.dataECG, content="dataframe")
            if self.var.dataECG is not None:
                st.write("Performing DFT on the data...")
                # Perform DFT on the raw ECG data
                self.loadDFT(self.var.dataECG, absolute=True, transformType="DFT")

                self.applyFilter("LPF", self.var.dft, fcl=self.var.fcl, fch = self.var.fch,orde=4, fs=self.var.fs)                


        else: st.write("Please upload data first on the 'Data' page.")


    def loadDisplayData(self, file_path=None, data_title="Raw ECG", action="hardcode"):
        if action == "hardcode":
            # Hardcoded file path
            file_path = "data/samples_10sec.csv"
        else:
            # Use the uploaded file path
            file_path = st.file_uploader("Upload CSV file", type=["csv"])

        # Read the CSV file and extract the ECG column
        self.var.dataECG = read_csv_file(file_path)
        if file_path is not None:
            try:
                data = self.var.dataECG

                if data is not None:
                    tableDisplay(f"Display {data_title} Data", data)
                    plotLine(f"{data_title} Data", data)
                    self.var.dataECG = data
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        
        elif self.var.dataECG is not None: plotLine(f"{data_title} Data", self.var.dataECG)
        else: st.error("Please upload a file first.")

    def loadDFT(self, data_input, absolute=False, transformType="DFT"):
        if data_input is None:
            st.warning("No data to process.")
            return
        
        try:
            if transformType == "DFT":
                # Perform DFT on the data
                dft_result = DFT(data_input)
                plotDFT("DFT Result", dft_result, absolute)
                self.var.dft = dft_result
            elif transformType == "IDFT":
                # Perform IDFT on the DFT data
                idft_result = IDFT(data_input)
                plotDFT("IDFT Result", idft_result, absolute)
                self.var.dft = idft_result

        except Exception as e:
            st.error(f"Error during {transformType}: {e}")
            LOG_INFO("Error", self.var.dft, content="dataframe")
            self.var.dft = None

    def applyFilter(self, filter_type="LPF", data_input=None, fcl=None, fch=None, orde=None, fs=None):
        if data_input is None:
            st.warning("No data to filter.")
            return

        try:
            if filter_type == "LPF":
                h_i = LPF(fcl, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)
                self.var.filtered_data = filtered_data
                
                filtered_data = IDFT(filtered_data)
                plotDFT("LPF Result", filtered_data)
            elif filter_type == "HPF":
                h_i = HPF(fcl, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)
                self.var.filtered_data = filtered_data

                filtered_data = IDFT(filtered_data)
                plotDFT("LPF Result", filtered_data)
            elif filter_type == "BPF":
                h_i = BPF(fcl, fch, orde, fs)
                filtered_data = forward_filter(h_i, data_input)
                filtered_data = backward_filter(h_i, filtered_data)
                self.var.filtered_data = filtered_data

                filtered_data = IDFT(filtered_data)
                plotDFT("LPF Result", filtered_data)                
            else:
                st.error("Invalid filter type. Please choose LPF, HPF, or BPF.")
                self.var.filtered_data = None

        except Exception as e:
            st.error(f"Error during {filter_type}: {e}")
            self.var.filtered_data = None