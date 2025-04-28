import streamlit as st
import numpy as np
from .DisplayFunc import *
from .FilterLogic import *
from .config import DEFAULT_FILTER_PARAMS, DEFAULT_THRESHOLDS, DEFAULT_SAMPLE_INTERVAL
from traceback import print_exc

class Logic:
    def __init__(self, variable):
        self.var = variable
        # self.file_path = "data/samples_10sec.csv"
        self.file_path = "data/samples_1minute.csv"
        self.filter_params = DEFAULT_FILTER_PARAMS
        self.thresholds = DEFAULT_THRESHOLDS
        self.segmentation = {}

    # data flow section
    def process_data(self):
        try:
            # step 1
            if self.var.dataECG is None:
                file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"])
                self.loadDisplayData(file_path)

            fs = self.samplingFrequency()
            duration = self.calculate_duration()
            if fs is None or duration is None:
                return
            st.write(f"fs: {fs}, Duration: {duration:.2f} seconds, data: {len(self.var.dataECG)}")

            # Perform DFT on the raw ECG data
            st.write("Performing DFT on the data...")
            # loadDFT(self.var.dataECG, absolute=True)

            if self.var.dataECG is not None:
                st.subheader("Plot filtered data...")

                self.get_filter_parameters()
                if self.filter_params:
                    self.filtered_data = self.applyFilter(
                        filter_type="LPF",
                        data_input=self.var.dataECG,
                        absolute=False,
                        fcl=self.filter_params["fc_l"],
                        fch=self.filter_params["fc_h"],
                        orde=self.filter_params["orde_filter"],
                        frequencySampling=fs,
                    )

            if self.var.filtered_data is not None:
                st.subheader("Detecting ECG peaks...")
                # Input thresholds for ECG peak detection
                self.get_thresholds()
                if self.thresholds:
                    self.segment_ecg_signal()
                    self.display_segmentation_results(duration)
            else:
                st.warning("Please upload data first on the 'Data' page.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    def samplingFrequency(self):
        if self.var.dataECG is None:
            st.warning("No data to process.")
            return None
        return 2 * np.max(np.abs(self.var.dataECG))

    def calculate_duration(self):
        if self.var.dataECG is None:
            st.warning("No data to process.")
            return None
        return len(self.var.dataECG) * DEFAULT_SAMPLE_INTERVAL

    def get_filter_parameters(self):
        """Gets filter parameters from user input."""
        filter_param_inputs = {
            "fc_l": st.text_input("Low Cutoff Frequency (Hz)", value=str(self.filter_params["fc_l"])),
            "fc_h": st.text_input("High Cutoff Frequency (Hz)", value=str(self.filter_params["fc_h"])),
            "orde_filter": st.number_input("Filter Order", min_value=1, max_value=10, value=self.filter_params["orde_filter"]),
        }

        try:
            self.filter_params = {
                key: float(value) if key != "orde_filter" else int(value)
                for key, value in filter_param_inputs.items()
            }
        except ValueError:
            st.error("Please enter numeric values for filter parameters.")
            return None

        st.write(
            f"fc_l: {self.filter_params['fc_l']}, fc_h: {self.filter_params['fc_h']}, orde: {self.filter_params['orde_filter']}"
        )
    
    def get_thresholds(self):
        """Gets thresholds for peak detection from user input."""
        threshold_inputs = {
            "P": st.text_input("Threshold for P peak detection (min max)", value=self.thresholds["P"]),
            "Q": st.text_input("Threshold for Q peak detection (min max)", value=self.thresholds["Q"]),
            "R": st.text_input("Threshold for R peak detection (min max)", value=self.thresholds["R"]),
            "S": st.text_input("Threshold for S peak detection (min max)", value=self.thresholds["S"]),
            "T": st.text_input("Threshold for T peak detection (min max)", value=self.thresholds["T"]),
        }

        try:
            self.thresholds = {
                key: list(map(float, value.split())) for key, value in threshold_inputs.items()
            }
        except ValueError:
            st.error("Please enter two numeric values separated by a space for each threshold.")
            return None      

    def segment_ecg_signal(self):
        """Segments the ECG signal to detect peaks."""
        if self.var.filtered_data is None:
            st.error("No filtered data to segment.")
            return

        try:
            self.segmentation = segment_ecg(
                self.var.filtered_data,
                threshold_p=self.thresholds["P"],
                threshold_q=self.thresholds["Q"],
                threshold_r=self.thresholds["R"],
                threshold_s=self.thresholds["S"],
                threshold_t=self.thresholds["T"],
            )
        except ValueError:
            st.error("Error during ECG segmentation. Check threshold values.")
            self.segmentation = {}
            return  

    def display_segmentation_results(self, duration):
        """Displays the ECG segmentation results."""
        if not self.segmentation:
            st.warning("No segmentation results to display.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(self.var.filtered_data)

        for Component in self.segmentation:
            plt.scatter(self.segmentation[Component]["address"], self.segmentation[Component]["value"], label=Component)

        for peak_type, peak_data in self.segmentation.items():
            if "address" in peak_data:
                heart_rate = calculate_heart_rate(peak_data["address"], duration)
                st.write(f"Signal detected {peak_type} : {int(heart_rate)} in minute")

        plt.legend()
        st.pyplot(plt)

        if "R" in self.segmentation and self.segmentation["R"]["address"]:
            heart_rate = int(calculate_heart_rate(self.segmentation["R"]["address"], duration))
            st.write(f"Heart Rate: {heart_rate} bpm")
            self.var.heart_rate = heart_rate
        else:
            st.write("No R peaks found in the filtered data.")
            self.var.heart_rate = None

    # load section
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
            print_exc()
            LOG_INFO("Error", self.var.dft, content="dataframe")
            self.var.dft = None

    def applyFilter(self, filter_type="LPF", data_input=None, absolute=False,fcl=None, fch=None, orde=None, frequencySampling=None):
        if data_input is None:
            st.warning("No data to filter.")
            return

        import pandas as pd
        if isinstance(data_input, pd.DataFrame):
            data_input = data_input.values.flatten()  # Convert DataFrame to 1D array
        
        if fch is None or orde is None or frequencySampling is None:
            st.error("Cutoff frequency, filter order, and sampling frequency must be specified.")
            return

        if orde <= 0:
            st.error("Filter order must be greater than 0.")
            return

        if frequencySampling <= 0:
            st.error("Sampling frequency must be greater than 0.")
            return
        try:
            if filter_type == "LPF":
                b, a = LPF(fcl, orde, frequencySampling)
                filtered_data = forward_backward_filter(b, a, data_input)
                plotLine("Filtered Data", filtered_data)
                if absolute:
                    hasil_data = DFT(filtered_data)
                    plotDFT("Filtered Data", hasil_data, absolute=True)

            elif filter_type == "HPF":
                b, a = HPF(fch, orde, frequencySampling)
                filtered_data = forward_backward_filter(b, a, data_input)
                plotLine("Filtered Data(Highpass)", filtered_data)

            elif filter_type == "BPF":
                b, a = BPF(fcl, fch, orde, frequencySampling)
                filtered_data = forward_backward_filter(b, a, data_input)
                plotLine("Filtered Data(Bandpass)", filtered_data)

            elif filter_type == "BSF":
                b, a = BSF(fcl, fch, orde, frequencySampling)
                filtered_data = forward_backward_filter(b, a, data_input)
                plotLine("Filtered Data(Bandstop)", filtered_data)

            self.var.filtered_data = np.real(filtered_data)
        except ValueError as e:
            st.error(f"Filter error: {e}")
        except Exception as e:
            print_exc()
            st.error(f"Error during {filter_type}: {e}")
            self.var.filtered_data = None