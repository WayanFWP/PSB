import streamlit as st
import numpy as np
from .DisplayFunc import *
from .FilterLogic import *
from traceback import print_exc

class Logic:
    def __init__(self, variable):
        self.var = variable
        # self.file_path = "data/samples_10sec.csv"
        self.file_path = "data/samples_1minute.csv"

    # data flow section
    def process_data(self):
        # step 1
        if self.var.dataECG is None:
            file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"])
            self.loadDisplayData(file_path)
        # elif self.var.dataECG is not None:
        #     loadDisplayData(self.var.dataECG)  # For testing, using a default file path
        # hardcoded
        # self.loadDisplayData(file_path=self.file_path)

        # Perform DFT on the raw ECG data
        # st.write("Performing DFT on the data...")
        # self.loadDFT(self.var.dataECG, absolute=True, transformType="DFT")
        if self.var.dataECG is not None:
            fs = 2 * np.max(np.abs(self.var.dataECG))  # Default sampling frequency using fs = 2 * max absolute value            
            duration = len(self.var.dataECG) * 0.01  # Duration in seconds why using 0.01? because the sample interval is 0.01 seconds
            st.write(f"fs: {fs},Duration: {duration:.2f} seconds, data: {len(self.var.dataECG)}")
            st.subheader("Plot filtered data...")

            filter_param = {
                "fc_l": st.text_input("Low Cutoff Frequency (Hz)", value="1"),
                "fc_h": st.text_input("High Cutoff Frequency (Hz)", value="0.5"),
                "orde_filter": st.number_input("Orde Frequency", max_value=10, min_value=1, value=2),
            }
            try:
                parsed_param = {
                    key: float(value) if key != "orde_filter" else int(value) for key, value in filter_param.items()
                }
            except ValueError:
                st.error("Please enter two numeric values separated by a space for each threshold.")

            amplitude = self.var.dataECG
            st.write(f"fc_l: {parsed_param['fc_l']}, fc_h: {parsed_param['fc_h']}, orde: {parsed_param['orde_filter']}")
            # self.applyFilter(filter_type="BPF", data_input=amplitude, absolute=False, fcl=fc_l, fch = fc_h, orde=orde_LPF, frequencySampling=fs)
            self.filtered_data = self.applyFilter(filter_type="LPF", data_input=amplitude, absolute=False, fcl=parsed_param["fc_l"], fch=parsed_param["fc_h"], orde=parsed_param["orde_filter"], frequencySampling=fs)     

        if self.var.filtered_data is not None:
            # Input thresholds for ECG peak detection
            thresholds = {
                "P": st.text_input("Threshold for P peak detection (min max)", value="0.11 0.2"),
                "Q": st.text_input("Threshold for Q peak detection (min max)", value="-0.28 -0.22"),
                "R": st.text_input("Threshold for R peak detection (min max)", value="1 1.5"),
                "S": st.text_input("Threshold for S peak detection (min max)", value="-0.9 -0.50"),
                "T": st.text_input("Threshold for T peak detection (min max)", value="0.23 0.38"),
            }

            try:
                # Parse thresholds
                parsed_thresholds = {
                    key: list(map(float, value.split())) for key, value in thresholds.items()
                }
                # Perform ECG segmentation
                segmentation = segment_ecg(
                    self.var.filtered_data,
                    threshold_p=parsed_thresholds["P"],
                    threshold_q=parsed_thresholds["Q"],
                    threshold_r=parsed_thresholds["R"],
                    threshold_s=parsed_thresholds["S"],
                    threshold_t=parsed_thresholds["T"],
                )
            except ValueError:
                st.error("Please enter two numeric values separated by a space for each threshold.")

            import matplotlib.pyplot as plt
                
            plt.figure(figsize=(10, 4))
            plt.plot(self.var.filtered_data)

            for komponen in segmentation:
                plt.scatter(segmentation[komponen]['lokasi'], segmentation[komponen]['nilai'], label=komponen)

            for peak_type, peak_data in segmentation.items():
                if 'lokasi' in peak_data:
                    heart_rate = calculate_heart_rate(peak_data['lokasi'], duration)
                    st.write(f"Signal detected {peak_type} : {heart_rate} in second")
            # st.write(f"P: {segmentation}, Q: {}, R: {}, S: {}, T: {}")
            plt.legend()
            st.pyplot(plt)   
            # Use st.pyplot to display the plot in Streamlit
            if 'R' in segmentation and segmentation['R']['lokasi']:
                heart_rate =calculate_heart_rate(segmentation['R']['lokasi'], duration)
                st.write(f"Heart Rate: {heart_rate} bpm")
            else: st.write("no R peaks found in the filtered data.")

                
        else: st.write("Please upload data first on the 'Data' page.")


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
                plotLine("Filtered Data(Lowpass)", filtered_data)
                if absolute:
                    hasil_data = DFT(filtered_data)
                    plotDFT("Filtered Data(Lowpass)", hasil_data, absolute=True)

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