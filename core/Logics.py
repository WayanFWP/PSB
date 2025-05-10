import streamlit as st
import numpy as np
from .DisplayFunc import *
from .FilterLogic import *
from .config import DEFAULT_SAMPLE_INTERVAL
from traceback import print_exc

class Logic:
    def __init__(self, variable):
        self.var = variable
        # self.file_path = "data/samples_10sec.csv"
        # self.file_path = "data/samples_1minute.csv"
        self.file_path = "data/samples (7).csv"
        self.segmentation = {}
        self.dataSignal = pd.DataFrame()
        self.filter_result = {}

    # data flow section
    def process_data(self):
        try:
            # Check if the user has uploaded a file
            file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"])
            if file_path is not None:
                self.loadDisplayData(file_path, data_log = True) # Use the uploaded file
            else:
                self.loadDisplayData(self.file_path) # Use the default file path if no file is uploaded


            if self.var.dataECG is not None:
                fs = 2 * np.max(np.abs(self.var.dataECG))
                duration = len(self.var.dataECG) * DEFAULT_SAMPLE_INTERVAL

            if fs is None or duration is None:
                return

            self.merged_data_plot("Raw ECG",["Raw ECG"])
            st.write(f"fs: {fs}, Duration: {duration:.2f} seconds, data: {len(self.var.dataECG)}")

            # Perform DFT on the raw ECG data
            st.write("Performing DFT on the data...")
            # self.loadDFT(self.var.dataECG, absolute=True)
            # self.dft_plot("DFT Raw", ["Raw ECG"], absolute=True, fs=fs)

            if self.var.dataECG is not None:
                st.subheader("Plot filtered data...")
                fcl = np.max(np.abs(self.var.dataECG)) * 0.45

                self.get_filter_parameters(fcl, 0, 2)
                if self.filter_params:
                    self.filtered_data = self.applyFilter(
                        filter_type="LPF",
                        data_input=self.var.dataECG,
                        plot=False,
                        fcl=self.filter_params["fc_l"],
                        fch=self.filter_params["fc_h"],
                        orde=self.filter_params["orde_filter"],
                        frequencySampling=fs
                    )
                self.dataSignal["LPF"] = self.filtered_data
                self.merged_data_plot()
                # self.dft_plot("DFT Raw vs LPF", ["Raw ECG", "LPF"], absolute=True, fs=fs)

                fcl_bpf = np.max(np.abs(self.var.filtered_data)) * 0.65
                fch_bpf = np.max(np.abs(self.var.filtered_data)) * 0.3

                self.get_filter_parameters(fcl_bpf,fch_bpf,4)
                if self.filter_params:
                    self.filtered_data = self.applyFilter(
                        filter_type="BPF",
                        data_input=self.var.filtered_data,
                        plot=False,
                        fcl=self.filter_params["fc_l"],
                        fch=self.filter_params["fc_h"],
                        orde=self.filter_params["orde_filter"],
                        frequencySampling=fs
                    )
                self.dataSignal["BPF"] = self.filtered_data
                self.merged_data_plot("BPF",["BPF"])
                # self.dft_plot("DFT BPF", ["BPF"], absolute=True, fs=fs)


            if self.var.filtered_data is not None:
                # st.write("Performing MAV")
                self.applyMAV(self.var.filtered_data, window_size=10)

                st.subheader("ECG Segmentation")

            else:
                st.warning("Please upload data first on the 'Data' page.")
        except Exception as e:
            print_exc()
            st.error(f"An error occurred: {e}")

    def samplingFrequency(self):
        if self.var.dataECG is None:
            st.warning("No data to process.")
            return None
        return 2 * np.max(np.abs(self.var.dataECG))

    def get_filter_parameters(self, fc_l, fc_h, orde):
        """Gets filter parameters from user input."""
        filter_param_inputs = {
            "parameter": st.text_input("Filter Parameters (Low-Cutoff High-Cutoff Order)", value=f"{fc_l} {fc_h} {orde}"),
        }

        try:
            # Split the parameter string by spaces and convert to appropriate types
            param_values = filter_param_inputs["parameter"].split()
            if len(param_values) != 3:
                st.error("Please enter three values separated by spaces: Low Cutoff, High Cutoff, and Order")
                return None
                
            self.filter_params = {
                "fc_l": float(param_values[0]),
                "fc_h": float(param_values[1]),
                "orde_filter": int(param_values[2])
            }
            
            st.write(
                f"fc_l: {self.filter_params['fc_l']}, fc_h: {self.filter_params['fc_h']}, orde: {self.filter_params['orde_filter']}"
            )
            
        except ValueError:
            st.error("Please enter valid numeric values: two floats for cutoff frequencies and an integer for order")
            return None
    
    def get_threshold(self, data=None):
        return np.abs(data) * 0.75

    def applyFilter(self, filter_type="LPF", data_input=None, plot=False,fcl=None, fch=None, orde=None, frequencySampling=None):
        if data_input is None:
            st.warning("No data to filter.")
            return

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
                if plot:
                    self.dft_plot("DFT LPF", ["LPF"], absolute=True, fs=frequencySampling)
            elif filter_type == "HPF":
                b, a = HPF(fch, orde, frequencySampling)
                filtered_data = forward_backward_filter(b, a, data_input)                
                if plot:
                    self.dft_plot("DFT HPF", ["HPF"], absolute=True, fs=frequencySampling)

            elif filter_type == "BPF":
                filtered_data = BPF(data_input, fcl, fch, orde, frequencySampling)
                if plot:
                    self.dft_plot("DFT BPF", ["BPF"], absolute=True, fs=frequencySampling)

            self.var.filtered_data = np.real(filtered_data)
            # Store the filtered data label in the session state
            if not hasattr(self, 'filter_result'):
                self.filter_result ={}
            
            self.filter_result[filter_type] = np.real(filtered_data)

        except ValueError as e:
            st.error(f"Filter error: {e}")
        except Exception as e:
            print_exc()
            st.error(f"Error during {filter_type}: {e}")
            self.var.filtered_data = None

    def applyMAV(self, data_input=None, window_size=10):
        if data_input is None:
            st.warning("No data to process.")
            return
        
        data_input = np.abs(data_input)
        plotLine("absolute", data_input)

        try:
            mav = moving_average(data_input, window_size)
            plotLine("MAV Result", mav)
            threshold = np.mean(mav) * 0.85
            st.write(f"Threshold: {threshold:.2f}")

        except Exception as e:
            st.error(f"Error during MAV calculation: {e}")

    # load section
    def loadDisplayData(self, file_path=None, data_title="Raw ECG", data_log = False):
        # Read the CSV file and extract the ECG column
        if file_path is not None:
            self.var.dataECG = read_csv_file(file_path)
            try:
                data = self.var.dataECG
                if data is not None:
                    self.var.dataECG = data
                    if data_log:
                        tableDisplay(f"Display {data_title} Data", data)
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        
        elif self.var.dataECG is not None: plotLine(f"{data_title} Data", self.var.dataECG)

    def merged_data_plot(self,title ="data", filters_to_show=None):
        if not hasattr(self, 'filter_result'):
            self.filter_result = {}
        
        # Make a raw ECG data available if not already present
        if 'Raw ECG' not in self.filter_result and self.var.dataECG is not None:
            self.filter_result['Raw ECG'] = self.var.dataECG

        # make a dictionary to store the data to be plotted        
        data_to_plot = {}

        # if filters_to_show is None, show all filters
        if filters_to_show is None:
            data_to_plot = self.filter_result
        else:
            # Check if the specified filters are in the filter_result
            for filter_name in filters_to_show:
                if filter_name in self.filter_result:
                    data_to_plot[filter_name] = self.filter_result[filter_name]
                else:
                    st.warning(f"Filter '{filter_name}' tidak ditemukan.")
        
        # Plot the data using the plotData function
        if data_to_plot:
            plotData(f"{title}", data_to_plot)
        else:
            st.warning("there is no data to plot.")

    def dft_plot(self, title="DFT Comparison", filters_to_show=None, absolute=True, fs=None):
        if not hasattr(self, 'filter_result'):
            self.filter_result = {}

        # Make a raw ECG data available if not already present        
        if 'Raw ECG' not in self.filter_result and self.var.dataECG is not None:
            self.filter_result['Raw ECG'] = self.var.dataECG
        
        # if fs is None, use the sampling frequency from the data
        if fs is None:
            fs = self.samplingFrequency()  # Default 1000Hz jika tidak ada data
        
        # Make a dictionary to store the DFT results
        dft_results = {}

        # filters_to_process = []
        filters_to_process = list(self.filter_result.keys()) if filters_to_show is None else filters_to_show

        # Check if the specified filters are in the filter_result
        for filter_name in filters_to_process:
            if filter_name in self.filter_result:
                # Take the data for the current filter
                data = self.filter_result[filter_name]
                
                # Perform DFT on the data
                dft_data = DFT(data)
                
                # Store the DFT result in the dictionary
                dft_results[filter_name] = dft_data
            else:
                st.warning(f"Filter '{filter_name}' tidak ditemukan.")

        # Plot the DFT results using the plotDFTs function        
        if dft_results:
            plotDFTs(f"{title} - DFT", dft_results, fs, absolute)
        else:
            st.warning("Tidak ada data yang dipilih untuk dibandingkan.")
