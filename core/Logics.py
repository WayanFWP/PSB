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
        self.dft_cache = {}

    # ======= main section =========
    def process_data(self):
        try:
            # Check if the user has uploaded a file
            file_path = st.sidebar.file_uploader("Upload data(csv or txt)", type=["csv", "txt"])
            if file_path is not None:
                self.loadDisplayData(file_path, data_log = True) # Use the uploaded file
            else:
                self.loadDisplayData(self.file_path) # Use the default file path if no file is uploaded

            show_dft = st.sidebar.checkbox("Show DFT", value=False)
            if self.var.dataECG is not None:
                fs = 2 * np.max(np.abs(self.var.dataECG))
                duration = len(self.var.dataECG) * DEFAULT_SAMPLE_INTERVAL
                self.compute_dft("Raw ECG", self.var.dataECG, fs)

            # Display the data
            st.subheader("Plot Raw data...")
            self.merged_data_plot("Raw ECG",["Raw ECG"])
            st.write(f"fs: {fs}, Duration: {duration:.2f} seconds, data: {len(self.var.dataECG)}")
            if show_dft:
               self.dft_plot("DFT Raw", ["Raw ECG"], absolute=True, fs=fs)
       
            if self.var.dataECG is not None:
                st.subheader("Plot filtered data...")
                # Apply filters threshold
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
                self.merged_data_plot("After Prefiltering")
                self.compute_dft("LPF", self.filtered_data, fs)
                if show_dft:
                    self.dft_plot("DFT Raw vs LPF", ["Raw ECG", "LPF"], absolute=True, fs=fs)                    
                
                seg_data = self.var.filtered_data
                self.process_segmentation(seg_data, window_name="Segmentation")
                    
                # Apply high-pass filter threshold 
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
                self.compute_dft("BPF", self.filtered_data, fs)
                if show_dft:
                    self.dft_plot("DFT BPF", ["BPF"], absolute=True, fs=fs)

            if self.var.filtered_data is not None:
                st.subheader("Performing Heart beat calculation...")
                # Calculate MAV
                mav, threshold = self.applyMAV(
                    self.var.filtered_data, 
                    window_size=10
                )

                config = {
                    "threshold": threshold,
                    "interval": 80,
                    "sample_interval": DEFAULT_SAMPLE_INTERVAL
                }
                
                # Process ECG signal for heart beat detection
                mav, r_peak, r_value, threshold, heart_rate = process_heart_rate(mav, config)
                self.var.heart_rate = heart_rate
                visualize_heart_rate(mav, r_peak, r_value, threshold)
            else:
                st.warning("Please upload data first on the page.")

        except Exception as e:
            print_exc()
            st.error(f"An error occurred: {e}")
    
    # ======== utility section =========
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
        
    def process_segmentation(self, ecg_data, expanded= False, window_name="segmentation"):
        segments = {}

        with st.expander(f"{window_name}...", expanded=expanded):
            threshold = np.max(ecg_data) * 0.6 + np.std(ecg_data)
            r_peaks = []
            
            for i in range(1, len(ecg_data)-1):
                if (ecg_data[i] > ecg_data[i-1] and 
                    ecg_data[i] > ecg_data[i+1] and 
                    ecg_data[i] > threshold):
                    # Ensure minimum distance between peaks
                    if not r_peaks or i - r_peaks[-1] > 30:  # ~300ms at 100Hz
                        r_peaks.append(i)
            
            if r_peaks:
                # Perform segmentation
                segments = segment_ecg(ecg_data, r_peaks)
                
                # Visualize the segments using Altair
                visualize_pqrst_altair(ecg_data, segments)

                # Display metrics for each wave type
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("P", f"{len(segments['P'])}")
                with col2:
                    st.metric("Q", f"{len(segments['Q'])}")
                with col3:
                    st.metric("R", f"{len(segments['R'])}")
                with col4:
                    st.metric("S", f"{len(segments['S'])}")
                with col5:
                    st.metric("T", f"{len(segments['T'])}")
                    
    # ======== filter section =========
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

    def applyMAV(self, data_input=None, window_size=30):
        if data_input is None:
            st.warning("No data to process.")
            return
        
        self.filter_result = {}
        
        data_input = np.abs(data_input)
        self.filter_result["Abs"] = data_input

        try:
            mav = moving_average(data_input, window_size)
            self.filter_result["MAV"] = mav
            self.merged_data_plot("MAV", ["MAV"])

            st.write(f"Mean Absolute Value (MAV) calculated with window size {window_size}.")
            threshold = (np.max(mav) - np.mean(mav)) * 0.25
            st.write(f"Threshold: {threshold:.5f}")

            return mav , threshold
            
        except Exception as e:
            st.error(f"Error during MAV calculation: {e}")

    # ========= load section =========
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
                    st.warning(f"Filter '{filter_name}' Not Found.")
        
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
            fs = self.samplingFrequency()
        
        # Make a dictionary to store the DFT results
        dft_results = {}

        # filters_to_process = []
        filters_to_process = list(self.filter_result.keys()) if filters_to_show is None else filters_to_show

        # Check if the specified filters are in the filter_result
        for filter_name in filters_to_process:
            if filter_name in self.filter_result:
                dft_data, used_fs = self.compute_dft(filter_name, self.filter_result[filter_name], fs)
                
                # Store the DFT result in the dictionary
                dft_results[filter_name] = dft_data
            else:
                st.warning(f"Filter '{filter_name}' Not Found (404).")

        # Plot the DFT results using the plotDFTs function        
        if dft_results:
            plotDFTs(f"{title} - DFT", dft_results, fs, absolute)
        else:
            st.warning("There is no data to compare.")

    #@make_a_paralel_processing
    def compute_dft(self, label, data=None, fs=None):
        if label in self.dft_cache:
            return self.dft_cache[label]['dft'], self.dft_cache[label]['fs']

        if data is None:
            if label in self.filter_result:
                data = self.filter_result[label]
            else:
                st.warning(f"Filter '{label}' Not Found (404).")
                return None, None
        
        dft_result = DFT(data)
        self.dft_cache[label] = {
            'dft': dft_result,
            'fs': fs
        }
        return dft_result, fs
