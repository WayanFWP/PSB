import streamlit as st
import numpy as np
from .DisplayFunc import *
from .FilterLogic import *
from .config import DEFAULT_SAMPLE_INTERVAL
from traceback import print_exc

class Logic:
    def __init__(self, variable):
        self.var = variable
        self.file_path = "data/samples_10sec.csv"
        # self.file_path = "data/samples_1minute.csv"
        # self.file_path = "data/samples (7).csv"
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

            # Display the data
            st.subheader("Plot Raw data...")
            self.merged_data_plot("Raw ECG",["Raw ECG"])
            st.write(f"fs: {fs}, Duration: {duration:.2f} seconds, data: {len(self.var.dataECG)}")
       
            if self.var.dataECG is not None:
                st.subheader("Plot filtered data...")
                # Apply filters threshold
                fcl = 100
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
                
                seg_data = self.var.filtered_data
                t_idx, p_idx, qrs_idx, p_dat, qrs_dat, t_dat = segmented_ecg(seg_data)
                
                self.filter_result["P-wave"] = p_dat
                self.filter_result["QRS-complex"] = qrs_dat
                self.filter_result["T-wave"] = t_dat

                # Create DataFrame for Altair - show only 80 samples
                # Get a window of 80 samples (or fewer if signal is shorter)
                window_size = min(80, len(seg_data))
                
                # If the signal has QRS complexes, center the window around the first QRS complex
                start_idx = 0
                if len(qrs_idx) > 0:
                    # Center the window around the first QRS complex
                    center = qrs_idx[0]
                    start_idx = max(0, center - window_size // 2)
                    end_idx = min(len(seg_data), start_idx + window_size)
                    # Adjust start if end went beyond signal length
                    if end_idx - start_idx < window_size:
                        start_idx = max(0, end_idx - window_size)
                else:
                    end_idx = window_size
                
                # Slice the data
                windowed_data = seg_data[start_idx:end_idx]
                
                # Create base dataframe with filtered ECG data
                df_ecg = pd.DataFrame({
                    'index': range(start_idx, end_idx),
                    'value': windowed_data,
                    'type': 'Filtered ECG'
                })
                
                # Filter points to only show those in our window
                in_window = lambda idx: idx >= start_idx and idx < end_idx
                
                # Create dataframes for P, QRS, T points in the window
                df_p = pd.DataFrame({
                    'index': [i for i in p_idx if in_window(i)], 
                    'value': [p_dat[list(p_idx).index(i)] for i in p_idx if in_window(i)], 
                    'type': 'P-wave'
                }) if len(p_idx) > 0 else pd.DataFrame()
                
                df_qrs = pd.DataFrame({
                    'index': [i for i in qrs_idx if in_window(i)], 
                    'value': [qrs_dat[list(qrs_idx).index(i)] for i in qrs_idx if in_window(i)], 
                    'type': 'QRS-complex'
                }) if len(qrs_idx) > 0 else pd.DataFrame()
                
                df_t = pd.DataFrame({
                    'index': [i for i in t_idx if in_window(i)], 
                    'value': [t_dat[list(t_idx).index(i)] for i in t_idx if in_window(i)], 
                    'type': 'T-wave'
                }) if len(t_idx) > 0 else pd.DataFrame()
                
                # Combine all points for the scatter plot
                df_points = pd.concat([df_p, df_qrs, df_t])
                
                # Create the line chart
                line_chart = alt.Chart(df_ecg).mark_line(opacity=0.5, color='blue').encode(
                    x=alt.X('index:Q', title='Sample Index'),
                    y=alt.Y('value:Q', title='Amplitude')
                )
                
                # Create the scatter plot for wave points
                if not df_points.empty:
                    scatter_chart = alt.Chart(df_points).mark_circle(size=100).encode(
                        x='index:Q',
                        y='value:Q',
                        color=alt.Color('type:N', scale=alt.Scale(
                            domain=['P-wave', 'QRS-complex', 'T-wave'],
                            range=['green', 'red', 'purple']
                        )),
                        tooltip=['type', 'index', 'value']
                    )
                    
                    # Combine charts
                    chart = (line_chart + scatter_chart).properties(
                        title='Segmented ECG Signal (80 Samples Window)',
                        width=700,
                        height=400
                    ).configure_axis(
                        grid=True
                    ).configure_view(
                        strokeWidth=0
                    ).interactive()
                else:
                    chart = line_chart.properties(
                        title='Segmented ECG Signal (80 Samples Window)',
                        width=700,
                        height=400
                    ).configure_axis(
                        grid=True
                    ).configure_view(
                        strokeWidth=0
                    ).interactive()
                
                st.altair_chart(chart, use_container_width=True)

                # Store DFT results in the cache for later plotting
                fs_seg = fs
                self.compute_dft("P-wave", p_dat, fs_seg)
                self.compute_dft("QRS-complex", qrs_dat, fs_seg)
                self.compute_dft("T-wave", t_dat, fs_seg)
                
                # Plot segmentation components
                if show_dft and len(p_dat) > 0 and len(qrs_dat) > 0 and len(t_dat) > 0:
                    self.dft_plot("ECG Segments DFT", ["P-wave", "QRS-complex", "T-wave"], absolute=True, fs=fs)
                
                dft_result ,used_fs =self.compute_dft("QRS-complex", fs_seg)
                freq_values, mag_value = dft_result 

                fch_bpf, fcl_bpf =peak_magnitude(freq_values, mag_value)

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

                self.display_filter_response(filter_type="BPF", 
                                             fcl=self.filter_params["fc_l"],
                                             fch=self.filter_params["fc_h"],
                                             fs=fs)                

            if self.var.filtered_data is not None:
                st.subheader("Performing Heart beat calculation...")
                # Calculate MAV
                mav, threshold = self.applyMAV(
                    self.var.filtered_data, 
                    window_size=30
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
                filtered_data = LPF(data_input, fcl, frequencySampling)
                if plot:
                    self.dft_plot("DFT LPF", ["LPF"], absolute=True, fs=frequencySampling)
            elif filter_type == "HPF":
                filtered_data = HPF(data_input, fch, frequencySampling)
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
            threshold = (np.max(mav)) * 0.05 + np.mean(mav)
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
                dft_result, used_fs = self.compute_dft(filter_name, self.filter_result[filter_name], fs)
                
                # Extract the magnitude values (second element of the tuple)
                # Store only the magnitude part in the dictionary
                if dft_result is not None:
                    freq_values, mag_values = dft_result
                    dft_results[filter_name] = mag_values
                else:
                    st.warning(f"Failed to compute DFT for '{filter_name}'")
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
        
        freq_values, mag_values = DFT(data, fs)
        dft_result = (freq_values, mag_values)
        self.dft_cache[label] = {
            'dft': dft_result,
            'fs': fs
        }
        return dft_result, fs

    def display_filter_response(self, filter_type="BPF", fcl=None, fch=None, fs=None):
        """
        Calculate and display the frequency response of a filter.
        
        Args:
            filter_type (str): Filter type (BPF, LPF, HPF)
            fcl (float): Low cutoff frequency
            fch (float): High cutoff frequency
            fs (float): Sampling frequency
        """
        if not fcl or not fch or not fs:
            st.warning("Please provide valid filter parameters.")
            return
        
        st.subheader(f"{filter_type} Frequency Response")
        st.write(f"Low cutoff: {fcl} Hz, High cutoff: {fch} Hz, Sampling rate: {fs} Hz")
        
        # Calculate the frequency response
        freq, response = frequency_response(
            signal=np.zeros(100),  # Dummy signal, not used in calculation
            fs=fs,
            fl=fcl,
            fh=fch
        )
        
        # Plot the frequency response
        plotFrequencyResponse(
            title="Filter Frequency Response",
            freq=freq,
            response=response,
            filter_name=filter_type
        )