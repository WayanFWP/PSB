import streamlit as st

class Variable:
    def __init__(self):
        self.amplitude = 1  # Amplitude in mV
        self.GLOBAL_fcl = st.sidebar.slider("Low Cutoff Frequency", min_value=10,max_value=100, value=10)  # Low cutoff frequency in Hz
        self.GLOBAL_fch = st.sidebar.slider("High Cutoff Frequency", min_value=1, max_value=50, value=1) # High cutoff frequency in Hz
        self.GLOBAL_orde = st.sidebar.slider("Filter Order", min_value=1, max_value=100, value=1)
        self.GLOBAL_MAV = 30 # Moving average window size

        self._dataECG = None
        self._dft = None
        self._filtered_data = None

        # Initialize session state for data if it doesn't exist
        st.session_state.setdefault('dataECG', None)
        st.session_state.setdefault('dft', None)
        st.session_state.setdefault('filtered_data', None)

    # Getter and Setter for each variable
    @property
    def dataECG(self): return self._dataECG
    @dataECG.setter
    def dataECG(self, value): self._dataECG = value; st.session_state.dataECG = value
    @property
    def dft(self): return self._dft
    @dft.setter
    def dft(self, value): self._dft = value; st.session_state.dft = value
    @property
    def filtered_data(self): return self._filtered_data
    @filtered_data.setter
    def filtered_data(self, value): self._filtered_data = value; st.session_state.filtered_data = value
        