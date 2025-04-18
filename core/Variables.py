import streamlit as st

class Variable:
    def __init__(self):
        self.fs = 400 # Sampling frequency in Hz
        self.amplitude = 1  # Amplitude in mV
        self.fcl = 200  # Low cutoff frequency in Hz
        self.fch = 0 # High cutoff frequency in Hz
        self.orde = 6
        self.move = 30 # Moving average window size

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
        