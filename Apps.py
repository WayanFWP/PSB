import streamlit as st
from core.Logics import Logic
from core.Variables import Variable
from core.DisplayFunc import *

st.set_page_config(page_title="ECG Signal Processing", layout="wide")

class App:
    def __init__(self):
        self.vars = Variable()
        self.logic = Logic(self.vars)
        self.page = None
    
    def run(self):
        self.sidebar()
        self.content()

    def sidebar(self):
        st.sidebar.title("Navigation")
        # self.page = st.sidebar.radio("Select a page:", ["Home", "Flow_data"])
        self.page = st.sidebar.radio("Select a page:", ["Flow_data"])
        if self.page == "Home":
            st.sidebar.info("Welcome to the ECG Signal Processing App!")        
        
    def content(self):
        if self.page == "Home":
            file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"])

            if file_path is None:
                st.title("ECG Signal Processing")
                st.write("Welcome to the ECG Signal Processing App!")
                st.write("Select a page from the sidebar to get started.")

            # Use Streamlit's session state to store the file path
            if file_path is not None:
                st.session_state.file_path = file_path

                # Display filtered data if it exists in session state
                if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
                    st.write("Filtered Data:")
                    plotLine("Filtered Data", st.session_state.dataECG , st.session_state.filtered_data, title2="Filtered Data")

                # Display heart rate if it exists in session state
                if "heart_rate" in st.session_state and st.session_state.heart_rate is not None:
                    st.write(f"Heart Rate: {int(st.session_state.heart_rate)} bpm")
                else: st.write("do the Flow_data first then come back.")

        elif self.page == "Flow_data":
            self.logic.process_data()

if __name__ == "__main__":
    app = App()
    app.run()