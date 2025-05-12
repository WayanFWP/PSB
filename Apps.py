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

        if 'file_path' not in st.session_state:
            st.session_state.file_path = None
        if 'data_processed' not in st.session_state:
            st.session_state.data_processed = None
    
    def run(self):
        self.sidebar()
        self.content()
        self.footer()

    def sidebar(self):
        st.sidebar.title("Navigation")
        self.page = st.sidebar.radio("Select a section:", ["Home", "Flow_data"])
        
        # debugging purpose
        # self.page = st.sidebar.radio("Select a page:", ["Flow_data"])

        
    def content(self):
        if self.page == "Home":
            st.title("ECG Signal Processing")

            # File uploader
            uploaded_file = st.sidebar.file_uploader("Upload data(csv or txt)", type=["csv", "txt"])
            if uploaded_file is not None and st.session_state.file_path != uploaded_file:
                st.session_state.file_path = uploaded_file
                st.session_state.data_processed = False

                with st.spinner("Loading..."):
                    # Read the CSV file
                    self.vars.dataECG = read_csv_file(uploaded_file)
                    st.success("File loaded successfully! Go to Flow_data page to see the analysis.")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ## About this app
                This application allows you to analyze ECG signals with advanced processing techniques.
                
                ### Features:
                - Upload and visualize ECG data
                - Filter noise and detect abnormalities
                - Analyze heart rate variability
                - Export processed signals
                """)
                
            with col2:
                st.markdown("""
                ### How to use:
                1. Upload your ECG data file
                2. Navigate to Flow_data page
                3. Analyze your results

                *Need sample data? Click the button below:*
                """)
                if st.button("Load Sample Data"):
                    self.vars.dataECG = read_csv_file(self.logic.file_path)
                    st.session_state.file_path = self.logic.file_path
                    st.session_state.data_processed = False
            
            

        elif self.page == "Flow_data":
            self.logic.process_data()
            st.session_state.data_processed = True
    
    def footer(self):
        """Add a compact footer with profile information and links that stays accessible without blocking navbar."""
        # Add a spacer before footer content
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Add an expander for footer content instead of fixed positioning
        with st.expander("Project Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Profile:**  
                [I Wayan Firdaus Winarta Putra](https://github.com/WayanFWP) | 5023231064
                """)
            
            with col2:
                st.markdown("""
                **Resources:**  
                [GitHub](https://github.com/WayanFWP/PSB) | 
                [Documentation](https://github.com/WayanFWP/PSB/blob/main/README.md)
                """)
            
            st.markdown("""
            <div style="text-align:center; color:gray; font-size:0.9em; margin-top:10px;">
                Created for Biomedical Signal Processing Project
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = App()
    app.run()