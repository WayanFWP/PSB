import streamlit as st
import pandas as pd

class App:
    def __init__(self):
        self.page = None
        # Initialize session state for data if it doesn't exist
        if 'data' not in st.session_state:
            st.session_state.data = None

    def run(self):
        self.sidebar()
        self.content()

    def sidebar(self):
        st.sidebar.title("Navigation")
        self.page = st.sidebar.radio("Go to", ("Home", "Data", "Chart"))

    def content(self):
        if self.page == "Home":
            st.write("Welcome to the home page!")
        elif self.page == "Data":
            self.load_and_display_data_raw()
        elif self.page == "Chart":
            self.display_chart()

    def load_and_display_data_raw(self):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                # First read all columns to inspect them
                df = pd.read_csv(uploaded_file)

                # Find the ECG column (case insensitive and handling quotes)
                ecg_col = None
                for col in df.columns:
                    # Strip quotes and whitespace and check case-insensitive match
                    clean_col = col.strip("'\"").strip()
                    if clean_col.lower() == 'ecg':
                        ecg_col = col
                        break

                if ecg_col is not None:
                    # Keep only the ECG column and reset the column name if needed
                    st.session_state.data = df[[ecg_col]].copy()
                    if ecg_col != 'ECG':
                        st.session_state.data.columns = ['ECG']

                    st.title("PLOT DATA")
                    st.subheader("ECG Data Samples")
                    st.dataframe(st.session_state.data)
                else:
                    st.error("Could not find 'ECG' column in the CSV file.")
                    st.session_state.data = None

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.data = None

        # Display existing ECG data if already loaded
        elif st.session_state.data is not None:
            st.title("PLOT DATA")
            st.subheader("ECG Data Samples")
            st.dataframe(st.session_state.data)
        else:
            st.info("Upload a CSV file containing an 'ECG' column.")


    def display_chart(self):
        # Access data from session state
        if st.session_state.data is not None:
            st.subheader("Line Chart")
            # Ensure data is suitable for line chart (e.g., numeric columns)
            # You might need to select specific columns or set an index
            try:
                # Attempt to display the line chart
                st.line_chart(st.session_state.data)
            except Exception as e:
                st.error(f"Could not display chart. Ensure data is numeric and properly formatted. Error: {e}")
                st.dataframe(st.session_state.data) # Show data for debugging
        else:
            st.write("Please upload data first on the 'Data' page.")

if __name__ == "__main__":
    app = App()
    app.run()
