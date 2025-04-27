import streamlit as st
from core.Logics import Logic
from core.Variables import Variable

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
            file_path = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"])
            self.logic.loadDisplayData(self.vars.dataECG)

        
    def content(self):
        if self.page == "Home":
            st.title("ECG Signal Processing")
            st.write("Welcome to the ECG Signal Processing App!")
            st.write("Select a page from the sidebar to get started.")
        elif self.page == "Data":
            st.title("Data Display")
            self.logic.loadDisplayData()
        elif self.page == "Flow_data":
            self.logic.process_data()

if __name__ == "__main__":
    app = App()
    app.run()