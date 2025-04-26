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
        self.page = st.sidebar.radio("Select a page:", ["Chart"])
        if self.page == "Home":
            st.sidebar.info("Welcome to the ECG Signal Processing App!")
        elif self.page == "Data":
            st.sidebar.info("Display Data Page")
        elif self.page == "Chart":
            st.sidebar.info("Chart Page")
        
    def content(self):
        if self.page == "Home":
            st.title("ECG Signal Processing")
            st.write("Welcome to the ECG Signal Processing App!")
            st.write("Select a page from the sidebar to get started.")
        elif self.page == "Data":
            st.title("Data Display")
            self.logic.loadDisplayData()
        elif self.page == "Chart":
            self.logic.process_data()

if __name__ == "__main__":
    app = App()
    app.run()