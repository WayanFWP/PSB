# Filtering Amplitude to ECG Morphology. 

Biomedical Signal Processing Project that converting the raw amplitude values in the data folder(default) and applying a bandpass filter to obtain a clear BPM.

## Prerequisites

- Python 3.x installed on your system
- pip (Python package manager)

## Installation steps
Clone the repository:
 ```bash
 git clone https://github.com/WayanFWP/PSB.git ECG_Morphology
 cd ECG_Morphology
 ```

Install dependencies:
```bash
pip install -r requirements.txt
```

#### Using Project Environment (Optional)
Create virtual env:
```bash
python -m venv .venv
``` 
then use the virtual env
```bash 
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows
```
then install depedencies: 
```bash
pip install -r requirements.txt
```
Ensure all the above steps are completed before running the application.

## Run the application
- Windows
use the gui program for gui output:
    ```bash
    streamlit run Apps.py
    ```

- Linux
use the gui program for gui output:
    ```bash
    streamlit run Apps.py
    ```
