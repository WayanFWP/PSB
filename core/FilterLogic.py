import numpy as np
import streamlit as st

def DFT(signal, fs):
    # Convert to numpy array if it's a DataFrame
    if hasattr(signal, 'values'):
        signal = signal.values.flatten()
    elif not isinstance(signal, np.ndarray):
        signal = np.array(signal)
        
    N = len(signal)
    Re = np.zeros(N) 
    Im = np.zeros(N)
    Mag = np.zeros(N)


    for k in range(N):
        for n in range(N):
            omega = 2 * np.pi * k * n / N
            Re[k] += signal[n] * np.cos(omega)
            Im[k] -= signal[n] * np.sin(omega)

        Mag[k] = np.sqrt(Re[k] ** 2 + Im[k] ** 2)

    f = np.arange(0, N // 2) * fs / N

    return f, Mag[:N//2]

# filter
def LPF(signal, fl, fs):
    N = len(signal)
    T = 1 / fs
    Wc = 2 * np.pi * fl

    # Koefisien
    denom = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
    b1 = ((8 / T**2) - (2 * Wc**2)) / denom
    b2 = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denom
    a0 = Wc**2 / denom
    a1 = 2 * Wc**2 / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(2, N-2):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y
  
def HPF(signal,fh,fs):
    N = len(signal)
    T = 1/fs
    Wc = 2 * np.pi * fh

    #   koefisien
    denom = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/ denom
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/ denom
    a0 = (4/T**2) / denom
    a1 = (-8/T**2) / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(0, N-1):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y
    

def BPF(signal, fc1, fc2, order, fs):
    orde = int(order / 2)

    lpf_data = LPF(signal, fc1, fs)
    filtered_data = HPF(lpf_data, fc2, fs)
    return filtered_data

def segmented_ecg(sig):
    # col1, col2, col3 = st.columns(3)

    # Input untuk P wave
    # t0p = col1.number_input("Start time of P wave (ms)", min_value=0, value=19, step=1)
    # t1p = col1.number_input("End time of P wave (ms)", min_value=0, value=35, step=1)
    start_p, end_p = 19, 35
    p_wave = sig[start_p:end_p]
    index_p = np.arange(start_p, end_p)

    # Input untuk QRS complex
    # t0qrs = col2.number_input("Start time of QRS complex (ms)", min_value=0, value=34, step=1)
    # t1qrs = col2.number_input("End time of QRS complex (ms)", min_value=0, value=46, step=1)
    start_qrs, end_qrs = 34, 46
    qrs_wave = sig[start_qrs:end_qrs]
    index_qrs = np.arange(start_qrs, end_qrs)

    # Input untuk T wave
    # t0t = col3.number_input("Start time of T wave (ms)", min_value=0, value=45, step=1)
    # t1t = col3.number_input("End time of T wave (ms)", min_value=0, value=78, step=1)
    start_t, end_t = 45,78
    t_wave = sig[start_t:end_t]
    index_t = np.arange(start_t, end_t)
    return index_t, index_p, index_qrs, p_wave, qrs_wave, t_wave

def peak_magnitude(f_qrs, Mag_qrs):
    index_max = np.argmax(Mag_qrs)
    fc_low = f_qrs[index_max]

    if fc_low < 0.1:
        fc_low = 0.1
    
    mag_qrs_copy = np.copy(Mag_qrs)
    mag_qrs_copy[index_max] = -np.inf

    fc_high = f_qrs[np.argmax(mag_qrs_copy)]

    return fc_low, fc_high

def frequency_response(signal, fs, fl, fh):
    N = len(signal)
    T = 1 / fs
    wc_lpf = 2 * np.pi * fl
    wc_hpf = 2 * np.pi * fh
    num_points = 1000
    omegas = np.linspace(0, np.pi, num_points)
    frequencies = omegas * fs / (2 * np.pi)  
    magnitude_response_bpf = np.zeros(num_points)

    for i, omega in enumerate(omegas):
        # High-pass filter (HPF) - perhitungan respons kompleks
        numR_hpf = (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
        numI_hpf = (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
        denumR_hpf = (
            wc_hpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
            + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
            + (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
        )
        denumI_hpf = (
            wc_hpf**2 * (2 * np.sin(omega) - np.sin(2 * omega))
            + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
            + (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
        )
        hpf_complex_response = (numR_hpf + 1j * numI_hpf) / (denumR_hpf + 1j * denumI_hpf)

        # Low-pass filter (LPF) - perhitungan respons kompleks
        numR_lpf = wc_lpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
        numI_lpf = -wc_lpf**2 * (2 * np.sin(omega) + np.sin(2 * omega))
        denumR_lpf_lpf = (
            (4 / T**2) + (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2
            - ((8 / T**2) - 2 * wc_lpf**2) * np.cos(omega)
            + ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.cos(2 * omega)
        )
        denumI_lpf_lpf = (
            ((8 / T**2) - 2 * wc_lpf**2) * np.sin(omega)
            - ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.sin(2 * omega)
        )
        lpf_complex_response = (numR_lpf + 1j * numI_lpf) / (denumR_lpf_lpf + 1j * denumI_lpf_lpf)

        # Band-pass filter (BPF) - perkalian respons kompleks
        bpf_complex_response = hpf_complex_response * lpf_complex_response
        magnitude_response_bpf[i] = np.abs(bpf_complex_response)

    return frequencies, magnitude_response_bpf

def moving_average(data, window_size):
    N = window_size
    smoothed_data = np.zeros(len(data))  
    for i in range(len(data)):
        sum_window = 0
        for k in range(N):  
            if i - k >= 0:  
                sum_window += data[i - k]
        smoothed_data[i] = sum_window / N 
    return smoothed_data

def process_heart_rate(mav, config):
    # Adjust threshold based on configuration
    threshold = config["threshold"]
    
    # Find R-peaks
    r_peaks, r_values = detect_peak(
        mav, 
        threshold, 
        config["interval"]
    )
    
    # Calculate and display heart rate metrics
    return calculate_and_display_heart_rate(r_peaks, r_values, mav, threshold)

def detect_peak(signal, threshold, peak_to_peak):
    r_peaks = []
    r_values = []
    
    i = 0
    while i < len(signal) - 1:
        # Check if current point is above threshold and is a local maximum
        if signal[i] > threshold and i > 0 and i < len(signal) - 1:
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                r_peaks.append(i)
                r_values.append(signal[i])
                i += peak_to_peak  # Skip forward to look for next peak
                continue
        i += 1
        
    return r_peaks, r_values

def calculate_and_display_heart_rate(r_peaks, r_values, mav, threshold):
    if not r_peaks:
        st.warning("No R-peaks detected. Try adjusting the configuration parameters.")
        return
        
    intervals = np.diff(r_peaks) * 0.01  
    avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
    heart_rate = 60 / avg_interval if avg_interval > 0 else 0

    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("R-peaks detected", f"{len(r_peaks)}")
    with metrics_col2:
        st.metric("Average interval", f"{avg_interval:.3f} s")
    with metrics_col3:
        st.metric("Heart Rate", f"{heart_rate:.1f} BPM")

    # Create visualization data
    return mav, r_peaks, r_values, threshold, heart_rate
    