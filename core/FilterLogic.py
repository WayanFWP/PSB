import numpy as np
import streamlit as st

def DFT(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D array.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: DFT of the input array.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of a 1D array.
    Args:
        X (np.ndarray): Input DFT array.
    Returns:
        np.ndarray: Reconstructed time-domain signal.
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    x = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            x[n] += X[k] * np.exp(angle)
        x[n] /= N
    return np.real(x)

def LPF(fc, order, fs):
    """
    Design a Butterworth Low-Pass Filter (LPF) using bilinear transform.
    
    Args:
        fc (float): Cutoff frequency (Hz).
        order (int): Filter order.
        fs (float): Sampling frequency (Hz).
        
    Returns:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
    """
    # Normalized digital cutoff frequency (prewarped)
    Wc = 2 * np.pi * fc
    T = 1 / fs
    prewarp = 2 / T * np.tan(Wc * T / 2)  # Prewarping for bilinear transform
    
    # Generate analog Butterworth poles
    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = prewarp * complex(-np.sin(theta), np.cos(theta))
        poles.append(pole)
    
    # Initialize coefficients
    a = np.array([1.0])
    b = np.array([1.0])
    
    # Bilinear transform for each pole
    for p in poles:
        # Transform analog pole to digital 
        zd = (2/T + p) / (2/T - p)
        a = np.convolve(a, [1, -zd.real])
        b = np.convolve(b, [1, 1])
    
    # Normalize gain at DC (ω=0) to 1
    gain = np.sum(b) / np.sum(a)
    if np.isclose(gain, 0) or np.isinf(gain) or np.isnan(gain):
        raise ValueError("Invalid gain calculated for. Check filter parameters.")
    b = b / gain
    
    return b.real, a.real

def HPF(fc, order, fs):
    """
    Design a Butterworth High-Pass Filter (HPF) using bilinear transform.
    Args:
        fc (float): Cutoff frequency (Hz).
        order (int): Filter order.
        fs (float): Sampling frequency (Hz).
        
    Returns:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
    """
    # Normalized digital cutoff frequency (prewarped)
    Wc = 2 * np.pi * fc
    T = 1 / fs
    prewarp = 2 / T * np.tan(Wc * T / 2)  # Prewarping for bilinear transform
    
    # Generate analog Butterworth poles
    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = prewarp * complex(-np.sin(theta), np.cos(theta))
        poles.append(pole)
    
    # Initialize coefficients
    a = np.array([1.0])
    b = np.array([1.0])
    
    # Bilinear transform for each pole (HPF version)
    for p in poles:
        # Transform analog pole to digital (HPF)
        zd = (2/T + p) / (2/T - p)
        a = np.convolve(a, [1, -zd.real])
        b = np.convolve(b, [1, -1])  # Note the [1, -1] here for HPF characteristic
    
    # Normalize gain at Nyquist (ω=π) to 1
    # For HPF, we evaluate at z=-1 (ω=π)
    gain = np.sum(b * np.power(-1, np.arange(len(b)))) / np.sum(a * np.power(-1, np.arange(len(a))))
    if np.isclose(gain, 0) or np.isinf(gain) or np.isnan(gain):
        raise ValueError("Invalid gain calculated for HPF. Check filter parameters.")
    b = b / gain
    
    return b.real, a.real
    

def BPF(signal, fc1, fc2, order, fs):
    orde = int(order / 2)

    b_lpf, a_lpf = LPF(fc1, orde, fs)
    filtered_lpf = forward_backward_filter(b_lpf, a_lpf, signal)

    b_hpf, a_hpf = HPF(fc2, orde, fs)
    filtered_data = forward_backward_filter(b_hpf, a_hpf, filtered_lpf)

    return filtered_data

def forward_filter_IIR(b, a, x):
    """
    forward filtering for IIR
    """
    y = np.zeros_like(x)
    # Input filtering IIR
    for n in range(len(x)):
        # Loop untuk setiap sample input
        for i in range(len(b)):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        
        for j in range(1, len(a)):  # Mulai dari a1
            if n - j >= len(x):
                y[n] -= a[j] * y[n - j]

    return y

def backward_filter_IIR(b, a, x):
    """
    backward filtering for IIR
    Returns:
        np.ndarray: Output filter
    """
    y = np.zeros_like(x)
    # Backward filtering IIR
    for n in range(len(x) - 1, -1, -1):
        for i in range(len(b)):
            if n + i < len(x):
                y[n] += b[i] * x[n + i]

        for j in range(1, len(a)):  # Mulai dari a1
            if n + j < len(x):
                y[n] -= a[j] * y[n + j]
                
    return y


def forward_backward_filter(b, a, x):
    """
    Apply forward-backward IIR filtering for zero-phase distortion.
    Args:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
        x (np.ndarray): Input signal.
    Returns:
        np.ndarray: Filtered signal.
    """
    b = b.astype(np.float64)
    a = a.astype(np.float64)
    x = x.astype(np.float64)
    
    y_forward = forward_filter_IIR(b, a, x)
    y = backward_filter_IIR(b, a, y_forward)
    return y

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
    """
    Process ECG signal to detect heart rate
    
    Parameters:
    -----------
    mav    : ndarray
        The Moving average values of the ECG signal
    config : dict
        Configuration parameters for heart rate detection
    """
    
    # Adjust threshold based on configuration
    threshold = config["threshold"]
    
    # Find R-peaks
    r_peaks, r_values = detect_r_peaks(
        mav, 
        threshold, 
        config["interval"]
    )
    
    # Calculate and display heart rate metrics
    return calculate_and_display_heart_rate(r_peaks, r_values, mav, threshold)

def detect_r_peaks(signal, threshold, peak_to_peak):
    """
    Detect R-peaks in the signal above the threshold
    
    Parameters:
    -----------
    signal : ndarray
        The processed signal to detect peaks from
    threshold : float
        The amplitude threshold for peak detection
    peak_to_peak : int
        Minimum number of samples between peaks
        
    Returns:
    --------
    tuple: (r_peaks, r_values)
        Lists of peak positions and their values
    """
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
    """
    Calculate and display heart rate metrics and visualization
    
    Parameters:
    -----------
    r_peaks : list
        Indices of detected R-peaks
    r_values : list
        Amplitude values at R-peaks
    mav : ndarray
        Moving average values
    threshold : float
        Threshold used for peak detection
    """
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
    