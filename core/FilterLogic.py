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
    Rancang Butterworth Low-Pass Filter digital dari spesifikasi analog
    tanpa prewarp (karena cutoff diberikan dalam domain analog).
    
    Args:
        fc (float): Analog cutoff frequency dalam rad/s
        order (int): Orde filter
        fs (float): Sampling frequency (Hz)
        
    Returns:
        b, a: Koefisien numerator dan denominator dari H(z)
    """
    T = 1 / fs  # sampling period

    # Cutoff frekuensi analog (sudah dalam rad/s, tidak perlu prewarp)
    Wc = fc * np.pi * 2

    # 1. Hitung analog poles Butterworth (di s-domain)
    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        s_k = Wc * complex(-np.sin(theta), np.cos(theta))
        poles.append(s_k)

    # 2. Transformasi bilinear (s → z)
    a = np.array([1.0])
    b = np.array([1.0])
    for p in poles:
        z_k = (2/T + p) / (2/T - p)
        a = np.convolve(a, [1, -z_k.real])  # hanya real part karena diasumsikan conjugate pair
        b = np.convolve(b, [1, 1])  # dummy numerator

    # 3. Normalisasi gain di DC
    gain = np.sum(b) / np.sum(a)
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
    
    # Generate analog Butterworth poles
    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = Wc * complex(-np.sin(theta), np.cos(theta))
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
        for i in range(len(b)):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        
        for j in range(1, len(a)):  
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

        for j in range(1, len(a)):  
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

def segment_ecg(ecg_signal, r_peaks, window_size=10):
    """
    Segment ECG signal into P, Q, R, S, and T waves based on R peaks
    Using a positional approach rather than amplitude-based detection
    
    Parameters:
    -----------
    ecg_signal : ndarray
        The filtered ECG signal
    r_peaks : list
        Indices of R-peaks
    window_size : int
        Window size for searching waves around R-peaks
        
    Returns:
    --------
    dict : Dictionary containing segmented waves
    """
    if ecg_signal is None or len(ecg_signal) == 0 or not r_peaks:
        st.warning("No ECG signal or R-peaks detected for segmentation.")
        return {}
    
    # Initialize segments dictionary
    segments = {
        "P": [],
        "Q": [],
        "R": r_peaks,
        "S": [],
        "T": []
    }
    
    # Find Q and S waves (immediately before and after R peaks)
    for r_idx in r_peaks:
        # Look for Q wave (minimum before R peak)
        q_search_start = max(0, r_idx - window_size//2)
        q_search_end = r_idx
        if q_search_start < q_search_end:
            q_idx = q_search_start + np.argmin(ecg_signal[q_search_start:q_search_end])
            segments["Q"].append(q_idx)
        
        # Look for S wave (minimum after R peak)
        s_search_start = r_idx + 1
        s_search_end = min(len(ecg_signal), r_idx + window_size//2)
        if s_search_start < s_search_end:
            s_idx = s_search_start + np.argmin(ecg_signal[s_search_start:s_search_end])
            segments["S"].append(s_idx)
    
    # Find P waves - look for the maximum point before each Q wave
    for i, q_idx in enumerate(segments["Q"]):
        # Determine P search region
        if i > 0 and i-1 < len(segments["S"]):  # If there's a previous beat
            # Look from previous T wave area to current Q
            prev_s_idx = segments["S"][i-1]
            # Estimate where T would likely end (about 2/3 between S and next Q)
            p_search_start = prev_s_idx + (q_idx - prev_s_idx) // 2  
        else:
            # If first beat, just look in reasonable window before Q
            p_search_start = max(0, q_idx - window_size * 2)
            
        p_search_end = max(p_search_start + 5, q_idx - window_size//4)  # Leave gap before Q
        
        if p_search_start < p_search_end:
            # Find the highest peak in this region
            p_region = ecg_signal[p_search_start:p_search_end]
            
            # Use peak detection rather than just max value
            p_candidates = []
            for i in range(1, len(p_region)-1):
                if p_region[i] > p_region[i-1] and p_region[i] > p_region[i+1]:
                    p_candidates.append((i, p_region[i]))
            
            if p_candidates:
                # Get the most prominent peak (highest amplitude)
                p_local_idx = max(p_candidates, key=lambda x: x[1])[0]
                p_idx = p_search_start + p_local_idx
                segments["P"].append(p_idx)
    
    # Find T waves - look for the maximum point after each S wave
    for i, s_idx in enumerate(segments["S"]):
        # Determine T search region
        if i < len(segments["S"]) - 1:  # If not the last beat
            next_q_idx = segments["Q"][i+1] if i+1 < len(segments["Q"]) else len(ecg_signal)-1
            # T wave occurs between current S and next P (which is before next Q)
            t_search_end = max(s_idx + window_size//2, next_q_idx - window_size)
        else:
            # If last beat, just look in reasonable window after S
            t_search_end = min(len(ecg_signal), s_idx + window_size * 2)
            
        t_search_start = s_idx + window_size//4  # Leave gap after S
        
        if t_search_start < t_search_end:
            # Find the highest peak in this region
            t_region = ecg_signal[t_search_start:t_search_end]
            
            # Use peak detection rather than just max value
            t_candidates = []
            for i in range(1, len(t_region)-1):
                if t_region[i] > t_region[i-1] and t_region[i] > t_region[i+1]:
                    t_candidates.append((i, t_region[i]))
            
            if t_candidates:
                # Get the most prominent peak (highest amplitude)
                t_local_idx = max(t_candidates, key=lambda x: x[1])[0]
                t_idx = t_search_start + t_local_idx
                segments["T"].append(t_idx)
            else:
                # If no clear peak found, use the maximum point
                t_idx = t_search_start + np.argmax(t_region)
                segments["T"].append(t_idx)
    
    return segments

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
    