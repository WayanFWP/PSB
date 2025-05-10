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

def detect_r_peaks(signal, threshold_percentage=0.85, window_size=10):
    """
    Mendeteksi puncak R dalam sinyal ECG menggunakan threshold adaptif.
    
    Args:
        signal: Sinyal ECG (biasanya hasil filter BPF atau MAV)
        threshold_percentage: Persentase dari nilai maksimum untuk threshold (0-1)
        window_size: Ukuran jendela untuk moving average (opsional)
    
    Returns:
        Tuple berisi (indeks puncak R, nilai threshold)
    """
    if signal is None or len(signal) == 0:
        return [], 0
    
    # Ambil nilai absolut sinyal
    abs_signal = np.abs(signal)
    
    # Hitung threshold adaptif
    threshold = np.mean(abs_signal) * threshold_percentage
    
    # Temukan puncak R yang melebihi threshold
    r_peaks = []
    i = 1
    while i < len(signal) - 1:
        # Jika nilai saat ini melebihi threshold dan lebih besar dari tetangganya (local maximum)
        if abs_signal[i] > threshold and abs_signal[i] > abs_signal[i-1] and abs_signal[i] > abs_signal[i+1]:
            # Tambahkan indeks ke daftar puncak R
            r_peaks.append(i)
            
            # Lompati beberapa sampel untuk menghindari deteksi ganda pada satu kompleks QRS
            refractory_period = int(0.2 * 100)  # 200ms refractory period assuming 100Hz
            i += max(1, refractory_period)
        else:
            i += 1
    
    return r_peaks, threshold

def segment_qrs(signal, r_peaks, window_before=50, window_after=50):
    """
    Segmentasi kompleks QRS berdasarkan puncak R yang terdeteksi.
    
    Args:
        signal: Sinyal ECG
        r_peaks: Indeks puncak R
        window_before: Jumlah sampel sebelum puncak R untuk segmentasi
        window_after: Jumlah sampel setelah puncak R untuk segmentasi
    
    Returns:
        Dictionary dengan key 'qrs_segments', 'q_points', 's_points'
    """
    if not r_peaks:
        return {"qrs_segments": [], "q_points": [], "s_points": []}
    
    qrs_segments = []
    q_points = []
    s_points = []
    
    for peak in r_peaks:
        # Definisikan batas jendela
        start = max(0, peak - window_before)
        end = min(len(signal), peak + window_after)
        
        # Ekstrak segmen QRS
        segment = signal[start:end]
        qrs_segments.append(segment)
        
        # Temukan titik Q (minimum sebelum puncak R)
        q_search_start = max(0, peak - window_before)
        q_search_end = peak
        if q_search_end > q_search_start:
            q_idx = q_search_start + np.argmin(signal[q_search_start:q_search_end])
            q_points.append(q_idx)
        else:
            q_points.append(peak)
        
        # Temukan titik S (minimum setelah puncak R)
        s_search_start = peak
        s_search_end = min(len(signal), peak + window_after)
        if s_search_end > s_search_start:
            s_idx = s_search_start + np.argmin(signal[s_search_start:s_search_end])
            s_points.append(s_idx)
        else:
            s_points.append(peak)
    
    return {
        "qrs_segments": qrs_segments,
        "q_points": q_points,
        "s_points": s_points
    }