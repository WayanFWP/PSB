import numpy as np

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
    X = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

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
        zd = (2/T - p) / (2/T + p)
        a = np.convolve(a, [1, -zd.real])
        b = np.convolve(b, [1, -1])  # HPF characteristic
    
    # Normalize gain at Nyquist (ω=π) to 1
    # For HPF, we evaluate at z=-1 (ω=π)
    gain = np.sum(b * (-1)**np.arange(len(b))) / np.sum(a * (-1)**np.arange(len(a)))
    if np.isclose(gain, 0) or np.isinf(gain) or np.isnan(gain):
        raise ValueError("Invalid gain calculated for HPF. Check filter parameters.")
    b = b / gain
    
    return b.real, a.real


def BPF(fc1, fc2, order, fs):
    """
    Design a Butterworth Band-Pass Filter (BPF) using bilinear transform.
    Args:
        fc1 (float): Lower cutoff frequency (Hz).
        fc2 (float): Upper cutoff frequency (Hz).
        order (int): Filter order (must be even).
        fs (float): Sampling frequency (Hz).
    Returns:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
    """
    if order % 2 != 0:
        raise ValueError("Order of BPF must be even.")

    W1 = 2 * np.pi * fc1
    W2 = 2 * np.pi * fc2
    T = 1 / fs

    W1p = 2 / T * np.tan(W1 * T / 2)
    W2p = 2 / T * np.tan(W2 * T / 2)

    W0 = np.sqrt(W1p * W2p)  # Center frequency
    Bw = W2p - W1p           # Bandwidth

    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = Bw/2 * (-np.sin(theta) + 1j * np.cos(theta))
        poles.append(pole)

    a = np.array([1.0])
    b = np.array([1.0])
    for p in poles:
        s_pole = p + W0**2 / p  # Bandpass transformation
        k = (2 / T + s_pole) / (2 / T - s_pole)
        a = np.convolve(a, [1, -k.real])
        b = np.convolve(b, [1, 1])

    gain = np.sqrt(np.sum(b**2)) / np.sqrt(np.sum(a**2))
    if gain == 0 or np.isinf(gain) or np.isnan(gain):  # Check for invalid gain
        raise ValueError("Invalid gain calculated for BPF. Check filter parameters.")
    b = b / gain
    return b.real, a.real

def forward_filter_IIR(b, a, x):
    """
    Melakukan forward filtering untuk IIR (kasus filter digital)
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
    Melakukan backward filtering untuk IIR
    Args:
        b (np.ndarray): Koefisien feedforward
        a (np.ndarray): Koefisien feedback
        x (np.ndarray): Input signal
    Returns:
        np.ndarray: Output dari filter
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

# def segment_ecg(data, threshold_p=[15, 21], threshold_q=[-25, -35], threshold_r=[115, 140], threshold_s=[-60, -85], threshold_t=[32, 43], interval=80):
def segment_ecg(data, threshold_p=[15, 21], threshold_q=[-35, -25], threshold_r=[115, 140], threshold_s=[-85, -60], threshold_t=[32, 43], interval=80):
    """
    Segmentasi sinyal EKG berdasarkan threshold nilai P, Q, R, S, dan T dengan interval waktu.
    """
    segmentation = {
        'P': {'lokasi': [], 'nilai': []},
        'Q': {'lokasi': [], 'nilai': []},
        'R': {'lokasi': [], 'nilai': []},
        'S': {'lokasi': [], 'nilai': []},
        'T': {'lokasi': [], 'nilai': []}
    }

    last_detection = {
        'P': -interval,
        'Q': -interval,
        'R': -interval,
        'S': -interval,
        'T': -interval
    }

    for i, nilai in enumerate(data):
        # Deteksi gelombang P
        if threshold_p[0] <= nilai <= threshold_p[1] and i - last_detection['P'] > interval:
            segmentation['P']['lokasi'].append(i)
            segmentation['P']['nilai'].append(nilai)
            last_detection['P'] = i
            print(f"P terdeteksi pada indeks {i}, nilai {nilai}")

        # Deteksi gelombang Q
        if threshold_q[0] <= nilai <= threshold_q[1] and i - last_detection['Q'] > interval:
            segmentation['Q']['lokasi'].append(i)
            segmentation['Q']['nilai'].append(nilai)
            last_detection['Q'] = i
            print(f"Q terdeteksi pada indeks {i}, nilai {nilai}")

        # Deteksi gelombang R
        if threshold_r[0] <= nilai <= threshold_r[1] and i - last_detection['R'] > interval:
            segmentation['R']['lokasi'].append(i)
            segmentation['R']['nilai'].append(nilai)
            last_detection['R'] = i
            print(f"R terdeteksi pada indeks {i}, nilai {nilai}")

        # Deteksi gelombang S
        if threshold_s[0] <= nilai <= threshold_s[1] and i - last_detection['S'] > interval:
            segmentation['S']['lokasi'].append(i)
            segmentation['S']['nilai'].append(nilai)
            last_detection['S'] = i
            print(f"S terdeteksi pada indeks {i}, nilai {nilai}")

        # Deteksi gelombang T
        if threshold_t[0] <= nilai <= threshold_t[1] and i - last_detection['T'] > interval:
            segmentation['T']['lokasi'].append(i)
            segmentation['T']['nilai'].append(nilai)
            last_detection['T'] = i
            print(f"T terdeteksi pada indeks {i}, nilai {nilai}")
        # else:
        #     print(f"Tidak ada gelombang terdeteksi pada indeks {i}, nilai {nilai}")

    return segmentation

def calculate_heart_rate(r_locations, duration):
    """
    Menghitung detak jantung (HR) berdasarkan lokasi puncak gelombang R.

    Args:
        r_locations (list): List yang berisi indeks lokasi puncak gelombang R.
        fs (float): Sampling frequency (Hz).
        duration (float): Durasi sinyal dalam detik.

    Returns:
        float: Detak jantung dalam BPM (Beats Per Minute).
               Mengembalikan 0 jika tidak ada gelombang R yang terdeteksi.
    """
    if not r_locations:
        return 0  # Tidak ada gelombang R yang terdeteksi

    # Hitung jumlah total gelombang R
    num_r_peaks = len(r_locations)
    # print("Jumlah gelombang R terdeteksi:", num_r_peaks)

    # Hitung detak jantung dalam BPM
    hr = (num_r_peaks / duration) * 60

    return hr