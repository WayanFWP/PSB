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
    Wc = 2 * np.pi * fc
    T = 1 / fs
    prewarp = 2 / T * np.tan(Wc * T / 2)

    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = prewarp * (-np.sin(theta) + 1j * np.cos(theta))
        poles.append(pole)

    a = np.array([1.0])
    b = np.array([1.0])
    for p in poles:
        k = (2 / T + p) / (2 / T - p)
        a = np.convolve(a, [1, -k.real])
        b = np.convolve(b, [1, 1])

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
    Wc = 2 * np.pi * fc
    T = 1 / fs
    prewarp = 2 / T * np.tan(Wc * T / 2)

    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = prewarp * (-np.sin(theta) + 1j * np.cos(theta))
        poles.append(pole)

    a = np.array([1.0])
    b = np.array([1.0])
    for p in poles:
        k = (2 / T - p) / (2 / T + p)  # Different from LPF!
        a = np.convolve(a, [1, -k.real])
        b = np.convolve(b, [1, -1])

    gain = np.sum(b) / np.sum(a)
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
    b = b / gain

    return b.real, a.real

def BSF(fc1, fc2, order, fs):
    """
    Design a Butterworth Band-Stop Filter (BSF, notch) using bilinear transform.
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
        raise ValueError("Order of BSF must be even.")

    W1 = 2 * np.pi * fc1
    W2 = 2 * np.pi * fc2
    T = 1 / fs

    W1p = 2 / T * np.tan(W1 * T / 2)
    W2p = 2 / T * np.tan(W2 * T / 2)

    W0 = np.sqrt(W1p * W2p)
    Bw = W2p - W1p

    poles = []
    for k in range(order):
        theta = np.pi * (2 * k + 1) / (2 * order)
        pole = Bw/2 * (-np.sin(theta) + 1j * np.cos(theta))
        poles.append(pole)

    a = np.array([1.0])
    b = np.array([1.0])
    for p in poles:
        s_pole = p + W0**2 / p  # Bandstop transformation
        k = (2 / T - s_pole) / (2 / T + s_pole)  # Different from BPF!
        a = np.convolve(a, [1, -k.real])
        b = np.convolve(b, [1, 1])

    gain = np.sum(b) / np.sum(a)
    b = b / gain

    return b.real, a.real

def forward_filter(b, a, x):
    """
    Apply an IIR filter in the forward direction (Direct Form I).
    Args:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
        x (np.ndarray): Input signal.
    Returns:
        np.ndarray: Filtered signal.
    """
    y = np.zeros_like(x)
    for n in range(len(x)):
        for i in range(len(b)):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, len(a)):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
        y[n] /= a[0]
    return y

def backward_filter(b, a, x):
    """
    Apply an IIR filter in the backward direction.
    Args:
        b (np.ndarray): Feedforward coefficients.
        a (np.ndarray): Feedback coefficients.
        x (np.ndarray): Input signal.
    Returns:
        np.ndarray: Filtered signal.
    """
    x_rev = x[::-1]  # Reverse the signal
    y_rev = forward_filter(b, a, x_rev)
    y = y_rev[::-1]  # Reverse the output
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
    y_forward = forward_filter(b, a, x)
    y = backward_filter(b, a, y_forward)
    return y

def moving_average(data, window_size):
    """
    Compute the moving average of a 1D array.
    Args:
        data (np.ndarray): Input signal.
        window_size (int): Moving average window size.
    Returns:
        np.ndarray: Smoothed signal.
    """
    N = len(data)
    y = []

    for n in range(N):
        sum_val = 0
        count = 0
        for k in range(window_size):
            if n - k >= 0:
                sum_val += data[n - k]
                count += 1
        y.append(sum_val / count)
    return np.array(y)

def square_wave_signal(data, threshold):
    """
    Generate a square wave signal based on a threshold.
    Args:
        data (np.ndarray): Input signal.
        threshold (float): Threshold for switching to 1.
    Returns:
        np.ndarray: Square wave output.
    """
    square_wave = np.zeros_like(data)
    for i in range(len(data)):
        if data[i] > threshold:
            square_wave[i] = 1
    return square_wave
