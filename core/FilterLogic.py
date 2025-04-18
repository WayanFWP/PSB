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
        X (np.ndarray): Input array.
    Returns:
        np.ndarray: IDFT of the input array.
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

def LPF(fc, orde, fs):   
    """
    Compute the coefficients of a low-pass filter using the sinc function.
    Args:
        fc (float): Cutoff frequency.
        orde (int): Filter order.
        fs (float): Sampling frequency.
    Returns:
        np.ndarray: Filter coefficients.
    """ 
    omega_c = 2 * np.pi * fc / fs
    M = (orde - 1) // 2
    h = np.zeros(orde)
    for n in range(-M, M + 1):
        if n == 0:
            h[n + M] = omega_c / np.pi
        else:
            h[n + M] = np.sin(omega_c * n) / (np.pi * n)
    return h

def HPF(fc, orde, fs):
    """
    Compute the coefficients of a high-pass filter using the sinc function.
    Args:
        fc (float): Cutoff frequency.
        orde (int): Filter order.
        fs (float): Sampling frequency.
    Returns:
        np.ndarray: Filter coefficients.
    """
    omega_c = 2 * np.pi * fc / fs
    M = (orde - 1) // 2
    h = np.zeros(orde)
    for n in range(-M, M + 1):
        if n == 0:
            h[n + M] = 1 - omega_c / np.pi
        else:
            h[n + M] = -np.sin(omega_c * n) / (np.pi * n)
    return h

def BPF(fc1, fc2, orde, fs):
    """
    Compute the coefficients of a band-pass filter using the sinc function.
    Args:
        fc1 (float): Lower cutoff frequency.
        fc2 (float): Upper cutoff frequency.
        orde (int): Filter order.
        fs (float): Sampling frequency.
    Returns:
        np.ndarray: Filter coefficients.
    """
    omega_c1 = 2 * np.pi * fc1 / fs
    omega_c2 = 2 * np.pi * fc2 / fs
    M = (orde - 1) // 2
    h = np.zeros(orde)
    for n in range(-M, M + 1):
        if n == 0:
            h[n + M] = (omega_c2 - omega_c1) / np.pi
        else:
            h[n + M] = (np.sin(omega_c2 * n) - np.sin(omega_c1 * n)) / (np.pi * n)
    return h

def forward_filter(h, x):
    """
    Apply a forward filter to the input signal.
    Args:
        h (np.ndarray): Filter coefficients.
        x (np.ndarray): Input signal.
    Returns:
        np.ndarray: Filtered output signal.
    """
    y= np.zeros_like(x)
    for n in range(len(x)):
        for m in range(len(h)):
            if n - m >= 0:
                y[n] += h[m] * x[n - m]
    return y

def backward_filter(h, x):
    """
    Apply a backward filter to the input signal.
    Args:
        h (np.ndarray): Filter coefficients.
        x (np.ndarray): Input signal.
    Returns:
        np.ndarray: Filtered output signal.
    """
    y = np.zeros_like(x)
    for n in range(len(x)):
        for m in range(len(h)):
            if n + m < len(x):
                y[n] += h[m] * x[n + m]
    return y

def moving_average(data, window_size):
    """
    Compute the moving average of a 1D array.
    Args:
        data (np.ndarray): Input array.
        window_size (int): Size of the moving average window.
    Returns:
        np.ndarray: Moving average of the input array.
    """
    N = len(data)
    y = []

    for n in range(N):
        sum_val = 0
        count = 0
        for k in range(M):
            if n - k >= 0:
                sum_val += data[n - k]
                count += 1
        y.append(sum_val / count)
    return y    