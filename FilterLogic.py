import numpy as np

def DFT(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X


def IDFT(X):
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    x = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            x[n] += X[k] * np.exp(angle)
        x[n] /= N
    return x

def LPF(fc, orde, fs):
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
    y= np.zeros_like(x)
    for n in range(len(x)):
        for m in range(len(h)):
            if n - m >= 0:
                y[n] += h[m] * x[n - m]
    return y

def backward_filter(h, x):
    y = np.zeros_like(x)
    for n in range(len(x)):
        for m in range(len(h)):
            if n + m < len(x):
                y[n] += h[m] * x[n + m]
    return y