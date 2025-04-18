import numpy as np

def DFT(x):
    x = np.asarray(x, dtype=np.complex128)  # pastikan input berupa array kompleks
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