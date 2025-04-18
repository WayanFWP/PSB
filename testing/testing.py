import numpy as np

def low_pass_filter(X, cutoff):
    N = len(X)
    X_filtered = np.copy(X)
    for k in range(N):
        if > cutoff and k < N - cutoff:
            X_filtered[k] = 0
    return X_filtered

def dft_manual(x):
    x = np.asarray(x, dtype=np.complex128)  # pastikan input berupa array kompleks
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

def idft_manual(X):
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    x = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            x[n] += X[k] * np.exp(angle)
        x[n] /= N
    return x

def high_pass_filter(X, cutoff):
    N = len(X)
    X_filtered = np.copy(X)
    for k in range(N):
        if k < cutoff or k > N - cutoff:
            X_filtered[k] = 0
    return X_filtered

def band_pass_filter(X, low_cutoff, high_cutoff):
    N = len(X)
    X_filtered = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        if low_cutoff <= k <= high_cutoff or (N - high_cutoff) <= k <= (N - low_cutoff):
            X_filtered[k] = X[k]
    return X_filtered


data = np.array([1, 2, 3, 4, 3, 2, 1, 0])
X = dft_manual(data)

X_lpf = low_pass_filter(X, cutoff=2)
X_hpf = high_pass_filter(X, cutoff=2)
X_bpf = band_pass_filter(X, low_cutoff=2, high_cutoff=4)

x_lpf = np.real(idft_manual(X_lpf))
x_hpf = np.real(idft_manual(X_hpf))
x_bpf = np.real(idft_manual(X_bpf))

print("Asli:", data)
print("LPF:", x_lpf)
print("HPF:", x_hpf)
print("Bandpass:", x_bpf)
