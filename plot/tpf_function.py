def tpf_theory(T, k, J=1, N=32):
    th = np.tanh(J / T)
    return (th**k + th**(N-k)) / (1 + th**N)
