import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

class WienerFilter:
    def __init__(self, N=4000, Ac=1, wc=0.1 * np.pi, sigma_w=0.5):
        self.N = N
        self.Ac = Ac
        self.wc = wc
        self.sigma_w = sigma_w

    def apply_filter(self):
        phi_n = np.random.uniform(0, 2 * np.pi, self.N)
        s = self.Ac * np.cos(self.wc * np.arange(self.N) + phi_n)
        w = np.random.normal(0, self.sigma_w, self.N)
        x = s + w
        Rxx = np.correlate(x, x, mode='full') / self.N
        Rxx = Rxx[self.N - 1:]
        Rxx_matrix = toeplitz(Rxx[:100])
        Rww_matrix = (self.sigma_w ** 2) * np.eye(100)
        W = Rxx_matrix @ np.linalg.inv(Rxx_matrix + Rww_matrix)
        s_hat = np.convolve(x, W[0, :], mode='same')
        return s, s_hat

    def plot_results(self, s, s_hat, graph_length=100):
        plt.figure(figsize=(20, 5))
        plt.plot(s[:graph_length], label='Original Signal (s)')
        plt.plot(s_hat[:graph_length], label='Estimated Signal (s_hat)', linestyle='dashed')
        plt.xlabel('n')
        plt.ylabel('Amplitude')
        plt.title('Wiener Filter Smoothing')
        plt.legend()
        plt.show()

# Wiener Filter Experiment
wiener_filter = WienerFilter()
s, s_hat = wiener_filter.apply_filter()
wiener_filter.plot_results(s, s_hat)