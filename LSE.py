import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

class LeastSquaresEstimator:
    def __init__(self, N=1, A=2.0):
        self.N = N
        self.A = A

    def estimate(self, snr_db_range):
        errors, mean_square_errors = [], []
        for snr in snr_db_range:
            noise_power = 10 ** (-snr / 10)
            w = np.random.normal(0, np.sqrt(noise_power), self.N)
            h = np.ones(self.N)
            x = self.A * h + w
            theta_ls = np.sum(x * h) / np.sum(h ** 2)
            error = np.abs(self.A - theta_ls)
            mse = (self.A - theta_ls) ** 2
            errors.append(error)
            mean_square_errors.append(mse)
        return errors, mean_square_errors

    def plot_results(self, snr_db_range, errors, mean_square_errors):
        plt.figure(figsize=(10, 6))
        plt.plot(snr_db_range, errors, 'o-', label='Absolute Error', color='blue')
        plt.plot(snr_db_range, mean_square_errors, 's-', label='Mean Square Error', color='orange')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Error')
        plt.title('Error vs SNR for Least Squares Estimation')
        plt.ylim(0, self.A)
        plt.grid()
        plt.legend()
        plt.show()

ls_estimator = LeastSquaresEstimator()
snr_db_range = np.linspace(0, 25, 10)
errors, mean_square_errors = ls_estimator.estimate(snr_db_range)
ls_estimator.plot_results(snr_db_range, errors, mean_square_errors)