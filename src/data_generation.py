# Imports
import numpy as np
import hdf5storage as h5


class MeasurementData:

    def __init__(self, n_sam, d_y, d_z, d_theta, sig_e, sig_eta):
        self.n_sam = n_sam       # Number of data points
        self.d_y = d_y           # Dimension of observable variables
        self.d_z = d_z
        self.d_theta = d_theta   # Dimension of model parameters
        self.sig_e = sig_e       # Variance of measurement noise
        self.sig_eta = sig_eta   # Variance of predicting noise
        self.y_data = np.zeros(shape=(n_sam, d_y))  # Initialize the dataset
        self.y_scaled_data = np.zeros(shape=(n_sam, d_y))  # Initialize the scaled dataset
        self.z_data = np.zeros(shape=(n_sam, d_z))      # Initialize the predicted dataset
        self.z_scaled_data = np.zeros(shape=(n_sam, d_z))  # Initialize the predicted scaled dataset
        self.y_mean = np.zeros(shape=(1, d_y))  # Initialize the statistical mean of dataset
        self.y_std = np.zeros(shape=(1, d_y))  # Initialize the statistical standard deviation of dataset
        self.z_mean = np.zeros(shape=(1, d_z))  # Initialize the statistical mean of predictive data
        self.z_std = np.zeros(shape=(1, d_z))  # Initialize the statistical standard deviation of predictive data

    # For generating 1d observed data
    def generate_data_1d(self):
        n_sam = self.n_sam
        theta_data = np.random.randn(n_sam, 1)
        std_e = np.sqrt(self.sig_e)
        std_eta = np.sqrt(self.sig_eta)
        err_data = std_e*np.random.randn(n_sam, 1)
        eta_data = std_eta * np.random.randn(n_sam, 1)
        self.y_data = 2.*theta_data+err_data
        self.y_mean = np.mean(self.y_data, axis=0, keepdims=True)
        self.y_std = np.std(self.y_data, axis=0, keepdims=True)
        self.z_data = 3.*theta_data+eta_data
        self.z_mean = np.mean(self.z_data, axis=0, keepdims=True)
        self.z_std = np.std(self.z_data, axis=0, keepdims=True)
        self.y_scaled_data = MeasurementData.standardize_data(self.y_data, self.y_mean, self.y_std)
        self.z_scaled_data = MeasurementData.standardize_data(self.z_data, self.z_mean, self.z_std)
        return

    def generate_data_1d_case2(self):
        n_sam = self.n_sam
        theta_data = np.random.randn(n_sam, 1)
        std_e = np.sqrt(self.sig_e)
        std_eta = np.sqrt(self.sig_eta)
        err_data = std_e*np.random.randn(n_sam, 1)
        eta_data = std_eta * np.random.randn(n_sam, 1)
        self.y_data = 0.2*theta_data**2+0.1+err_data
        self.y_mean = np.mean(self.y_data, axis=0, keepdims=True)
        self.y_std = np.std(self.y_data, axis=0, keepdims=True)
        self.z_data = np.exp(theta_data)+0.2+eta_data
        self.z_mean = np.mean(self.z_data, axis=0, keepdims=True)
        self.z_std = np.std(self.z_data, axis=0, keepdims=True)
        self.y_scaled_data = MeasurementData.standardize_data(self.y_data, self.y_mean, self.y_std)
        self.z_scaled_data = MeasurementData.standardize_data(self.z_data, self.z_mean, self.z_std)
        return

    def generate_data_2d_case3(self):
        n_sam = self.n_sam
        d_theta = self.d_theta
        d_y = self.d_y
        d_z = self.d_z
        theta_data = np.random.randn(n_sam, d_theta)
        std_e = np.sqrt(self.sig_e)
        std_eta = np.sqrt(self.sig_eta)
        err_data = std_e*np.random.randn(n_sam, d_y)
        eta_data = std_eta * np.random.randn(n_sam, d_z)
        self.y_data = MeasurementData.f_fun(theta_data)+err_data
        self.y_mean = np.mean(self.y_data, axis=0, keepdims=True)
        self.y_std = np.std(self.y_data, axis=0, keepdims=True)
        self.z_data = MeasurementData.h_fun(theta_data)+eta_data
        self.z_mean = np.mean(self.z_data, axis=0, keepdims=True)
        self.z_std = np.std(self.z_data, axis=0, keepdims=True)
        self.y_scaled_data = MeasurementData.standardize_data(self.y_data, self.y_mean, self.y_std)
        self.z_scaled_data = MeasurementData.standardize_data(self.z_data, self.z_mean, self.z_std)
        return

    @staticmethod
    def h_fun(x):
        h1 = np.exp(x[:, 0]) + 0.2
        h2 = np.exp(x[:, 1]) + 0.1
        return np.stack([h1, h2], axis=1)

    @staticmethod
    def f_fun(x):
        f1 = 2. * x[:, 0] ** 2 + 2.
        f2 = x[:, 1] ** 4 + x[:, 1] + 1.
        return np.stack([f1, f2], axis=1)

    # Preprocess the data: standardization
    @staticmethod
    def standardize_data(y_data, y_mean, y_std):
        one_array = np.ones(shape=(y_data.shape[0], 1))
        y_std_data = (y_data-np.kron(one_array, y_mean))/np.kron(one_array, y_std)
        return y_std_data

    # Save data
    def save_data(self, file_path, file_name):
        data_dic = {'y_data': self.y_data,
                    'y_scaled_data': self.y_data,
                    'z_data': self.z_data,
                    'z_scaled_data': self.z_data,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'z_mean': self.z_mean,
                    'z_std': self.z_std}
        h5.write(data=data_dic, path=file_path, filename=file_name)
        return
