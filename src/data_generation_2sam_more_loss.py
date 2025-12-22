# Imports
import time
import numpy as np
import hdf5storage as h5
import matplotlib.pyplot as plt
import scipy.stats as stats
import fem_preprocess as fp
import fem_solver as fs
import fem_solver_tf as fstf
import fem_postprocess as fpp
import tensorflow as tf


class MeasurementData:

    theta_mean = np.zeros((2, ))
    theta_std = np.ones((2, ))

    node_id = 231
    ele_id, nipt_id = 12, np.array([1, 3], dtype=int)
    num_parallel_cores = 10

    def __init__(self, n_sam, ne_sam, d_y, d_z, d_theta, sig_e, sig_eta):
        self.n_sam = n_sam       # Number of data points
        self. ne_sam = ne_sam    # Number of reparameterization sampling points
        self.d_y = d_y           # Dimension of observable variables
        self.d_theta = d_theta   # Dimension of model parameters
        self.d_z = d_z          # Dimension of predictive variables
        self.sig_e = sig_e       # Variance of measurement noise
        self.sig_eta = sig_eta   # Variance of predicting noise
        self.e_data = np.zeros(shape=(n_sam, d_theta))  # Initialize the seed data in Eq. 21
        self.y_data = np.zeros(shape=(n_sam, d_y))  # Initialize the dataset
        self.y_scaled_data = np.zeros(shape=(n_sam, d_y))  # Initialize the scaled dataset
        self.z_data = np.zeros(shape=(n_sam, d_z))      # Initialize the predicted dataset
        self.log_z_data = np.zeros(shape=(n_sam, d_z))
        self.z_scaled_data = np.zeros(shape=(n_sam, d_z))  # Initialize the predicted scaled dataset
        self.y_mean = np.zeros(shape=(1, d_y))  # Initialize the statistical mean of dataset
        self.y_std = np.zeros(shape=(1, d_y))  # Initialize the statistical standard deviation of dataset
        self.z_mean = np.zeros(shape=(1, d_z))  # Initialize the statistical mean of predictive data
        self.z_std = np.zeros(shape=(1, d_z))  # Initialize the statistical standard deviation of predictive data
        '''self.log_z_mean_post = np.zeros(shape=(n_sam, d_z))
        self.log_z_sig_post = np.zeros(shape=(n_sam, d_z))'''

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
        self.e_data = np.random.randn(self.ne_sam, self.d_theta)
        return

    # For FEM problem usage
    def generate_data_fem(self):
        n_sam = self.n_sam
        ne_sam = self.ne_sam
        theta_data = np.random.randn(n_sam, self.d_theta)

        std_e = np.sqrt(self.sig_e)
        std_eta = np.sqrt(self.sig_eta)
        err_data = std_e * np.random.randn(n_sam, self.d_y)
        eta_data = std_eta * np.random.randn(n_sam, self.d_z)
        self.e_data = np.random.randn(ne_sam, self.d_theta)
        start = time.time()
        fem_fh_fun_loop_rev = tf.function(MeasurementData.fem_fh_fun_loop_rev)
        f_data, h_data = fem_fh_fun_loop_rev(tf.constant(theta_data, dtype=tf.float64))
        end = time.time()
        self.y_data = f_data.numpy() + err_data
        print(f'Has generated {n_sam} measurement and predictive data points...')
        print(f'Takes {end-start} seconds.')
        '''for i in range(n_sam):
            self.y_data[i, :] = MeasurementData.fem_f_fun(theta_data[i, :]) + err_data[i, :]
            if (i+1) % 50 == 0:
                print(f'Has generated {i+1} measurement data points...')'''
        self.y_mean = np.mean(self.y_data, axis=0, keepdims=True)
        self.y_std = np.std(self.y_data, axis=0, keepdims=True)

        '''for i in range(n_sam):
            self.z_data[i, :] = MeasurementData.fem_h_fun(theta_data[i, :]) + eta_data[i, :]
            if (i+1) % 50 == 0:
                print(f'Has generated {i+1} predictive data points...')'''
        self.z_data = h_data.numpy() + eta_data
        self.log_z_data = np.log(self.z_data)
        self.z_mean = np.mean(self.z_data, axis=0, keepdims=True)
        self.z_std = np.std(self.z_data, axis=0, keepdims=True)
        return

    @classmethod
    def fem_h_fun(cls, x):
        try:
            ele_id, nipt_id = cls.ele_id, cls.nipt_id
            fp.PreProcessing.out_data = {}
            fp.PreProcessing.assign_storage_topopt()
            fp.PreProcessing.model_data['material'][0]['E'] = np.exp(cls.theta_std[0]*x[0]+cls.theta_mean[0])
            fp.PreProcessing.model_data['material'][0]['v'] = 0.5 / (1. +
                                                                     np.exp(-cls.theta_std[1]*x[1]-cls.theta_mean[1]))
            fs.FemSolver.fea_solution(input_data=None)
            step_id = len(fp.PreProcessing.out_data['step'])
            return fpp.PostProcessing.von_mises_stress(step_id, ele_id, nipt_id)
        except Exception as err:
            print('Input dimension is not appropriate...'+repr(err))

    @classmethod
    def fem_f_fun(cls, x):
        try:
            node_id = cls.node_id
            fp.PreProcessing.out_data = {}
            fp.PreProcessing.assign_storage_topopt()
            fp.PreProcessing.model_data['material'][0]['E'] = np.exp(cls.theta_std[0]*x[0]+cls.theta_mean[0])
            fp.PreProcessing.model_data['material'][0]['v'] = 0.5 / (1. +
                                                                     np.exp(-cls.theta_std[1]*x[1]-cls.theta_mean[1]))
            fs.FemSolver.fea_solution(input_data=None)
            return fp.PreProcessing.out_data['step'][-1]['nodal_disp'][:, node_id-1]
        except Exception as err:
            print('Input dimension is not appropriate...'+repr(err))

    @classmethod
    @tf.function
    def fem_f_fun_loop(cls, x):
        node_id = cls.node_id
        x_res = tf.reshape(x, (-1, x.shape[-1]))
        y = tf.zeros(x_res.shape, dtype=tf.float64)
        for i in tf.range(x_res.shape[0], dtype=tf.int64):
            fp.PreProcessing.model_data['material'][0]['E'] = tf.exp(cls.theta_std[0]*x_res[i, 0]+cls.theta_mean[0])
            fp.PreProcessing.model_data['material'][0]['v'] = 0.5/(1.+tf.exp(-cls.theta_std[1]*x_res[i, 1] -
                                                                             cls.theta_mean[1]))
            fstf.FemSolver.fea_solution(input_data=None)
            indices = tf.stack((i*tf.ones(y.shape[1], dtype=tf.int64), tf.range(y.shape[1], dtype=tf.int64)), axis=1)
            updates = tf.squeeze(fp.PreProcessing.sol_data['u_n1'][2*node_id - 2:2*node_id])
            y = tf.tensor_scatter_nd_update(y, indices, updates)
            # y[i, :] = fp.PreProcessing.out_data['step'][-1]['nodal_disp'][:, node_id - 1]
        return tf.reshape(y, x.shape)

    @staticmethod
    @tf.function
    def fem_fh_fun_loop(x):
        ele_id, nipt_id, node_id = MeasurementData.ele_id, MeasurementData.nipt_id, MeasurementData.node_id
        step_id = 2
        # x_res = tf.reshape(x, (-1, x.shape[-1]))
        y = tf.zeros(x.shape, dtype=tf.float64)
        h = tf.zeros(x.shape, dtype=tf.float64)
        for i in tf.range(x.shape[0], dtype=tf.int64):
            fp.PreProcessing.model_data['material'][0]['E'] = tf.exp(MeasurementData.theta_std[0] * x[i, 0] +
                                                                     MeasurementData.theta_mean[0])
            fp.PreProcessing.model_data['material'][0]['v'] = 0.5 / (1. +
                                                                     tf.exp(-MeasurementData.theta_std[1] * x[i, 1] -
                                                                            MeasurementData.theta_mean[1]))
            fstf.FemSolver.fea_solution(input_data=None)

            indices = tf.stack((i * tf.ones(y.shape[1], dtype=tf.int64), tf.range(y.shape[1], dtype=tf.int64)), axis=1)
            updates = tf.squeeze(fp.PreProcessing.sol_data['u_n1'][2 * node_id - 2:2 * node_id])
            y = tf.tensor_scatter_nd_update(y, indices, updates)

            indices = tf.stack((i * tf.ones(y.shape[1], dtype=tf.int64), tf.range(y.shape[1], dtype=tf.int64)), axis=1)
            updates = fpp.PostProcessing.von_mises_stress_tf(step_id, ele_id, nipt_id)
            h = tf.tensor_scatter_nd_update(h, indices, updates)
        return tf.reshape(y, x.shape), tf.reshape(h, x.shape)

    @staticmethod
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float64),))
    def fem_fh_fun_loop_rev(x):
        # x_res = tf.reshape(x, (-1, x.shape[-1]))
        y, h = tf.map_fn(MeasurementData.fem_fh_fun_one_loop, x, fn_output_signature=[tf.float64, tf.float64],
                         parallel_iterations=MeasurementData.num_parallel_cores)
        return y, h

    @staticmethod
    def fem_fh_fun_one_loop(x):
        ele_id, nipt_id, node_id = MeasurementData.ele_id, MeasurementData.nipt_id, MeasurementData.node_id
        step_id = 2
        x_res = tf.squeeze(x)
        fp.PreProcessing.model_data['material'][0]['E'] = tf.exp(
            MeasurementData.theta_std[0] * x_res[0] + MeasurementData.theta_mean[0])
        fp.PreProcessing.model_data['material'][0]['v'] = 0.5 / (
                1. + tf.exp(-MeasurementData.theta_std[1] * x_res[1] -
                            MeasurementData.theta_mean[1]))
        fstf.FemSolver.fea_solution(input_data=None)

        y = tf.squeeze(fp.PreProcessing.sol_data['u_n1'][2 * node_id - 2:2 * node_id])
        h = fpp.PostProcessing.von_mises_stress_tf(step_id, ele_id, nipt_id)

        return [y, h]

    # Plot PDF for predictive variables z
    def plot_z_data(self, mf, n_points):
        z_mean = np.squeeze(self.z_mean)
        z_std = np.squeeze(self.z_std)
        z_points = np.linspace(z_mean - mf * z_std, z_mean + mf * z_std, n_points)
        kde = stats.gaussian_kde(np.squeeze(self.z_data))
        zpdf = kde(z_points)
        fig, ax = plt.subplots(1, 1)
        ax.plot(z_points, zpdf, 'b-', label='KDE')
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')

    # Preprocess the data: standardization
    @staticmethod
    def standardize_data(y_data, y_mean, y_std):
        one_array = np.ones(shape=(y_data.shape[0], 1))
        y_std_data = (y_data-np.kron(one_array, y_mean))/np.kron(one_array, y_std)
        return y_std_data

    @staticmethod
    def h_fun_1d_case1(theta_sam):
        h_sam = 3.*theta_sam
        return h_sam

    @staticmethod
    def h_fun_1d_case2(theta_sam):
        h_sam = np.exp(theta_sam) + 0.2
        return h_sam

    @staticmethod
    def f_fun_1d_case2(theta_sam):
        f_sam = 2. * theta_sam ** 2 + 2.
        return f_sam

    @staticmethod
    def f_fun_2d_case3(x):
        if len(x.shape) == 2:
            f1 = 2. * x[:, 0] ** 2 + 2.
            f2 = x[:, 1] ** 4 + x[:, 1] + 1.
            return np.stack([f1, f2], axis=1)
        elif len(x.shape) == 3:
            f1 = 2. * x[:, :, 0] ** 2 + 2.
            f2 = x[:, :, 1] ** 4 + x[:, :, 1] + 1.
            return np.stack([f1, f2], axis=2)
        else:
            raise ValueError('The dimension of input x is not correct.')

    @staticmethod
    def h_fun_2d_case3(x):
        if len(x.shape) == 2:
            h1 = np.exp(x[:, 0]) + 0.2
            h2 = np.exp(x[:, 1]) + 0.1
            return np.stack([h1, h2], axis=1)
        elif len(x.shape) == 3:
            h1 = np.exp(x[:, :, 0]) + 0.2
            h2 = np.exp(x[:, :, 1]) + 0.1
            return np.stack([h1, h2], axis=2)
        else:
            raise ValueError('The dimension of input x is not correct.')

    # Save data
    def save_data(self, file_path, file_name):
        data_dic = {'y_data': self.y_data,
                    'y_scaled_data': self.y_data,
                    'z_data': self.z_data,
                    'log_z_data': self.log_z_data,
                    'z_scaled_data': self.z_data,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'z_mean': self.z_mean,
                    'z_std': self.z_std,
                    'e_data': self.e_data}
        h5.write(data=data_dic, path=file_path, filename=file_name)
        return
