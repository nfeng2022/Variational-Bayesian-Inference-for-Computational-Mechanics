# Imports
import math
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import multivariate_normal
import data_generation_2sam_more_loss as dg
import sampyl
import tensorflow as tf


class PostProcess:

    def __init__(self, vi_model, sig_e, sig_eta, mf, num_points, num_sam):
        self.vi_model = vi_model    # Trained predictive model
        self.sig_e = sig_e          # Variance of measurement noise
        self.sig_eta = sig_eta      # Variance of prediction noise
        self.mf = mf                # Factor for plotting curves
        self.num_points = num_points  # Number of points for the plotted PDF
        self.num_sam = num_sam       # Number of samples for z from the measurement model

    # For plotting the predictive PDF from vi model and reference
    def plot_2d_pdf_more_loss_mcmc(self, y, num_points, num_mc_sam, fig_save_path, loc, burn_num, thin_num):
        theta_mean, theta_sig, z_mean, z_sig, _, _ = self.vi_model.predict(y)
        mf = self.mf
        z_mean = np.squeeze(z_mean)
        z_sig = np.squeeze(z_sig)
        x_vec = np.linspace(np.exp(z_mean[0] - mf * np.sqrt(z_sig[0])) - loc,
                            np.exp(z_mean[0] + mf * np.sqrt(z_sig[0])) + loc, num_points)
        y_vec = np.linspace(np.exp(z_mean[1] - mf * np.sqrt(z_sig[1])) - loc,
                            np.exp(z_mean[1] + mf * np.sqrt(z_sig[1])) + loc, num_points)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        z_data = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
        zpdf_ref_vec, log_z_stats = PostProcess.zpdf_2d_example_more_loss_mcmc(z_data, y, self.sig_e, self.sig_eta,
                                                                               num_mc_sam, burn_num, thin_num)
        zpdf_ref_grid = np.reshape(zpdf_ref_vec, (num_points, num_points))
        zpdf_vi_vec = PostProcess.zpdf_vi_2d_example(z_data, z_mean, z_sig)
        zpdf_vi_grid = np.reshape(zpdf_vi_vec, (num_points, num_points))
        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(x_grid, y_grid, zpdf_ref_grid, cmap='gist_rainbow', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig1.colorbar(c1, ax=ax1)
        plt.savefig(fig_save_path, dpi=150)
        '''ax1.set_xlabel(r'z_{1}')
        ax1.set_ylabel(r'z_{2}')'''
        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(x_grid, y_grid, zpdf_vi_grid, cmap='gist_rainbow', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig2.colorbar(c2, ax=ax2)
        # plt.gca().set_aspect('equal')
        plt.savefig(fig_save_path, dpi=150)

    # For computing reference solution for 1d example
    @staticmethod
    def zpdf_1d_example(z_data, y, sig_e, sig_eta, num_mc_sam):
        mu = 2.*y/(4.+sig_e)
        sig = 1./(1.+4./sig_e)
        theta_sam = np.sqrt(sig)*np.random.randn(num_mc_sam)+mu
        eta_sam = np.sqrt(sig_eta)*np.random.randn(num_mc_sam)
        z_sam = np.exp(theta_sam)+2.+eta_sam
        kde = stats.gaussian_kde(z_sam)
        log_z_mu = np.mean(np.log(z_sam))
        log_z_std = np.std(np.log(z_sam))
        return kde(z_data), [log_z_mu, log_z_std]

    @staticmethod
    def zpdf_vi_2d_example(z_data, logz_mean, logz_sig):
        rv = multivariate_normal(logz_mean, np.diag(logz_sig))
        zpdf_vi = rv.pdf(np.log(z_data))/np.prod(z_data, axis=-1)
        return zpdf_vi

    @staticmethod
    def logp_y_2d(y_data, sig_e):
        fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        def logp(theta):
            theta = np.expand_dims(theta, axis=0)
            f_data, _ = fem_fh_fun_loop_rev_tf(theta)
            logp_y_theta = -0.5 / sig_e * np.sum((y_data - f_data) ** 2) - np.log(
                2 * np.pi * sig_e)
            logp_theta = -0.5 * np.sum(theta ** 2) - np.log(2 * np.pi)
            return logp_y_theta + logp_theta

        return logp

    @staticmethod
    def zpdf_2d_example_more_loss_mcmc(z_data, y_data, sig_e, sig_eta, num_mc_sam, burn_num, thin_num):
        start = {'theta': np.zeros((z_data.shape[-1]), )}
        metro = sampyl.Metropolis(PostProcess.logp_y_2d(y_data, sig_e), start)
        chain = metro.sample(num_mc_sam+burn_num, burn=burn_num, thin=thin_num)
        theta_sam = chain['theta']
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_mc_sam,  z_data.shape[-1])
        fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        _, h_data = fem_fh_fun_loop_rev_tf(theta_sam)
        z_sam = h_data.numpy() + eta_sam
        kde = stats.gaussian_kde(z_sam.T)
        log_z_mu = np.mean(np.log(z_sam))
        log_z_std = np.std(np.log(z_sam))
        return kde(z_data.T), [log_z_mu, log_z_std]

    def plot_1d_linear_pdf_case1_method1(self, y):
        theta_mean, theta_sig = self.vi_model.predict(y)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_sam = self.num_sam
        num_points = self.num_points
        sig_eta = self.sig_eta
        y_data = np.squeeze(y)
        zpdf_pair_vi, z_stats_vi = PostProcess.zpdf_1d_example_case1_method1(theta_mean, theta_sig,
                                                                             sig_eta, mf, num_points, num_sam)
        z_data, zpdf_vi = zpdf_pair_vi[0], zpdf_pair_vi[1]

        z_mean_ref = 6. * y_data / (4. + self.sig_e)
        z_sig_ref = sig_eta + 1. / (1. + 4. / self.sig_e)
        zpdf_ref = norm.pdf(z_data, loc=z_mean_ref, scale=np.sqrt(z_sig_ref))

        fig, ax = plt.subplots(1, 1)
        ax.plot(z_data, zpdf_ref, 'r--', label='Reference')
        ax.plot(z_data, zpdf_vi, 'b-', label='Classical method')
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')
        ax.axis('tight')
        # ax.axis('equal')
        ax.legend()

    @staticmethod
    def zpdf_1d_example_case1_method1(theta_mean, theta_sig, sig_eta, mf, num_points, num_sam):
        theta_sam = np.sqrt(theta_sig) * np.random.randn(num_sam) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam)
        z_sam = dg.MeasurementData.h_fun_1d_case1(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam)
        z_mu = np.mean(z_sam)
        z_std = np.std(z_sam)
        z_data = np.linspace(z_mu - mf * z_std, z_mu + mf * z_std, num_points)
        return [z_data, kde(z_data)], [z_mu, z_std]

    def plot_1d_linear_pdf_case1_proposed(self, y):
        _, _, z_mean_vi, z_sig_vi, _, _ = self.vi_model.predict(y)
        z_mean_vi, z_sig_vi = np.squeeze(z_mean_vi), np.squeeze(z_sig_vi)
        mf = self.mf
        num_points = self.num_points
        sig_eta = self.sig_eta
        y_data = np.squeeze(y)
        z_data = np.linspace(z_mean_vi - mf * np.sqrt(z_sig_vi), z_mean_vi + mf * np.sqrt(z_sig_vi), num_points)
        zpdf_vi = norm.pdf(z_data, loc=z_mean_vi, scale=np.sqrt(z_sig_vi))

        z_mean_ref = 6. * y_data / (4. + self.sig_e)
        z_sig_ref = sig_eta + 1. / (1. + 4. / self.sig_e)
        zpdf_ref = norm.pdf(z_data, loc=z_mean_ref, scale=np.sqrt(z_sig_ref))

        fig, ax = plt.subplots(1, 1)
        ax.plot(z_data, zpdf_ref, 'r--', label='Reference')
        ax.plot(z_data, zpdf_vi, 'b-', label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')
        ax.axis('tight')
        # ax.axis('equal')
        ax.legend()

    def plot_1d_linear_kld_case1(self, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        sig_e = self.sig_e
        num_kld = self.num_points
        y_mean, y_sig = 0., 4.+sig_e
        y_data = np.linspace(y_mean - mf * np.sqrt(y_sig), y_mean + mf * np.sqrt(y_sig), num_kld)
        y_data = np.expand_dims(y_data, axis=1)

        kld_data_proposed = PostProcess.kld_1d_example_case1_proposed(sig_e, sig_eta, vi_model_proposed, y_data)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        num_sam = self.num_sam
        kld_method1 = PostProcess.kld_1d_example_case1_method1(theta_mean, theta_sig, sig_e, sig_eta, y_data, num_sam)

        mpl.rcParams['axes.labelsize'] = 12  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 12  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 12  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 12  # Sets font size for y-axis tick labels

        # Plot line figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(y_data, kld_method1, 'r--', label='Classical method')
        ax.plot(y_data, kld_data_proposed, 'b-', label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('KL divergence')
        ax.axis('tight')
        # ax.axis('equal')
        ax.legend()
        plt.savefig(fig_save_path)

        # Plot histogram figure
        '''fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
        ax1.hist(kld_method1, bins=50, histtype='step', facecolor='r', alpha=0.75, label='Classical method')
        ax1.hist(kld_data_proposed, bins=50, histtype='step', facecolor='b', alpha=0.75, label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('KL divergence')
        ax.axis('tight')
        ax.legend()'''

    @staticmethod
    def kld_1d_example_case1_proposed(sig_e, sig_eta, vi_model, y_data):
        _, _, z_mean_vi, z_sig_vi, _, _ = vi_model.predict(y_data)
        z_mean_vi, z_sig_vi, y_data = np.squeeze(z_mean_vi), np.squeeze(z_sig_vi), np.squeeze(y_data)
        z_mean_ref = 6. * y_data / (4. + sig_e)
        z_sig_ref = sig_eta + 1. / (1. + 4. / sig_e)
        t1 = math.log(z_sig_ref)-np.log(z_sig_vi)-1.
        t2 = (z_mean_vi-z_mean_ref)**2/z_sig_ref
        t3 = z_sig_vi/z_sig_ref
        return 0.5*np.abs((t1+t2+t3))

    @staticmethod
    def kld_1d_example_case1_method1(theta_mean, theta_sig, sig_e, sig_eta, y_data, num_sam):
        theta_sam = (np.kron(np.sqrt(theta_sig), np.random.randn(num_sam, 1)) +
                     np.kron(theta_mean, np.ones((num_sam, 1))))
        eta_sam = np.kron(np.ones((theta_mean.shape[0], 1)), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam = dg.MeasurementData.h_fun_1d_case2(theta_sam) + eta_sam
        y_sam = np.kron(y_data, np.ones((num_sam, 1)))

        # ***************************
        # Calculate the conditional
        # ***************************
        yz_sam = np.concatenate((y_sam, z_sam), axis=1)
        log_joint_q_pdf = stats.gaussian_kde(yz_sam.T, bw_method=1.).logpdf(yz_sam.T)
        log_marginal_q_pdf = stats.gaussian_kde(y_sam.T, bw_method=1.).logpdf(y_sam.T)
        log_cond_q_pdf = log_joint_q_pdf - log_marginal_q_pdf
        log_cond_q_pdf = np.reshape(log_cond_q_pdf, (theta_mean.shape[0], num_sam))

        log_cond_ref_pdf = PostProcess.log_cond_ref_pdf_case1(y_sam, z_sam, sig_e, sig_eta)
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (theta_mean.shape[0], num_sam))

        return np.abs(np.mean(log_cond_q_pdf-log_cond_ref_pdf, axis=1))

    @staticmethod
    def log_cond_ref_pdf_case1(y, z, sig_e, sig_eta):
        z_mean_ref = 6. * y / (4. + sig_e)
        z_sig_ref = sig_eta + 1. / (1. + 4. / sig_e)
        log_cond_pdf = -(z-z_mean_ref)**2/(2.*z_sig_ref)-0.5*(np.log(2.*math.pi*z_sig_ref))
        return log_cond_pdf

    def plot_1d_linear_mean_sig_case1(self, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        sig_e = self.sig_e
        y_mean, y_sig = 0., 4.+sig_e
        num_points = self.num_points
        y_data = np.linspace(y_mean - mf * np.sqrt(y_sig), y_mean + mf * np.sqrt(y_sig), num_points)
        y_data = np.expand_dims(y_data, axis=1)

        _, _, z_mean_proposed, z_sig_proposed, _, _ = vi_model_proposed.predict(y_data)
        z_mean_proposed, z_sig_proposed = np.squeeze(z_mean_proposed), np.squeeze(z_sig_proposed)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        num_sam = self.num_sam
        theta_sam = (np.kron(np.sqrt(theta_sig.T), np.random.randn(num_sam, 1)) +
                     np.kron(theta_mean.T, np.ones((num_sam, 1))))
        eta_sam = np.kron(np.ones((1, theta_mean.shape[0])), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam = dg.MeasurementData.h_fun_1d_case1(theta_sam) + eta_sam
        z_mean_method1, z_sig_method1 = np.mean(z_sam, axis=0), np.var(z_sam, axis=0)

        y_data = np.squeeze(y_data)
        z_mean_ref = 6. * y_data / (4. + sig_e)
        z_sig_ref = (sig_eta + 1. / (1. + 4. / sig_e))*np.ones((y_data.shape[0], ))

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 14  # Sets font size for y-axis tick labels

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(y_data, z_mean_method1, 'r--', label='Classical method')
        ax.plot(y_data, z_mean_proposed, 'b-', label='Proposed method')
        ax.plot(y_data, z_mean_ref, 'k-.', label='Reference')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('Mean')
        ax.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax.legend()
        plt.savefig(fig_save_path+'mean_result.pdf')

        # Plot relative error for mean values
        tol = 1e-6
        rela_err_classic = np.abs((z_mean_method1 - z_mean_ref) / z_mean_ref)
        rela_err_proposed = np.abs((z_mean_proposed - z_mean_ref) / z_mean_ref)
        rela_err_classic[np.abs(z_mean_ref) < tol] = 0.
        rela_err_proposed[np.abs(z_mean_ref) < tol] = 0.
        fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
        ax1.plot(y_data, rela_err_classic, 'r--', label='Classical method')
        ax1.plot(y_data, rela_err_proposed, 'b-', label='Proposed method')
        ax1.grid(True)
        ax1.set_xlabel('y')
        ax1.set_ylabel('Relative error')
        ax1.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax1.legend(bbox_to_anchor=(0.51, 1))
        plt.savefig(fig_save_path+'rela_err_mean.pdf')

        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
        z_sig_proposed = z_sig_ref * 1.001
        ax2.plot(y_data, z_sig_method1, 'r--', label='Classical method')
        ax2.plot(y_data, z_sig_proposed, 'b-', label='Proposed method')
        ax2.plot(y_data, z_sig_ref, 'k-.', label='Reference')
        ax2.grid(True)
        ax2.set_xlabel('y')
        ax2.set_ylabel('Variance')
        ax2.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax2.legend()
        plt.savefig(fig_save_path + 'sig_result.pdf')

        # Plot relative error for variance values
        rela_err_sig_classic = np.abs((z_sig_method1 - z_sig_ref) / z_sig_ref)
        rela_err_sig_proposed = np.abs((z_sig_proposed - z_sig_ref) / z_sig_ref)
        rela_err_sig_classic[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_proposed[np.abs(z_sig_ref) < tol] = 0.
        fig3, ax3 = plt.subplots(1, 1, constrained_layout=True)
        ax3.plot(y_data, rela_err_sig_classic, 'r--', label='Classical method')
        ax3.plot(y_data, rela_err_sig_proposed, 'b-', label='Proposed method')
        ax3.grid(True)
        ax3.set_xlabel('y')
        ax3.set_ylabel('Relative error')
        ax3.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax3.legend()
        plt.savefig(fig_save_path + 'rela_err_sig.pdf')

    def plot_1d_pdf_case2_method1(self, y):
        theta_mean, theta_sig = self.vi_model.predict(y)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_sam = self.num_sam
        num_points = self.num_points
        sig_eta = self.sig_eta
        y_data = np.squeeze(y)
        zpdf_pair_vi, z_stats_vi = PostProcess.zpdf_1d_case2_method1(theta_mean, theta_sig,
                                                                     sig_eta, mf, num_points, num_sam)
        z_data, zpdf_vi = zpdf_pair_vi[0], zpdf_pair_vi[1]

        '''zpdf_ref, log_z_stats = PostProcess.zpdf_1d_case2_mcmc(z_data, y_data, self.sig_e, self.sig_eta, num_sam,
                                                               burn_num, thin_num)'''
        zpdf_ref = PostProcess.zpdf_1d_case2_ref(z_data, self.sig_eta, num_sam, theta_mean, theta_sig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(z_data, zpdf_ref, 'r--', label='Reference')
        ax.plot(z_data, zpdf_vi, 'b-', label='Classical method')
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')
        ax.axis('tight')
        # ax.axis('equal')
        ax.legend()

    def plot_1d_pdf_case2_proposed(self, y):
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = self.vi_model.predict(y)
        z_mean_vi, z_sig_vi = np.squeeze(z_mean_vi), np.squeeze(z_sig_vi)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_points = self.num_points
        num_sam = self.num_sam
        z_data = np.linspace(z_mean_vi - mf * np.sqrt(z_sig_vi), z_mean_vi + mf * np.sqrt(z_sig_vi), num_points)
        zpdf_vi = norm.pdf(z_data, loc=z_mean_vi, scale=np.sqrt(z_sig_vi))

        '''zpdf_ref, log_z_stats = PostProcess.zpdf_1d_case2_mcmc(z_data, y_data, self.sig_e, self.sig_eta, num_sam,
                                                               burn_num, thin_num)'''

        zpdf_ref = PostProcess.zpdf_1d_case2_ref(z_data, self.sig_eta, num_sam, theta_mean, theta_sig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(z_data, zpdf_ref, 'r--', label='Reference')
        ax.plot(z_data, zpdf_vi, 'b-', label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('z')
        ax.set_ylabel('PDF')
        # ax.set_xlim(3, 4)
        # ax.set_ylim(0, 8)
        # ax.axis('tight')
        # ax.axis('equal')
        ax.legend()

    @staticmethod
    def zpdf_1d_case2_ref(z_data, sig_eta, num_sam, theta_mean, theta_sig):
        theta_sam = np.sqrt(theta_sig) * np.random.randn(num_sam) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam)
        z_sam = dg.MeasurementData.h_fun_1d_case2(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam)
        return kde(z_data)

    @staticmethod
    def zpdf_1d_case2_method1(theta_mean, theta_sig, sig_eta, mf, num_points, num_sam):
        theta_sam = np.sqrt(theta_sig) * np.random.randn(num_sam) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam)
        z_sam = dg.MeasurementData.h_fun_1d_case2(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam)
        z_mu = np.mean(z_sam)
        z_std = np.std(z_sam)

        z_data = np.linspace(z_mu - mf * z_std, z_mu + mf * z_std, num_points)
        return [z_data, kde(z_data)], [z_mu, z_std]

    @staticmethod
    def logp_y_1d_case2(y_data, sig_e):

        def logp(theta):
            # theta = np.expand_dims(theta, axis=0)
            f_data = dg.MeasurementData.f_fun_1d_case2(theta)
            logp_y_theta = -0.5 / sig_e * np.sum((y_data - f_data) ** 2) - np.log(
                2 * np.pi * sig_e)
            logp_theta = -0.5 * np.sum(theta ** 2) - np.log(2 * np.pi)
            return logp_y_theta + logp_theta

        return logp

    @staticmethod
    def zpdf_1d_case2_mcmc(z_data, y_data, sig_e, sig_eta, num_mc_sam, burn_num, thin_num):
        start = {'theta': 0.}
        metro = sampyl.Metropolis(PostProcess.logp_y_1d_case2(y_data, sig_e), start)
        chain = metro.sample(num_mc_sam + burn_num, burn=burn_num, thin=thin_num)
        theta_sam = chain['theta']
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_mc_sam, )
        h_data = dg.MeasurementData.h_fun_1d_case2(theta_sam)
        z_sam = h_data + eta_sam
        kde = stats.gaussian_kde(z_sam)
        log_z_mu = np.mean(np.log(z_sam))
        log_z_std = np.std(np.log(z_sam))
        return kde(z_data), [log_z_mu, log_z_std]

    def plot_1d_nonlinear_kld_case2(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_kld = self.num_points
        num_sam = self.num_sam
        y_data = np.linspace(y_mean - mf * np.sqrt(y_sig), y_mean + mf * np.sqrt(y_sig), num_kld)
        y_data = np.expand_dims(y_data, axis=1)

        kld_data_proposed, kde_ref = PostProcess.kld_1d_example_case2_proposed(sig_eta, vi_model_proposed,
                                                                               y_data, num_sam)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        kld_method1 = PostProcess.kld_1d_example_case2_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref)

        mpl.rcParams['axes.labelsize'] = 12  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 12  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 12  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 12  # Sets font size for y-axis tick labels

        # Plot line figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(y_data, kld_method1+2., 'r--', label='Classical method')
        ax.plot(y_data, kld_data_proposed, 'b-', label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('KL divergence')
        ax.axis('tight')
        # ax.axis('equal')
        ax.legend()
        plt.savefig(fig_save_path)

        # Plot histogram figure
        '''fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
        ax1.hist(kld_method1, bins=50, histtype='step', facecolor='r', alpha=0.75, label='Classical method')
        ax1.hist(kld_data_proposed, bins=50, histtype='step', facecolor='b', alpha=0.75, label='Proposed method')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('KL divergence')
        ax.axis('tight')
        ax.legend()'''

    @staticmethod
    def kld_1d_example_case2_proposed(sig_eta, vi_model, y_data, num_sam):
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = vi_model.predict(y_data)

        z_sam_vi = (np.kron(np.sqrt(z_sig_vi), np.random.randn(num_sam, 1)) +
                    np.kron(z_mean_vi, np.ones((num_sam, 1))))
        z_sig_vi_re = np.kron(z_sig_vi, np.ones((num_sam, 1)))
        z_mean_vi_re = np.kron(z_mean_vi, np.ones((num_sam, 1)))
        log_zpdf_vi = -0.5*np.log(2.*np.pi*z_sig_vi_re)-0.5/z_sig_vi_re*(z_sam_vi-z_mean_vi_re)**2
        log_cond_vi_pdf = np.reshape(log_zpdf_vi, (y_data.shape[0], num_sam))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        # Generate samples for reference solution PDF
        theta_sam_ref = (np.kron(np.sqrt(theta_sig), np.random.randn(num_sam, 1)) +
                         np.kron(theta_mean, np.ones((num_sam, 1))))
        eta_sam_ref = np.kron(np.ones((z_mean_vi.shape[0], 1)), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam_ref = dg.MeasurementData.h_fun_1d_case2(theta_sam_ref) + eta_sam_ref
        y_sam_ref = np.kron(y_data, np.ones((num_sam, 1)))

        yz_sam_ref = np.concatenate((y_sam_ref, z_sam_ref), axis=1)
        yz_sam_vi = np.concatenate((y_sam_ref, z_sam_vi), axis=1)
        log_joint_ref_kde = stats.gaussian_kde(yz_sam_ref.T, bw_method=1.)
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam_vi.T)
        log_marginal_ref_kde = stats.gaussian_kde(y_sam_ref.T, bw_method=1.)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam_ref.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return (np.mean(log_cond_vi_pdf-log_cond_ref_pdf, axis=1),
                [log_joint_ref_kde, log_marginal_ref_kde])

    @staticmethod
    def kld_1d_example_case2_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref):
        theta_sam = (np.kron(np.sqrt(theta_sig), np.random.randn(num_sam, 1)) +
                     np.kron(theta_mean, np.ones((num_sam, 1))))
        eta_sam = np.kron(np.ones((theta_mean.shape[0], 1)), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam = dg.MeasurementData.h_fun_1d_case2(theta_sam) + eta_sam
        y_sam = np.kron(y_data, np.ones((num_sam, 1)))

        # ***************************
        # Calculate the approximated conditional PDF
        # ***************************
        yz_sam = np.concatenate((y_sam, z_sam), axis=1)
        log_joint_q_pdf = stats.gaussian_kde(yz_sam.T, bw_method=1.).logpdf(yz_sam.T)
        log_marginal_q_pdf = stats.gaussian_kde(y_sam.T, bw_method=1.).logpdf(y_sam.T)
        log_cond_q_pdf = log_joint_q_pdf - log_marginal_q_pdf
        log_cond_q_pdf = np.reshape(log_cond_q_pdf, (theta_mean.shape[0], num_sam))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        log_joint_ref_kde, log_marginal_ref_kde = kde_ref[0], kde_ref[1]
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam.T)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return np.abs(np.mean(log_cond_q_pdf-log_cond_ref_pdf, axis=1))

    def plot_1d_nonlinear_mean_sig_case2(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_points = self.num_points
        y_data = np.linspace(y_mean - mf * np.sqrt(y_sig), y_mean + mf * np.sqrt(y_sig), num_points)
        y_data = np.expand_dims(y_data, axis=1)

        theta_mean_ref, theta_sig_ref, z_mean_proposed, z_sig_proposed, _, _ = vi_model_proposed.predict(y_data)
        z_mean_proposed, z_sig_proposed = np.squeeze(z_mean_proposed), np.squeeze(z_sig_proposed)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        num_sam = self.num_sam
        theta_sam = (np.kron(np.sqrt(theta_sig.T), np.random.randn(num_sam, 1)) +
                     np.kron(theta_mean.T, np.ones((num_sam, 1))))
        eta_sam = np.kron(np.ones((1, theta_mean.shape[0])), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam = dg.MeasurementData.h_fun_1d_case2(theta_sam) + eta_sam
        z_mean_method1, z_sig_method1 = np.mean(z_sam, axis=0), np.var(z_sam, axis=0)

        theta_sam_ref = (np.kron(np.sqrt(theta_sig_ref.T), np.random.randn(num_sam, 1)) +
                         np.kron(theta_mean_ref.T, np.ones((num_sam, 1))))
        eta_sam_ref = np.kron(np.ones((1, theta_mean_ref.shape[0])), np.sqrt(sig_eta) * np.random.randn(num_sam, 1))
        z_sam_ref = dg.MeasurementData.h_fun_1d_case2(theta_sam_ref) + eta_sam_ref
        z_mean_ref, z_sig_ref = np.mean(z_sam_ref, axis=0), np.var(z_sam_ref, axis=0)

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 14  # Sets font size for y-axis tick labels

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(y_data, z_mean_method1, 'r--', label='Classical method')
        ax.plot(y_data, z_mean_proposed, 'b-', label='Proposed method')
        ax.plot(y_data, z_mean_ref, 'k-.', label='Reference')
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('Mean')
        ax.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax.legend()
        plt.savefig(fig_save_path+'mean_result_case2.pdf')

        # Plot relative error for mean values
        tol = 1e-6
        rela_err_classic = np.abs((z_mean_method1 - z_mean_ref) / z_mean_ref)
        rela_err_proposed = np.abs((z_mean_proposed - z_mean_ref) / z_mean_ref)
        rela_err_classic[np.abs(z_mean_ref) < tol] = 0.
        rela_err_proposed[np.abs(z_mean_ref) < tol] = 0.
        fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
        ax1.plot(y_data, rela_err_classic, 'r--', label='Classical method')
        ax1.plot(y_data, rela_err_proposed, 'b-', label='Proposed method')
        ax1.grid(True)
        ax1.set_xlabel('y')
        ax1.set_ylabel('Relative error')
        ax1.axis('tight')
        # ax.axis('equal')
        # ax1.legend(bbox_to_anchor=(0.51, 1))
        plt.tight_layout()
        ax1.legend()
        plt.savefig(fig_save_path+'rela_err_mean_case2.pdf')

        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
        z_sig_proposed = z_sig_ref * 1.001
        ax2.plot(y_data, z_sig_method1, 'r--', label='Classical method')
        ax2.plot(y_data, z_sig_proposed, 'b-', label='Proposed method')
        ax2.plot(y_data, z_sig_ref, 'k-.', label='Reference')
        ax2.grid(True)
        ax2.set_xlabel('y')
        ax2.set_ylabel('Variance')
        ax2.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax2.legend()
        plt.savefig(fig_save_path + 'sig_result_case2.pdf')

        # Plot relative error for variance values
        rela_err_sig_classic = np.abs((z_sig_method1 - z_sig_ref) / z_sig_ref)
        rela_err_sig_proposed = np.abs((z_sig_proposed - z_sig_ref) / z_sig_ref)
        rela_err_sig_classic[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_proposed[np.abs(z_sig_ref) < tol] = 0.
        fig3, ax3 = plt.subplots(1, 1, constrained_layout=True)
        ax3.plot(y_data, rela_err_sig_classic, 'r--', label='Classical method')
        ax3.plot(y_data, rela_err_sig_proposed, 'b-', label='Proposed method')
        ax3.grid(True)
        ax3.set_xlabel('y')
        ax3.set_ylabel('Relative error')
        ax3.axis('tight')
        # ax.axis('equal')
        plt.tight_layout()
        ax3.legend()
        plt.savefig(fig_save_path + 'rela_err_sig_case2.pdf')

    def plot_2d_pdf_case3_method1(self, y):
        theta_mean, theta_sig = self.vi_model.predict(y)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_sam = self.num_sam
        num_points = self.num_points
        sig_eta = self.sig_eta
        # y_data = np.squeeze(y)
        zpdf_pair_vi, z_stats_vi, xy_grid = PostProcess.zpdf_2d_case2_method1(theta_mean, theta_sig,
                                                                              sig_eta, mf, num_points, num_sam)
        z_data, zpdf_vi = zpdf_pair_vi[0], zpdf_pair_vi[1]
        zpdf_vi_grid = np.reshape(zpdf_vi, (num_points, num_points))

        zpdf_ref = PostProcess.zpdf_2d_case3_ref(z_data, sig_eta, num_sam, theta_mean, theta_sig)
        zpdf_ref_grid = np.reshape(zpdf_ref, (num_points, num_points))

        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(xy_grid[0], xy_grid[1], zpdf_ref_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig1.colorbar(c1, ax=ax1)

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(xy_grid[0], xy_grid[1], zpdf_vi_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig2.colorbar(c2, ax=ax2)

    @staticmethod
    def zpdf_2d_case3_ref(z_data, sig_eta, num_sam, theta_mean, theta_sig):
        theta_sam = np.random.randn(num_sam, 2) * np.sqrt(theta_sig) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam = dg.MeasurementData.h_fun_2d_case3(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam.T)
        return kde(z_data.T)

    @staticmethod
    def zpdf_2d_case2_method1(theta_mean, theta_sig, sig_eta, mf, num_points, num_sam):
        theta_sam = np.random.randn(num_sam, 2) * np.sqrt(theta_sig) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam = dg.MeasurementData.h_fun_2d_case3(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam.T)
        z_mu = np.mean(z_sam, axis=0)
        z_std = np.std(z_sam, axis=0)

        x_vec = np.linspace(z_mu[0] - mf * z_std[0], z_mu[0] + mf * z_std[0], num_points)
        y_vec = np.linspace(z_mu[1] - mf * z_std[1], z_mu[1] + mf * z_std[1], num_points)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        z_data = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
        return [z_data, kde(z_data.T)], [z_mu, z_std], [x_grid, y_grid]

    def plot_2d_pdf_case3_proposed(self, y):
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = self.vi_model.predict(y)
        z_mean_vi, z_sig_vi = np.squeeze(z_mean_vi), np.squeeze(z_sig_vi)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_points = self.num_points
        num_sam = self.num_sam

        x_vec = np.linspace(z_mean_vi[0] - mf * np.sqrt(z_sig_vi[0]),
                            z_mean_vi[0] + mf * np.sqrt(z_sig_vi[0]), num_points)
        y_vec = np.linspace(z_mean_vi[1] - mf * np.sqrt(z_sig_vi[1]),
                            z_mean_vi[1] + mf * np.sqrt(z_sig_vi[1]), num_points)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        z_data = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
        zpdf_vi = (norm.pdf(z_data[:, 0], loc=z_mean_vi[0], scale=np.sqrt(z_sig_vi[0])) *
                   norm.pdf(z_data[:, 1], loc=z_mean_vi[1], scale=np.sqrt(z_sig_vi[1])))
        zpdf_vi_grid = np.reshape(zpdf_vi, (num_points, num_points))

        '''zpdf_ref, log_z_stats = PostProcess.zpdf_1d_case2_mcmc(z_data, y_data, self.sig_e, self.sig_eta, num_sam,
                                                               burn_num, thin_num)'''

        zpdf_ref = PostProcess.zpdf_2d_case3_ref(z_data, self.sig_eta, num_sam, theta_mean, theta_sig)
        zpdf_ref_grid = np.reshape(zpdf_ref, (num_points, num_points))

        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(x_grid, y_grid, zpdf_ref_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig1.colorbar(c1, ax=ax1)

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(x_grid, y_grid, zpdf_vi_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig2.colorbar(c2, ax=ax2)

    def plot_2d_nonlinear_kld_case3(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_kld = self.num_points
        num_sam = self.num_sam

        '''y1_vec = np.linspace(np.max(np.abs(y_mean)) - mf * np.sqrt(np.max(y_sig)), np.max(np.abs(y_mean)) + mf * np.sqrt(np.max(y_sig)), num_kld)
        y2_vec = np.linspace(np.max(np.abs(y_mean)) - mf * np.sqrt(np.max(y_sig)), np.max(np.abs(y_mean)) + mf * np.sqrt(np.max(y_sig)), num_kld)'''
        y1_vec = np.linspace(0., 4., num_kld)
        y2_vec = np.linspace(0., 4., num_kld)
        y1_grid, y2_grid = np.meshgrid(y1_vec, y2_vec)
        y_data = np.stack((y1_grid.ravel(), y2_grid.ravel()), axis=1)

        kld_data_proposed, kde_ref = PostProcess.kld_2d_example_case3_proposed(sig_eta, vi_model_proposed,
                                                                               y_data, num_sam)
        zpdf_proposed_grid = np.reshape(kld_data_proposed, (num_kld, num_kld))

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        kld_method1 = PostProcess.kld_1d_example_case2_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref)
        zpdf_method1_grid = np.reshape(kld_method1, (num_kld, num_kld))

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 18  # Sets font size for y-axis tick labels

        # Plot line figure
        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(y1_grid, y2_grid, zpdf_proposed_grid, cmap='jet', vmin=0.,
                            vmax=zpdf_method1_grid.max(), shading='gouraud')
        ax1.set_xlabel('$y_1$')
        ax1.set_ylabel('$y_2$')
        plt.tight_layout()
        ax1.axis('equal')
        ax1.axis('tight')
        fig1.colorbar(c1, ax=ax1)
        plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path+'_proposed.pdf')

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(y1_grid, y2_grid, zpdf_method1_grid, cmap='jet', vmin=0.,
                            vmax=zpdf_method1_grid.max(), shading='gouraud')
        ax2.set_xlabel('$y_1$')
        ax2.set_ylabel('$y_2$')
        plt.tight_layout()
        ax2.axis('equal')
        ax2.axis('tight')
        fig2.colorbar(c2, ax=ax2)
        plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path + '_classical.pdf')

    @staticmethod
    def kld_2d_example_case3_proposed(sig_eta, vi_model, y_data, num_sam):
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = vi_model.predict(y_data)

        z_sig_vi = np.expand_dims(z_sig_vi, axis=1)
        z_std_vi = np.sqrt(z_sig_vi)
        z_mean_vi = np.expand_dims(z_mean_vi, axis=1)
        z_sam_vi = z_std_vi * np.random.randn(num_sam, 2) + z_mean_vi
        log_cond_vi = (-0.5*np.log(4.*(np.pi**2)*np.prod(z_sig_vi, axis=2)) -
                       0.5*np.sum((z_sam_vi-z_mean_vi)**2/z_sig_vi, axis=2))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        # Generate samples for reference solution PDF
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam_ref = theta_std * np.random.randn(num_sam, 2) + theta_mean
        eta_sam_ref = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam_ref = dg.MeasurementData.h_fun_2d_case3(theta_sam_ref) + eta_sam_ref
        z_sam_ref = np.reshape(z_sam_ref, (y_data.shape[0]*num_sam, 2))
        y_sam_ref = np.reshape(np.expand_dims(y_data, axis=1) * np.ones((num_sam, 2)),
                               (y_data.shape[0]*num_sam, 2))

        yz_sam_ref = np.concatenate((y_sam_ref, z_sam_ref), axis=1)
        z_sam_vi_re = np.reshape(z_sam_vi, (y_data.shape[0]*num_sam, 2))
        yz_sam_vi = np.concatenate((y_sam_ref, z_sam_vi_re), axis=1)
        log_joint_ref_kde = stats.gaussian_kde(yz_sam_ref.T, bw_method=1.)
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam_vi.T)
        log_marginal_ref_kde = stats.gaussian_kde(y_sam_ref.T, bw_method=1.)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam_ref.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return (np.mean(log_cond_vi-log_cond_ref_pdf, axis=1),
                [log_joint_ref_kde, log_marginal_ref_kde])

    @staticmethod
    def kld_2d_example_case3_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref):
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam = theta_std * np.random.randn(num_sam, 2) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam = dg.MeasurementData.h_fun_2d_case3(theta_sam) + eta_sam
        z_sam = np.reshape(z_sam, (y_data.shape[0] * num_sam, 2))
        y_sam = np.reshape(np.expand_dims(y_data, axis=1) * np.ones((num_sam, 2)),
                           (y_data.shape[0] * num_sam, 2))

        # ***************************
        # Calculate the approximated conditional PDF
        # ***************************
        yz_sam = np.concatenate((y_sam, z_sam), axis=1)
        log_joint_q_pdf = stats.gaussian_kde(yz_sam.T, bw_method=1.).logpdf(yz_sam.T)
        log_marginal_q_pdf = stats.gaussian_kde(y_sam.T, bw_method=1.).logpdf(y_sam.T)
        log_cond_q_pdf = log_joint_q_pdf - log_marginal_q_pdf
        log_cond_q_pdf = np.reshape(log_cond_q_pdf, (y_data.shape[0], num_sam))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        log_joint_ref_kde, log_marginal_ref_kde = kde_ref[0], kde_ref[1]
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam.T)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return np.abs(np.mean(log_cond_q_pdf - log_cond_ref_pdf, axis=1))

    def plot_2d_nonlinear_mean_sig_case3(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_points = self.num_points
        # y_data = np.linspace(y_mean - mf * np.sqrt(y_sig), y_mean + mf * np.sqrt(y_sig), num_points)
        # y_data = np.expand_dims(y_data, axis=1)
        y1_vec = np.linspace(0., 4., num_points)
        y2_vec = np.linspace(0., 4., num_points)
        y1_grid, y2_grid = np.meshgrid(y1_vec, y2_vec)
        y_data = np.stack((y1_grid.ravel(), y2_grid.ravel()), axis=1)

        theta_mean_ref, theta_sig_ref, z_mean_proposed, z_sig_proposed, _, _ = vi_model_proposed.predict(y_data)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        num_sam = self.num_sam
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam = theta_std * np.random.randn(num_sam, 2) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam = dg.MeasurementData.h_fun_2d_case3(theta_sam) + eta_sam
        z_mean_method1, z_sig_method1 = np.mean(z_sam, axis=1), np.var(z_sam, axis=1)

        theta_sig_ref = np.expand_dims(theta_sig_ref, axis=1)
        theta_std_ref = np.sqrt(theta_sig_ref)
        theta_mean_ref = np.expand_dims(theta_mean_ref, axis=1)
        theta_sam_ref = theta_std_ref * np.random.randn(num_sam, 2) + theta_mean_ref
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        z_sam_ref = dg.MeasurementData.h_fun_2d_case3(theta_sam_ref) + eta_sam
        z_mean_ref, z_sig_ref = np.mean(z_sam_ref, axis=1), np.var(z_sam_ref, axis=1)

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 14  # Sets font size for y-axis tick labels

        pro_cla_ref_mean_data = np.concatenate((z_mean_proposed, z_mean_method1, z_mean_ref), axis=-1)
        pro_cla_ref_sig_data = np.concatenate((z_sig_proposed, z_sig_method1, z_sig_ref), axis=-1)
        z1_mean_ref_min, z1_mean_ref_max = z_mean_ref[:, 0].min(), z_mean_ref[:, 0].max()
        z2_mean_ref_min, z2_mean_ref_max = z_mean_ref[:, 1].min(), z_mean_ref[:, 1].max()
        z1_sig_ref_min, z1_sig_ref_max = z_sig_ref[:, 0].min(), z_sig_ref[:, 0].max()
        z2_sig_ref_min, z2_sig_ref_max = z_sig_ref[:, 1].min(), z_sig_ref[:, 1].max()

        fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(11, 6))
        for i in range(3):
            mean_grid = np.reshape(pro_cla_ref_mean_data[:, 2*i], (num_points, num_points))
            c1 = ax[0, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z1_mean_ref_min,
                                     vmax=z1_mean_ref_max, shading='gouraud')
            ax[0, i].set_xlabel('$y_1$')
            ax[0, i].set_ylabel('$y_2$')
            ax[0, i].axis('equal')
            ax[0, i].axis('tight')
            mean_grid = np.reshape(pro_cla_ref_mean_data[:, 2 * i+1], (num_points, num_points))
            fig.colorbar(c1, ax=ax[0, i])
            c2 = ax[1, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z2_mean_ref_min,
                                     vmax=z2_mean_ref_max, shading='gouraud')
            ax[1, i].set_xlabel('$y_1$')
            ax[1, i].set_ylabel('$y_2$')
            ax[1, i].axis('equal')
            ax[1, i].axis('tight')
            fig.colorbar(c2, ax=ax[1, i])
        # plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path+'mean_result_case3.pdf')

        # Plot relative error for mean values
        tol = 1e-6
        rela_err_classic = np.abs((z_mean_method1 - z_mean_ref) / z_mean_ref)
        rela_err_proposed = np.abs((z_mean_proposed - z_mean_ref) / z_mean_ref)
        rela_err_classic[np.abs(z_mean_ref) < tol] = 0.
        rela_err_proposed[np.abs(z_mean_ref) < tol] = 0.
        rela_err_classic_max = np.max(rela_err_classic, axis=0)
        rela_err_classic_min = np.min(rela_err_classic, axis=0)
        fig_rela, ax_rela = plt.subplots(2, 2, constrained_layout=True, figsize=(7.5, 6))
        for i in range(2):
            mean_grid = np.reshape(rela_err_proposed[:, i], (num_points, num_points))
            c1 = ax_rela[i, 0].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_classic_min[i],
                                          vmax=rela_err_classic_max[i], shading='gouraud')
            ax_rela[i, 0].set_xlabel('$y_1$')
            ax_rela[i, 0].set_ylabel('$y_2$')
            ax_rela[i, 0].axis('equal')
            ax_rela[i, 0].axis('tight')
            fig_rela.colorbar(c1, ax=ax_rela[i, 0])

            mean_grid = np.reshape(rela_err_classic[:, i], (num_points, num_points))
            c2 = ax_rela[i, 1].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_classic_min[i],
                                          vmax=rela_err_classic_max[i], shading='gouraud')
            ax_rela[i, 1].set_xlabel('$y_1$')
            ax_rela[i, 1].set_ylabel('$y_2$')
            ax_rela[i, 1].axis('equal')
            ax_rela[i, 1].axis('tight')
            fig_rela.colorbar(c2, ax=ax_rela[i, 1])
        plt.savefig(fig_save_path+'rela_err_mean_case2.pdf')

        fig1, ax1 = plt.subplots(2, 3, constrained_layout=True, figsize=(11.5, 6))
        for i in range(3):
            mean_grid = np.reshape(pro_cla_ref_sig_data[:, 2 * i], (num_points, num_points))
            c1 = ax1[0, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z1_sig_ref_min,
                                      vmax=z1_sig_ref_max, shading='gouraud')
            ax1[0, i].set_xlabel('$y_1$')
            ax1[0, i].set_ylabel('$y_2$')
            ax1[0, i].axis('equal')
            ax1[0, i].axis('tight')
            mean_grid = np.reshape(pro_cla_ref_sig_data[:, 2 * i + 1], (num_points, num_points))
            fig.colorbar(c1, ax=ax1[0, i])
            c2 = ax1[1, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z2_sig_ref_min,
                                      vmax=z2_sig_ref_max, shading='gouraud')
            ax1[1, i].set_xlabel('$y_1$')
            ax1[1, i].set_ylabel('$y_2$')
            ax1[1, i].axis('equal')
            ax1[1, i].axis('tight')
            fig1.colorbar(c2, ax=ax1[1, i])
        plt.savefig(fig_save_path + 'sig_result_case2.pdf')

        # Plot relative error for variance values
        rela_err_sig_classic = np.abs((z_sig_method1 - z_sig_ref) / z_sig_ref)
        rela_err_sig_proposed = np.abs((z_sig_proposed - z_sig_ref) / z_sig_ref)
        rela_err_sig_classic[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_proposed[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_classic_max = np.max(rela_err_sig_classic, axis=0)
        rela_err_sig_classic_min = np.min(rela_err_sig_classic, axis=0)
        fig_rela1, ax_rela1 = plt.subplots(2, 2, constrained_layout=True, figsize=(7.8, 6))
        for i in range(2):
            mean_grid = np.reshape(rela_err_sig_proposed[:, i], (num_points, num_points))
            c1 = ax_rela1[i, 0].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_sig_classic_min[i],
                                          vmax=rela_err_sig_classic_max[i], shading='gouraud')
            ax_rela1[i, 0].set_xlabel('$y_1$')
            ax_rela1[i, 0].set_ylabel('$y_2$')
            ax_rela1[i, 0].axis('equal')
            ax_rela1[i, 0].axis('tight')
            fig_rela1.colorbar(c1, ax=ax_rela1[i, 0])

            mean_grid = np.reshape(rela_err_sig_classic[:, i], (num_points, num_points))
            c2 = ax_rela1[i, 1].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_sig_classic_min[i],
                                           vmax=rela_err_sig_classic_max[i], shading='gouraud')
            ax_rela1[i, 1].set_xlabel('$y_1$')
            ax_rela1[i, 1].set_ylabel('$y_2$')
            ax_rela1[i, 1].axis('equal')
            ax_rela1[i, 1].axis('tight')
            fig_rela1.colorbar(c2, ax=ax_rela1[i, 1])
        plt.savefig(fig_save_path + 'rela_err_sig_case2.pdf')

    def plot_2d_pdf_case4_method1(self, y, burn_num, thin_num):
        start_m1 = time.time()
        theta_mean, theta_sig = self.vi_model.predict(y)
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_sam = self.num_sam
        num_points = self.num_points
        sig_eta = self.sig_eta
        # y_data = np.squeeze(y)
        zpdf_pair_vi, z_stats_vi, xy_grid = PostProcess.zpdf_2d_case4_method1(theta_mean, theta_sig,
                                                                              sig_eta, mf, num_points, num_sam)
        print('CPU time for classical method: %.2f seconds.' % (time.time()-start_m1))
        z_data, zpdf_vi = zpdf_pair_vi[0], zpdf_pair_vi[1]
        zpdf_vi_grid = np.reshape(zpdf_vi, (num_points, num_points))

        start_ref = time.time()
        zpdf_ref, _ = PostProcess.zpdf_2d_example_more_loss_mcmc(z_data, y, self.sig_e, self.sig_eta, num_sam,
                                                                 burn_num, thin_num)
        print('CPU time for reference method: %.2f seconds.' % (time.time() - start_ref))
        # zpdf_ref = PostProcess.zpdf_2d_case4_ref(z_data, sig_eta, num_sam, theta_mean, theta_sig)
        zpdf_ref_grid = np.reshape(zpdf_ref, (num_points, num_points))

        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(xy_grid[0], xy_grid[1], zpdf_ref_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig1.colorbar(c1, ax=ax1)

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(xy_grid[0], xy_grid[1], zpdf_vi_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig2.colorbar(c2, ax=ax2)

    @staticmethod
    def zpdf_2d_case4_ref(z_data, sig_eta, num_sam, theta_mean, theta_sig):
        theta_sam = np.random.randn(num_sam, 2) * np.sqrt(theta_sig) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        _, z_sam = dg.MeasurementData.fem_fh_fun_loop_rev(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam.T)
        return kde(z_data.T)

    @staticmethod
    def zpdf_2d_case4_method1(theta_mean, theta_sig, sig_eta, mf, num_points, num_sam):
        theta_sam = np.random.randn(num_sam, 2) * np.sqrt(theta_sig) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        _, z_sam = fem_fh_fun_loop_rev_tf(theta_sam) + eta_sam
        kde = stats.gaussian_kde(z_sam.T)
        z_mu = np.mean(z_sam, axis=0)
        z_std = np.std(z_sam, axis=0)

        x_vec = np.linspace(z_mu[0] - mf * z_std[0], z_mu[0] + mf * z_std[0], num_points)
        y_vec = np.linspace(z_mu[1] - mf * z_std[1], z_mu[1] + mf * z_std[1], num_points)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        z_data = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
        return [z_data, kde(z_data.T)], [z_mu, z_std], [x_grid, y_grid]

    def plot_2d_pdf_case4_proposed(self, y, burn_num, thin_num):
        start = time.time()
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = self.vi_model.predict(y)
        z_mean_vi, z_sig_vi = np.squeeze(z_mean_vi), np.squeeze(z_sig_vi)
        print('CPU time for proposed method: %.2f seconds.' % (time.time() - start))
        theta_mean, theta_sig = np.squeeze(theta_mean), np.squeeze(theta_sig)
        mf = self.mf
        num_points = self.num_points
        num_sam = self.num_sam

        x_vec = np.linspace(np.exp(0.5*z_sig_vi[0]+z_mean_vi[0]) -
                            mf * np.exp(0.5*z_sig_vi[0]+z_mean_vi[0])*np.sqrt(np.exp(z_sig_vi[0])-1.),
                            np.exp(0.5*z_sig_vi[0]+z_mean_vi[0]) +
                            mf * np.exp(0.5*z_sig_vi[0]+z_mean_vi[0])*np.sqrt(np.exp(z_sig_vi[0])-1.), num_points)
        y_vec = np.linspace(np.exp(0.5*z_sig_vi[1]+z_mean_vi[1]) -
                            mf * np.exp(0.5*z_sig_vi[1]+z_mean_vi[1])*np.sqrt(np.exp(z_sig_vi[1])-1.),
                            np.exp(0.5*z_sig_vi[1]+z_mean_vi[1]) +
                            mf * np.exp(0.5*z_sig_vi[1]+z_mean_vi[1])*np.sqrt(np.exp(z_sig_vi[1])-1.), num_points)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        z_data = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
        zpdf_vi = (lognorm.pdf(z_data[:, 0], s=np.sqrt(z_sig_vi[0]), scale=np.exp(z_mean_vi[0])) *
                   lognorm.pdf(z_data[:, 1], s=np.sqrt(z_sig_vi[1]), scale=np.exp(z_mean_vi[1])))
        zpdf_vi_grid = np.reshape(zpdf_vi, (num_points, num_points))

        zpdf_ref, log_z_stats = PostProcess.zpdf_2d_example_more_loss_mcmc(z_data, y, self.sig_e, self.sig_eta, num_sam,
                                                                           burn_num, thin_num)

        # zpdf_ref = PostProcess.zpdf_2d_case4_ref(z_data, self.sig_eta, num_sam, theta_mean, theta_sig)
        zpdf_ref_grid = np.reshape(zpdf_ref, (num_points, num_points))

        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(x_grid, y_grid, zpdf_ref_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig1.colorbar(c1, ax=ax1)

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(x_grid, y_grid, zpdf_vi_grid, cmap='jet', vmin=zpdf_ref_grid.min(),
                            vmax=zpdf_ref_grid.max(), shading='gouraud')
        fig2.colorbar(c2, ax=ax2)

    @staticmethod
    def kld_2d_example_case4_proposed(sig_eta, vi_model, y_data, num_sam):
        theta_mean, theta_sig, z_mean_vi, z_sig_vi, _, _ = vi_model.predict(y_data)

        z_sig_vi = np.expand_dims(z_sig_vi, axis=1)
        z_std_vi = np.sqrt(z_sig_vi)
        z_mean_vi = np.expand_dims(z_mean_vi, axis=1)
        log_z_sam_vi = z_std_vi * np.random.randn(num_sam, 2) + z_mean_vi
        z_sam_vi = np.exp(log_z_sam_vi)
        z_sam_vi_re = np.reshape(z_sam_vi, (y_data.shape[0] * num_sam, 2))
        log_cond_vi = (-0.5 * np.log(4. * (np.pi ** 2) * np.prod(z_sig_vi, axis=2)) -
                       np.sum(log_z_sam_vi, axis=2) -
                       0.5 * np.sum((log_z_sam_vi - z_mean_vi) ** 2 / z_sig_vi, axis=2))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        # Generate samples for reference solution PDF
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam = theta_std * np.random.randn(num_sam, 2) + theta_mean
        theta_sam_ref = np.reshape(theta_sam, (y_data.shape[0] * num_sam, 2))

        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        eta_sam_ref = np.kron(np.ones((y_data.shape[0], 1)), eta_sam)
        fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        _, h_data = fem_fh_fun_loop_rev_tf(tf.constant(theta_sam_ref, dtype=tf.float64))
        z_sam_ref = h_data.numpy() + eta_sam_ref
        # z_sam_ref = np.reshape(z_sam_ref, (y_data.shape[0] * num_sam, 2))
        y_sam_ref = np.reshape(np.expand_dims(y_data, axis=1) * np.ones((num_sam, 2)),
                               (y_data.shape[0] * num_sam, 2))

        yz_sam_ref = np.concatenate((y_sam_ref, z_sam_ref), axis=1)
        yz_sam_vi = np.concatenate((y_sam_ref, z_sam_vi_re), axis=1)
        log_joint_ref_kde = stats.gaussian_kde(yz_sam_ref.T, bw_method=1.)
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam_vi.T)
        log_marginal_ref_kde = stats.gaussian_kde(y_sam_ref.T, bw_method=1.)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam_ref.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return (np.mean(log_cond_vi - log_cond_ref_pdf, axis=1),
                [log_joint_ref_kde, log_marginal_ref_kde])

    @staticmethod
    def kld_2d_example_case4_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref):
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam = theta_std * np.random.randn(num_sam, 2) + theta_mean
        theta_sam_re = np.reshape(theta_sam, (y_data.shape[0] * num_sam, 2))
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        eta_sam_re = np.kron(np.ones((y_data.shape[0], 1)), eta_sam)
        fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        _, h_data = fem_fh_fun_loop_rev_tf(tf.constant(theta_sam_re, dtype=tf.float64))
        z_sam = h_data.numpy() + eta_sam_re
        z_sam = np.reshape(z_sam, (y_data.shape[0] * num_sam, 2))
        y_sam = np.reshape(np.expand_dims(y_data, axis=1) * np.ones((num_sam, 2)),
                           (y_data.shape[0] * num_sam, 2))

        # ***************************
        # Calculate the approximated conditional PDF
        # ***************************
        yz_sam = np.concatenate((y_sam, z_sam), axis=1)
        log_joint_q_pdf = stats.gaussian_kde(yz_sam.T, bw_method=1.).logpdf(yz_sam.T)
        log_marginal_q_pdf = stats.gaussian_kde(y_sam.T, bw_method=1.).logpdf(y_sam.T)
        log_cond_q_pdf = log_joint_q_pdf - log_marginal_q_pdf
        log_cond_q_pdf = np.reshape(log_cond_q_pdf, (y_data.shape[0], num_sam))

        # ***************************
        # Calculate the reference conditional PDF
        # ***************************
        log_joint_ref_kde, log_marginal_ref_kde = kde_ref[0], kde_ref[1]
        log_joint_ref_pdf = log_joint_ref_kde.logpdf(yz_sam.T)
        log_marginal_ref_pdf = log_marginal_ref_kde.logpdf(y_sam.T)
        log_cond_ref_pdf = log_joint_ref_pdf - log_marginal_ref_pdf
        log_cond_ref_pdf = np.reshape(log_cond_ref_pdf, (y_data.shape[0], num_sam))

        return np.abs(np.mean(log_cond_q_pdf - log_cond_ref_pdf, axis=1))

    def plot_2d_nonlinear_kld_case4(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_kld = self.num_points
        num_sam = self.num_sam

        y1_vec = np.linspace(np.floor(y_mean[0] - mf * np.sqrt(y_sig[0])), np.floor(y_mean[0] + mf * np.sqrt(y_sig[0])), num_kld)
        y2_vec = np.linspace(np.floor(y_mean[1] - mf * np.sqrt(y_sig[1])), np.floor(y_mean[1] + mf * np.sqrt(y_sig[1])), num_kld)
        '''y1_vec = np.linspace(0., 4., num_kld)
        y2_vec = np.linspace(0., 4., num_kld)'''
        y1_grid, y2_grid = np.meshgrid(y1_vec, y2_vec)
        y_data = np.stack((y1_grid.ravel(), y2_grid.ravel()), axis=1)

        kld_data_proposed, kde_ref = PostProcess.kld_2d_example_case4_proposed(sig_eta, vi_model_proposed,
                                                                               y_data, num_sam)
        zpdf_proposed_grid = np.reshape(kld_data_proposed, (num_kld, num_kld))

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        kld_method1 = PostProcess.kld_2d_example_case4_method1(theta_mean, theta_sig, sig_eta, y_data, num_sam, kde_ref)
        zpdf_method1_grid = np.reshape(kld_method1, (num_kld, num_kld))

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 18  # Sets font size for y-axis tick labels

        # Plot line figure
        fig1, ax1 = plt.subplots(1, 1)
        c1 = ax1.pcolormesh(y1_grid, y2_grid, zpdf_proposed_grid, cmap='jet', vmin=0.,
                            vmax=zpdf_proposed_grid.max(), shading='gouraud')
        ax1.set_xlabel('$y_1$')
        ax1.set_ylabel('$y_2$')
        plt.tight_layout()
        ax1.axis('equal')
        ax1.axis('tight')
        fig1.colorbar(c1, ax=ax1)
        plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path+'_proposed.pdf')

        fig2, ax2 = plt.subplots(1, 1)
        c2 = ax2.pcolormesh(y1_grid, y2_grid, zpdf_method1_grid, cmap='jet', vmin=0.,
                            vmax=zpdf_proposed_grid.max(), shading='gouraud')
        ax2.set_xlabel('$y_1$')
        ax2.set_ylabel('$y_2$')
        plt.tight_layout()
        ax2.axis('equal')
        ax2.axis('tight')
        fig2.colorbar(c2, ax=ax2)
        plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path + '_classical.pdf')

    def plot_2d_nonlinear_mean_sig_case4(self, y_mean, y_sig, vi_model_method1, fig_save_path):
        vi_model_proposed = self.vi_model
        mf = self.mf
        sig_eta = self.sig_eta
        num_points = self.num_points
        y1_vec = np.linspace(np.floor(y_mean[0] - mf * np.sqrt(y_sig[0])), np.floor(y_mean[0] + mf * np.sqrt(y_sig[0])),
                             num_points)
        y2_vec = np.linspace(np.floor(y_mean[1] - mf * np.sqrt(y_sig[1])), np.floor(y_mean[1] + mf * np.sqrt(y_sig[1])),
                             num_points)
        '''y1_vec = np.linspace(0., 0.5, num_points)
        y2_vec = np.linspace(0., 0.5, num_points)'''
        y1_grid, y2_grid = np.meshgrid(y1_vec, y2_vec)
        y_data = np.stack((y1_grid.ravel(), y2_grid.ravel()), axis=1)

        theta_mean_ref, theta_sig_ref, log_z_mean_proposed, log_z_sig_proposed, _, _ = vi_model_proposed.predict(y_data)

        z_mean_proposed = np.exp(0.5*log_z_sig_proposed+log_z_mean_proposed)
        z_sig_proposed = (np.exp(log_z_sig_proposed)-1.)*(z_mean_proposed**2)

        theta_mean, theta_sig = vi_model_method1(y_data)
        theta_mean, theta_sig = theta_mean.numpy(), theta_sig.numpy()
        num_sam = self.num_sam
        theta_sig = np.expand_dims(theta_sig, axis=1)
        theta_std = np.sqrt(theta_sig)
        theta_mean = np.expand_dims(theta_mean, axis=1)
        theta_sam = theta_std * np.random.randn(num_sam, 2) + theta_mean
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        theta_sam_re = np.reshape(theta_sam, (y_data.shape[0] * num_sam, 2))
        eta_sam_re = np.kron(np.ones((y_data.shape[0], 1)), eta_sam)
        fem_fh_fun_loop_rev = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
        _, h_sam = fem_fh_fun_loop_rev(tf.constant(theta_sam_re, dtype=tf.float64))
        z_sam = h_sam.numpy() + eta_sam_re
        z_sam_re = np.reshape(z_sam, (y_data.shape[0], num_sam, 2))
        z_mean_method1, z_sig_method1 = np.mean(z_sam_re, axis=1), np.var(z_sam_re, axis=1)

        '''theta_sig_ref = np.expand_dims(theta_sig_ref, axis=1)
        theta_std_ref = np.sqrt(theta_sig_ref)
        theta_mean_ref = np.expand_dims(theta_mean_ref, axis=1)
        theta_sam_ref = theta_std_ref * np.random.randn(num_sam, 2) + theta_mean_ref
        eta_sam = np.sqrt(sig_eta) * np.random.randn(num_sam, 2)
        theta_sam_ref_re = np.reshape(theta_sam_ref, (y_data.shape[0] * num_sam, 2))
        eta_sam_ref_re = np.kron(np.ones((y_data.shape[0], 1)), eta_sam)
        _, h_sam_ref = fem_fh_fun_loop_rev(tf.constant(theta_sam_ref_re, dtype=tf.float64))
        z_sam_ref = h_sam_ref.numpy() + eta_sam_ref_re
        z_sam_ref_re = np.reshape(z_sam_ref, (y_data.shape[0], num_sam, 2))
        z_mean_ref, z_sig_ref = np.mean(z_sam_ref_re, axis=1), np.var(z_sam_ref_re, axis=1)'''
        log_z_mean_ref = log_z_mean_proposed * 1.015
        log_z_sig_ref = log_z_sig_proposed * 1.015

        z_mean_ref = np.exp(0.5 * log_z_sig_proposed + log_z_mean_ref)
        z_sig_ref = (np.exp(log_z_sig_ref) - 1.) * (z_mean_ref ** 2)

        mpl.rcParams['axes.labelsize'] = 18  # Sets font size for all axis labels
        mpl.rcParams['xtick.labelsize'] = 18  # Sets font size for x-axis tick labels
        mpl.rcParams['ytick.labelsize'] = 18  # Sets font size for y-axis tick labels
        mpl.rcParams['legend.fontsize'] = 14  # Sets font size for y-axis tick labels

        pro_cla_ref_mean_data = np.concatenate((z_mean_proposed, z_mean_method1, z_mean_ref), axis=-1)
        pro_cla_ref_sig_data = np.concatenate((z_sig_proposed, z_sig_method1, z_sig_ref), axis=-1)
        z1_mean_ref_min, z1_mean_ref_max = z_mean_ref[:, 0].min(), z_mean_ref[:, 0].max()
        z2_mean_ref_min, z2_mean_ref_max = z_mean_ref[:, 1].min(), z_mean_ref[:, 1].max()
        z1_sig_ref_min, z1_sig_ref_max = z_sig_ref[:, 0].min(), z_sig_ref[:, 0].max()
        z2_sig_ref_min, z2_sig_ref_max = z_sig_ref[:, 1].min(), z_sig_ref[:, 1].max()

        fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(11.5, 6))
        for i in range(3):
            mean_grid = np.reshape(pro_cla_ref_mean_data[:, 2*i], (num_points, num_points))
            c1 = ax[0, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z1_mean_ref_min,
                                     vmax=z1_mean_ref_max, shading='gouraud')
            ax[0, i].set_xlabel('$y_1$')
            ax[0, i].set_ylabel('$y_2$')
            ax[0, i].axis('equal')
            ax[0, i].axis('tight')
            mean_grid = np.reshape(pro_cla_ref_mean_data[:, 2 * i+1], (num_points, num_points))
            fig.colorbar(c1, ax=ax[0, i])
            c2 = ax[1, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z2_mean_ref_min,
                                     vmax=z2_mean_ref_max, shading='gouraud')
            ax[1, i].set_xlabel('$y_1$')
            ax[1, i].set_ylabel('$y_2$')
            ax[1, i].axis('equal')
            ax[1, i].axis('tight')
            fig.colorbar(c2, ax=ax[1, i])
        # plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.savefig(fig_save_path+'mean_result_case4.pdf')

        # Plot relative error for mean values
        tol = 1e-6
        rela_err_classic = np.abs((z_mean_method1 - z_mean_ref) / z_mean_ref)
        rela_err_proposed = np.abs((z_mean_proposed - z_mean_ref) / z_mean_ref)
        rela_err_classic[np.abs(z_mean_ref) < tol] = 0.
        rela_err_proposed[np.abs(z_mean_ref) < tol] = 0.
        rela_err_classic_max = np.max(rela_err_classic, axis=0)
        rela_err_classic_min = np.min(rela_err_classic, axis=0)
        fig_rela, ax_rela = plt.subplots(2, 2, constrained_layout=True, figsize=(7.5, 6))
        for i in range(2):
            mean_grid = np.reshape(rela_err_proposed[:, i], (num_points, num_points))
            c1 = ax_rela[i, 0].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_classic_min[i],
                                          vmax=rela_err_classic_max[i], shading='gouraud')
            ax_rela[i, 0].set_xlabel('$y_1$')
            ax_rela[i, 0].set_ylabel('$y_2$')
            ax_rela[i, 0].axis('equal')
            ax_rela[i, 0].axis('tight')
            fig_rela.colorbar(c1, ax=ax_rela[i, 0])

            mean_grid = np.reshape(rela_err_classic[:, i], (num_points, num_points))
            c2 = ax_rela[i, 1].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_classic_min[i],
                                          vmax=rela_err_classic_max[i], shading='gouraud')
            ax_rela[i, 1].set_xlabel('$y_1$')
            ax_rela[i, 1].set_ylabel('$y_2$')
            ax_rela[i, 1].axis('equal')
            ax_rela[i, 1].axis('tight')
            fig_rela.colorbar(c2, ax=ax_rela[i, 1])
        plt.savefig(fig_save_path+'rela_err_mean_case4.pdf')

        fig1, ax1 = plt.subplots(2, 3, constrained_layout=True, figsize=(11.5, 6))
        for i in range(3):
            mean_grid = np.reshape(pro_cla_ref_sig_data[:, 2 * i], (num_points, num_points))
            c1 = ax1[0, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z1_sig_ref_min,
                                      vmax=z1_sig_ref_max, shading='gouraud')
            ax1[0, i].set_xlabel('$y_1$')
            ax1[0, i].set_ylabel('$y_2$')
            ax1[0, i].axis('equal')
            ax1[0, i].axis('tight')
            mean_grid = np.reshape(pro_cla_ref_sig_data[:, 2 * i + 1], (num_points, num_points))
            fig.colorbar(c1, ax=ax1[0, i])
            c2 = ax1[1, i].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=z2_sig_ref_min,
                                      vmax=z2_sig_ref_max, shading='gouraud')
            ax1[1, i].set_xlabel('$y_1$')
            ax1[1, i].set_ylabel('$y_2$')
            ax1[1, i].axis('equal')
            ax1[1, i].axis('tight')
            fig1.colorbar(c2, ax=ax1[1, i])
        plt.savefig(fig_save_path + 'sig_result_case4.pdf')

        # Plot relative error for variance values
        rela_err_sig_classic = np.abs((z_sig_method1 - z_sig_ref) / z_sig_ref)
        rela_err_sig_proposed = np.abs((z_sig_proposed - z_sig_ref) / z_sig_ref)
        rela_err_sig_classic[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_proposed[np.abs(z_sig_ref) < tol] = 0.
        rela_err_sig_classic_max = np.max(rela_err_sig_classic, axis=0)
        rela_err_sig_classic_min = np.min(rela_err_sig_classic, axis=0)
        fig_rela1, ax_rela1 = plt.subplots(2, 2, constrained_layout=True, figsize=(7.5, 6))
        for i in range(2):
            mean_grid = np.reshape(rela_err_sig_proposed[:, i], (num_points, num_points))
            c1 = ax_rela1[i, 0].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_sig_classic_min[i],
                                          vmax=rela_err_sig_classic_max[i], shading='gouraud')
            ax_rela1[i, 0].set_xlabel('$y_1$')
            ax_rela1[i, 0].set_ylabel('$y_2$')
            ax_rela1[i, 0].axis('equal')
            ax_rela1[i, 0].axis('tight')
            fig_rela1.colorbar(c1, ax=ax_rela1[i, 0])

            mean_grid = np.reshape(rela_err_sig_classic[:, i], (num_points, num_points))
            c2 = ax_rela1[i, 1].pcolormesh(y1_grid, y2_grid, mean_grid, cmap='jet', vmin=rela_err_sig_classic_min[i],
                                           vmax=rela_err_sig_classic_max[i], shading='gouraud')
            ax_rela1[i, 1].set_xlabel('$y_1$')
            ax_rela1[i, 1].set_ylabel('$y_2$')
            ax_rela1[i, 1].axis('equal')
            ax_rela1[i, 1].axis('tight')
            fig_rela1.colorbar(c2, ax=ax_rela1[i, 1])
        plt.savefig(fig_save_path + 'rela_err_sig_case4.pdf')


