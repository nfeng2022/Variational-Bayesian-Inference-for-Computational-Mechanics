# Imports
import tensorflow as tf
import hdf5storage as h5
from src import data_generation_2sam_more_loss as dg, postprocess_lib as pp
import numpy as np
import fem_preprocess as fp
import fem_postprocess as fpp
import fem_solver_tf as fstf

# tf.keras.backend.set_floatx('float64')     # computation precision

# Load observed data
file_path = ''
file_name = 'data_fem_test_big_noise.h5'
data = h5.read(path=file_path, filename=file_name)
y_data = data['y_data']
y_mean_stat = np.squeeze(data['y_mean'])
y_std_stat = np.squeeze(data['y_std'])
z_mean_stat = data['z_mean']
z_std_stat = data['z_std']

# Load FEM model data
num_parallel_cores = 12
dg.MeasurementData.num_parallel_cores = num_parallel_cores
model_file_name = 'model_file.mat'
data = fpp.PostProcessing.loadmat(model_file_name)
# Convert element data to dic
data['model_data']['element'] = list(data['model_data']['element'])
for i in range(len(data['model_data']['element'])):
    data['model_data']['element'][i] = fpp.PostProcessing.todict(data['model_data']['element'][i])
data['out_data']['step'] = [data['out_data']['step']]
data['model_data']['part']['body'] = data['model_data']['part']['body'].reshape(-1, 1)
data['model_data']['part'] = [data['model_data']['part']]
data['model_data']['material'] = [data['model_data']['material']]
data['model_data']['section'] = [data['model_data']['section']]

fp.PreProcessing.model_data = data['model_data']
fp.PreProcessing.sol_data = data['sol_data']
fp.PreProcessing.out_data = data['out_data']
fp.PreProcessing.constants = data['constants']
fp.PreProcessing.topo_data = data['topo_data']
fp.PreProcessing.topo_data_ele = data['topo_data_ele']
fp.PreProcessing.out_data_cells = data['out_data_cells']
fp.PreProcessing.sg2 = data['sg2']
fstf.FemSolver.convert_sparse_to_dense()

# Load predictive model for method 1
results_path_method1 = 'results_method1'
model_path_method1 = results_path_method1+'/final_model_test.keras'
vi_pred_model_method1 = tf.keras.models.load_model(model_path_method1)

# Load predictive model for proposed method
results_path_proposed = 'results_2steps_proposed2'
model_path_proposed = results_path_proposed+'/final_model_test.h5'
vi_pred_model_proposed = tf.keras.models.load_model(model_path_proposed)

# Plot predictions from different models
sig_e = 1e-1    # measurement noise
sig_eta = 3e-3   # prediction noise


# y = np.expand_dims(y_data[50, :], axis=0)
y = np.array([[0.1, 0.1]])
# For predicting results from method 1
pp.PostProcess(vi_pred_model_method1, sig_e, sig_eta, mf=2, num_points=200,
               num_sam=int(1e3)).plot_2d_pdf_case4_method1(y, burn_num=0, thin_num=1)
# For predicting results from proposed method
'''pp.PostProcess(vi_pred_model_proposed, sig_e, sig_eta, mf=4, num_points=200,
               num_sam=int(1e3)).plot_2d_pdf_case4_proposed(y, burn_num=500, thin_num=1)'''

'''pp.PostProcess(vi_pred_model_proposed, sig_e, sig_eta, mf=4., num_points=20,
               num_sam=10).plot_2d_nonlinear_kld_case4(y_mean_stat, y_std_stat**2,
                                                       vi_pred_model_method1,  'kld_result')'''

'''pp.PostProcess(vi_pred_model_proposed, sig_e, sig_eta, mf=4, num_points=20,
               num_sam=50).plot_2d_nonlinear_mean_sig_case4(y_mean_stat, y_std_stat**2,
                                                                  vi_pred_model_method1,  '')'''
c = 0
