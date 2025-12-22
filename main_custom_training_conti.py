# Imports
import math
import os
import numpy as np
import hdf5storage as h5
import data_generation_2sam_more_loss as dg, postprocess_lib as pp
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import fem_preprocess as fp
import fem_postprocess as fpp
import fem_solver_tf as fstf
import time
import scipy.io as sio

##########################################################
# Generate data and save data file
##########################################################
file_path = ''
file_name = 'data_fem_test_big_noise.h5'
y_dim = 2  # dimension of observed variable
theta_dim = 2  # dimension of model parameters
z_dim = 2  # dimension of predictive variables
sig_e = 1e-1  # measurement noise
sig_eta = 3e-3  # prediction noise
num_data = int(1000)  # number of observed data points
ne_sam = int(100)  # number of resampling points
data_generation_flag = 0
inputfilename = 'Armero_cooksm_20x10.txt'  # FEM mesh file
model_file_name = 'model_file.mat'  # FEM model data file name

theta_mean, theta_std = np.array([np.log(20.0), 0.0]), np.array([0.1, 0.015])
node_id, ele_id, nipt_id = 231, 12, np.array([1, 3], dtype=int)
num_parallel_cores = 12
dg.MeasurementData.num_parallel_cores = num_parallel_cores
dg.MeasurementData.theta_mean, dg.MeasurementData.theta_std = theta_mean, theta_std
dg.MeasurementData.node_id, dg.MeasurementData.ele_id = node_id, ele_id
dg.MeasurementData.nipt_id = nipt_id
# tf.config.run_functions_eagerly(True)
if data_generation_flag == 1:
    if os.path.exists(file_path + file_name):
        os.remove(file_path + file_name)
    y_data_generator = dg.MeasurementData(n_sam=num_data, ne_sam=ne_sam, d_y=y_dim, d_z=z_dim, d_theta=theta_dim,
                                          sig_e=sig_e,
                                          sig_eta=sig_eta)
    fp.PreProcessing.modeldata_initialization_topopt(inputfilename, model_file_name)
    fstf.FemSolver.convert_sparse_to_dense()
    y_data_generator.generate_data_fem()
    y_data_generator.save_data(file_path=file_path, file_name=file_name)

##########################################################
# Load FEM Model Data
##########################################################
if data_generation_flag == 0:
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

# Load data file
data = h5.read(path=file_path, filename=file_name)
y_data = data['y_data']
y_mean_stat = data['y_mean']
y_std_stat = data['y_std']
z_mean_stat = data['z_mean']
z_std_stat = data['z_std']
e_data = tf.constant(data['e_data'], dtype=tf.float64)

##########################################################
# Hyperparameter settings for model architecture and optimizer
##########################################################
num_neuron = 20  # number of neurons in each hidden layer
num_layers1 = 3  # number of hidden layers
num_layers2 = 3  # number of hidden layers

alpha = 1e-7
lr = 1e-3  # learning rate for optimizer
flg_lr_decay = 1  # flag for applying learning rate decay
lr_patience = 5  # epochs before decaying learning rate
decay_rate = 0.9  # the rate for reducing learning rate

batch_size = 32  # batch size of each mini-batch
num_epoch1 = 20  # number of epoch
num_epoch2 = 20  # number of epoch
model_save_freq1 = num_epoch1 // 5  # model saving frequency
model_save_freq2 = num_epoch2 // 5  # model saving frequency
if model_save_freq1 == 0:
    model_save_freq1 = 1
if model_save_freq2 == 0:
    model_save_freq2 = 1
flg_print = 1  # print training logs
print_freq = 5  # print every several steps in single epoch

tf.keras.backend.set_floatx('float64')  # computation precision
# tf.compat.v1.disable_eager_execution()     # disable eagle execution
# tf.config.run_functions_eagerly(True)       # disable tf.function

results_path = f'2d_results_2steps_{num_layers2}layers_{num_epoch2}epoch_big_noise_'
if not os.path.exists(results_path):
    os.mkdir(results_path)

##########################################################
# Prepare training datasets
##########################################################
train_dataset_step1 = tf.data.Dataset.from_tensor_slices(y_data)
train_dataset_step1 = train_dataset_step1.shuffle(buffer_size=num_data).batch(batch_size)

##########################################################
# Define training model p(theta|y)
##########################################################
# Define models for mean and variance of posterior PDF of model parameters
model_path_method1 = results_path + '/step1/15-4.03750058.h5'
vi_pred_model_step1 = tf.keras.models.load_model(model_path_method1)
y = vi_pred_model_step1.inputs[0]
logz_mean_inp = Input(shape=(z_dim,), name='model2_input1')
logz_sig_inp = Input(shape=(z_dim,), name='model2_input2')

# Define models for mean and variance of predictive PDF
# initial = tf.keras.initializers.Constant(3.5)
z_mean_hidden_list = [y]
for i in range(num_layers2):
    z_mean_hidden_temp = Dense(num_neuron, activation='relu', name='z_mean_dense'+str(i))(z_mean_hidden_list[-1])
    # z_mean_hidden_bn_temp = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(z_mean_hidden_temp,
    #                                                                                       training=True)
    z_mean_hidden_list.append(z_mean_hidden_temp)
z_mean = Dense(z_dim, name='z_mean')(z_mean_hidden_list[-1])
# For variance
# initial = tf.keras.initializers.Constant(np.log(0.02))
z_sig_hidden_list = [y]
for i in range(num_layers2):
    z_sig_hidden_temp = Dense(num_neuron, activation='relu', name='z_sig_dense'+str(i))(z_sig_hidden_list[-1])
    # z_sig_hidden_bn_temp = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(z_sig_hidden_temp,
    #                                                                                   training=True)
    z_sig_hidden_list.append(z_sig_hidden_temp)
log_z_sig = Dense(z_dim, name='log_z_sig')(z_sig_hidden_list[-1])
z_sig = tf.math.exp(log_z_sig)


# Combine the whole model for step 1
# vi_pred_model_step1 = Model(y, [theta_mean, theta_sig, log_theta_sig])


class TFMapFnLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TFMapFnLayer, self).__init__()

    def call(self, inputs, *args):
        return dg.MeasurementData.fem_fh_fun_loop_rev(inputs)


# Define callback functions
if not os.path.exists(results_path+'/step1'):
    os.mkdir(results_path+'/step1')


# Train and save model
# @tf.function
def train_step(model, optimizer, loss_fn,  x):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(x, y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


##########################################################
# Train model for approximating p(z|y)
##########################################################

# Freeze layers for model in step 1
vi_pred_model_step1.trainable = False

# Combine the whole model for step 2
vi_pred_model_step2 = Model([y, logz_mean_inp, logz_sig_inp],
                            [vi_pred_model_step1.outputs[0], vi_pred_model_step1.outputs[1],
                             z_mean, z_sig, vi_pred_model_step1.outputs[2], log_z_sig])

# Compute sample estimates of z from q(theta|y)
vi_pred_model_step1_simp = Model(vi_pred_model_step1.outputs[0],
                                 [vi_pred_model_step1.outputs[0], vi_pred_model_step1.outputs[1]])
theta_mean_temp, theta_sig_temp = vi_pred_model_step1_simp.predict(y_data)
theta_std_temp = np.sqrt(theta_sig_temp)
theta_std_sam = np.expand_dims(theta_std_temp, axis=1)
theta_mean_sam = np.expand_dims(theta_mean_temp, axis=1)
# Compute theta samples
theta_sam = e_data.numpy() * theta_std_sam + theta_mean_sam
eta_err = np.sqrt(sig_eta)*np.random.randn(ne_sam, z_dim)
# eta_err = np.kron(np.ones((num_data, 1)), eta_err_temp)
theta_sam = np.reshape(theta_sam, (-1, theta_dim))
theta_sam_batch = tf.data.Dataset.from_tensor_slices(tf.constant(theta_sam, dtype=tf.float64))
theta_sam_batch = theta_sam_batch.batch(batch_size*ne_sam)
h_sam_list = []
fem_fh_fun_loop_rev_tf = tf.function(dg.MeasurementData.fem_fh_fun_loop_rev)
for step, theta_sam_one_batch in enumerate(theta_sam_batch):
    _, h_sam_temp = fem_fh_fun_loop_rev_tf(theta_sam_one_batch)
    h_sam_list.append(tf.cast(h_sam_temp, dtype=tf.float64))
    if flg_print == 1:
        print("Seen so far: %d samples" % ((step + 1) * batch_size))
h_sam = tf.concat(h_sam_list, axis=0)
h_sam_res = np.reshape(h_sam.numpy(), (num_data, ne_sam, z_dim))
z_sam = h_sam_res + eta_err
# z_sam = np.exp(theta_sam)+2.+np.sqrt(sig_eta)*np.random.randn(ne_sam, 1)
logz_sam = np.log(z_sam)
logz_mean_post = np.mean(logz_sam, axis=1)
logz_sig_post = np.var(logz_sam, axis=1)
temp_data_dic = {'logz_mean_post': logz_mean_post, 'logz_sig_post': logz_sig_post}
sio.savemat(results_path+'/temp_data.mat', temp_data_dic)


##########################################################
# Prepare training datasets
##########################################################
'''data_step2 = sio.loadmat(results_path+'/temp_data.mat')
logz_mean_post = data_step2['logz_mean_post']
logz_sig_post = data_step2['logz_sig_post']'''
train_dataset_step2 = tf.data.Dataset.from_tensor_slices((y_data, logz_mean_post, logz_sig_post))
train_dataset_step2 = train_dataset_step2.shuffle(buffer_size=num_data).batch(batch_size)


# Compute the term in Eq. 23
def term4(mu_z, log_z_sig_point):
    loss = -0.5*tf.math.reduce_sum(log_z_sig_point, axis=-1)-tf.math.reduce_sum(mu_z, axis=-1)
    return tf.math.reduce_mean(loss)-0.5*z_dim*math.log(2.*math.pi)-0.5*z_dim


# term4_loss = term4(log_z_sig)


# Computer the term in Eq. 25
def term5(args):
    theta_mean_point, theta_sig_point, z_mean_point, z_sig_point = args
    e_data_point = e_data
    theta_std0 = tf.math.sqrt(theta_sig_point)
    theta_std1 = tf.expand_dims(theta_std0, axis=1)
    theta_mean_point = tf.expand_dims(theta_mean_point, axis=1)
    z_mean_point = tf.expand_dims(z_mean_point, axis=1)
    z_sig_point = tf.expand_dims(z_sig_point, axis=1)
    # Compute theta samples
    theta_data = e_data_point * theta_std1 + theta_mean_point
    # Compute scaled h mapping data
    theta_data = tf.reshape(theta_data, (-1, theta_data.shape[-1]))
    # _, h_data = dg.MeasurementData.fem_fh_fun_loop_rev(tf.convert_to_tensor(theta_data))
    _, h_data = TFMapFnLayer()(theta_data)
    l1 = -0.5/sig_eta*(tf.math.reduce_sum(tf.math.exp(2.*z_mean_point + 2.*z_sig_point), axis=-1))
    l2 = -0.5/sig_eta*tf.reduce_sum(-2.*(h_data*tf.math.exp(z_mean_point+0.5*z_sig_point))+h_data**2, axis=-1)
    l3 = -0.5*z_dim*math.log(2.*math.pi*sig_eta)
    return tf.math.reduce_mean(l1+l2)+l3


'''start = time.time()
term5_loss = term5([theta_mean, theta_sig, z_mean, z_sig])
end = time.time()
print(f'Analysis takes {end-start} seconds.')'''


def add_loss(args):
    z_mean_point, z_sig_point, logz_mean_post_data, logz_sig_post_data = args
    return tf.reduce_mean((z_mean_point-logz_mean_post_data)**2)+tf.reduce_mean((z_sig_point-logz_sig_post_data)**2)


# term5_loss = term5([theta_mean, theta_sig, z_mean, z_sig])

# Define training loss function
def vi_pred_loss_step2(x_inp, y_pred):
    term4_loss = term4(y_pred[2], y_pred[5])
    term5_loss = term5([y_pred[0], y_pred[1], y_pred[2], y_pred[3]])
    return (term4_loss-term5_loss)*alpha+add_loss([y_pred[2], y_pred[3], x_inp[1], x_inp[2]])


# Define optimizer and compile the model
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, epsilon=1e-10)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Change paths
if not os.path.exists(results_path+'/step2'):
    os.mkdir(results_path+'/step2')

# Train and save model
train_step_fn_step2 = tf.function(train_step)
hist_step2 = {'train_loss': np.zeros((num_epoch2, ))}
for epoch in range(num_epoch2):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset_step2):
        loss_value = train_step_fn_step2(vi_pred_model_step2, optimizer, vi_pred_loss_step2, x_batch_train)
        # Log every several batches.
        if step % print_freq == 0 and flg_print == 1:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Log every several epoches.
    if flg_print == 1:
        print("Training loss (for one batch) at Epoch %d: %.4f" % (epoch, float(loss_value)))
        print("Time taken: %.2fs" % (time.time() - start_time))

    # Save trained model every several epoches.
    if (epoch+1) % model_save_freq2 == 0:
        vi_pred_model_step2.save(filepath=results_path+f'/step2/{epoch:02d}-{loss_value:.8f}.h5')

    # Apply learning rate decay strategy
    if epoch % lr_patience == 0 and epoch > 0 and flg_lr_decay == 1:
        if hist_step2['train_loss'][epoch] - hist_step2['train_loss'][epoch - lr_patience] > 0:
            lr0 = optimizer.learning_rate.numpy()
            optimizer.learning_rate = decay_rate * optimizer.learning_rate
            lr1 = optimizer.learning_rate.numpy()
            print(f'Learning rate decays from {lr0: .8f} to {lr1: .8f}.')

    # Save training history at each epoch
    hist_step2['train_loss'][epoch] = float(loss_value)

vi_pred_model_step2.save(filepath=results_path+'/step2/final_model_step2.h5')
h5.write(data=hist_step2, filename=results_path+'/step2/train_hist_step2.h5')

vi_pred_test = Model(y, vi_pred_model_step2.outputs)
vi_pred_test.save(filepath=results_path+'/final_model_test.h5')


# Postprocess
y_test = np.expand_dims(y_data[1, :], axis=0)
'''pp.PostProcess(vi_pred_test, sig_e, sig_eta, 3).plot_1d_pdf_v1_more_loss_mcmc(y_test, num_points=5000, num_mc_sam=int(1e4),
                                                                         fig_save_path=results_path + '/step2/prediction.eps', loc=2.)'''
'''pp.PostProcess(vi_pred_test, sig_e, sig_eta, 5).plot_1d_pdf_v1(y_test, num_points=5000, num_mc_sam=int(1e5),
                                                               fig_save_path=results_path + '/step2/prediction.eps', loc=2.)'''
'''pp.PostProcess(vi_pred_test, sig_e, sig_eta, 6).plot_2d_pdf_more_loss_mcmc(y_test, num_points=200, num_mc_sam=int(5e3),
                                                               fig_save_path=results_path + '/step2/prediction.eps',
                                                                           loc=0., burn_num=500, thin_num=1)'''
# pp.PostProcess(vi_pred_model_step2, sig_e, sig_eta, 10).plot_1d_linear_theta_pdf(y, num_points=1000)
c = 0