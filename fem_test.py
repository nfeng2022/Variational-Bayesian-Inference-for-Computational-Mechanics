# Imports
import fem_preprocess as fp
import fem_solver as fs
import fem_postprocess as fpp
import scipy.io as sio
import numpy as np


# Initialize and Save Model Data
inputfilename = 'Armero_cooksm_20x10.txt'
model_file_name = 'model_file.mat'
data_ini_flag = 1
if data_ini_flag == 1:
    fp.PreProcessing.modeldata_initialization_topopt(inputfilename, model_file_name)


# Load Data
if data_ini_flag == 0:
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


# FEM Analysis
fs.FemSolver.fea_solution(input_data=None)

# Save Results
data_dic = {'model_data': fp.PreProcessing.model_data,
            'out_data': fp.PreProcessing.out_data,
            'sol_data': fp.PreProcessing.sol_data,
            'topo_data': fp.PreProcessing.topo_data,
            'topo_data_ele': fp.PreProcessing.topo_data_ele,
            'out_data_cells': fp.PreProcessing.out_data_cells}
sio.savemat('results.mat', data_dic)
fpp.PostProcessing.xdmf_h5data_save('xdmf_topology.h5')
# fpp.PostProcessing.create_xdfm_file('results.xfm', 'xdmf_topology.h5')

# Postprocess
mf = 1
flag = 2
step_id = len(fp.PreProcessing.out_data['step'])
fpp.PostProcessing.plot_2d_mesh(step_id, mf, flag)
ele_id = 12
nipt_id = np.array([1, 3], dtype=int)
von_mises_stress = fpp.PostProcessing.von_mises_stress(step_id, ele_id, nipt_id)
c = 0
