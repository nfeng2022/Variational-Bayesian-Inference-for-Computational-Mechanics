# Imports
import numpy as np
import time
import fem_preprocess as fp
import scipy.io as sio
# from scipy.sparse import csr_matrix, lil_matrix, linalg
import mat_subroutine_tf as ms
import tensorflow as tf


class FemSolver:

    @classmethod
    def fea_solution(cls, input_data):

        if input_data:
            try:
                data_buff = sio.loadmat(input_data, struct_as_record=False, squeeze_me=True)
                fp.PreProcessing.model_data = data_buff['model_data']
            except FileNotFoundError:
                print(f"Error: File '{input_data}' not found.")
                return

        '''solver_type = fp.PreProcessing.model_data['solution_control']['solver']
        if solver_type == 1:
            cls.global_linear_solver()
        else:
            raise ValueError('Illegal solver switch parameter ...')'''
        cls.global_linear_solver()

    @classmethod
    def global_linear_solver(cls):

        model_data = fp.PreProcessing.model_data
        # out_data = fp.PreProcessing.out_data
        # sol_data = fp.PreProcessing.sol_data

        '''isFieldResult = cls.myIsField(model_data['solution_control'], 'assemblefunction')
        if isFieldResult == 1:
            funcd = model_data['solution_control']['assemblefunction']
        else:
            funcd = cls.assemble_system_matrices'''
        funcd = cls.assemble_system_matrices

        model_data['loading']['load_factor'] = 1
        '''out_data['step'].append({'Pf': model_data['loading']['Pf'],
                                 'Us': model_data['loading']['Us']})'''

        tic = time.time()
        exit_flag = cls.newt_raphson_linear(funcd, 2)

        if exit_flag == 1:
            '''step_id = 2
            # out_data['step'][step_id-1]['Uf'] = sol_data['u_n1'][model_data['dof_info']['free_dof']-1]
            indices = np.reshape(model_data['dof_info']['free_dof']-1, (-1, 1))
            out_data['step'][step_id - 1]['Uf'] = tf.gather_nd(sol_data['u_n1'], indices)
            # out_data['step'][step_id-1]['Ps'] = sol_data['F_int'][model_data['dof_info']['supp_dof']-1]
            indices = np.reshape(model_data['dof_info']['supp_dof']-1, (-1, 1))
            out_data['step'][step_id - 1]['Ps'] = tf.gather_nd(sol_data['F_int'], indices)

            # Update nodal data: displacement and reactions
            cls.update_nodal_data(step_id)

            out_data['step'][step_id-1]['tol_vec'] = tol_vec
            out_data['step'][step_id-1]['iter_vec'] = np.array([[1]])
            out_data['step'][step_id-1]['load_ratio'] = np.array([[1.0]])'''

            if model_data['solution_control']['print_flag'] == 1:
                print('Analysis successfully Completed ...')
                print(f'Time: {time.time() - tic} sec')
        else:
            print('Analysis failed: Exiting')
            raise ValueError('Illegal exiting flag ...')

    @classmethod
    def to_sparse_tensor(cls, sp_mat):
        coo = sp_mat.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    @classmethod
    def to_tensor(cls, sp_mat):
        mat = sp_mat.toarray()
        return tf.constant(mat, dtype=tf.float64)

    @classmethod
    def newt_raphson_linear(cls, funcd, step_id):

        model_data = fp.PreProcessing.model_data
        sol_data = fp.PreProcessing.sol_data
        out_data = fp.PreProcessing.out_data

        out_data['ele_strain'] = tf.zeros(out_data['ele_strain'].shape, dtype=tf.float64)
        out_data['ele_stress'] = tf.zeros(out_data['ele_stress'].shape, dtype=tf.float64)

        # tol_cr = model_data['solution_control']['nr_param']['tol_cr']
        free_dof = tf.constant(model_data['dof_info']['free_dof'], dtype=tf.int64)
        supp_dof = tf.constant(model_data['dof_info']['supp_dof'], dtype=tf.int64)
        ndof = model_data['dof_info']['ndof']

        # Begin NR: Predictor Step
        '''sol_data['u_n'] = np.zeros((ndof, 1))
        sol_data['u_n'][free_dof - 1] = out_data['step'][step_id - 2]['Uf']
        sol_data['u_n'][supp_dof - 1] = out_data['step'][step_id - 2]['Us']'''
        sol_data['u_n'] = tf.zeros((ndof, 1), dtype=tf.float64)
        sol_data['u_n'] = tf.tensor_scatter_nd_update(sol_data['u_n'],
                                                      tf.stack([free_dof - 1,
                                                                tf.zeros(free_dof.shape, dtype=tf.int64)], axis=1),
                                                      tf.zeros((model_data['dof_info']['nfree'], ), dtype=tf.float64))
        sol_data['u_n'] = tf.tensor_scatter_nd_update(sol_data['u_n'],
                                                      tf.stack([supp_dof - 1,
                                                                tf.zeros(supp_dof.shape, dtype=tf.int64)], axis=1),
                                                      tf.zeros((model_data['dof_info']['nsupp'], ), dtype=tf.float64))

        # step : n+1 (predictor)
        '''sol_data['u_n1'] = sol_data['u_n'].copy()
        sol_data['u_n1'][supp_dof - 1] = out_data['step'][step_id - 1]['Us'].toarray()
        sol_data['du_n1'] = sol_data['u_n1'] - sol_data['u_n']'''
        sol_data['u_n1'] = tf.identity(sol_data['u_n'])
        sol_data['u_n1'] = tf.tensor_scatter_nd_update(sol_data['u_n1'],
                                                       tf.stack([supp_dof - 1,
                                                                 tf.zeros(supp_dof.shape, dtype=tf.int64)], axis=1),
                                                       tf.zeros((model_data['dof_info']['nsupp'], ), dtype=tf.float64))
        sol_data['du_n1'] = sol_data['u_n1'] - sol_data['u_n']

        # assemble matrices: Kt, Fint, Fext
        isw = 2
        funcd(step_id, isw)
        F_intf = tf.gather_nd(sol_data['F_int'],
                              tf.stack((free_dof-1, tf.zeros(free_dof.shape[0], dtype=tf.int64)), axis=1))
        F_extf = tf.gather_nd(sol_data['F_ext'],
                              tf.stack((free_dof-1, tf.zeros(free_dof.shape[0], dtype=tf.int64)), axis=1))
        R_force_f = F_intf - F_extf
        Kg_ff = tf.transpose(tf.gather_nd(tf.transpose(tf.gather_nd(sol_data['Kg'], tf.reshape(free_dof-1, (-1, 1)))),
                             tf.reshape(free_dof-1, (-1, 1))))

        duf = tf.linalg.solve(Kg_ff, -tf.expand_dims(R_force_f, axis=1))

        # Updates
        '''sol_data['u_n1'][free_dof - 1] = sol_data['u_n1'][free_dof - 1] + duf
        sol_data['du_n1'][free_dof - 1] = sol_data['u_n1'][free_dof - 1] - sol_data['u_n'][free_dof - 1]'''
        u_n1_f = tf.gather_nd(sol_data['u_n1'],
                              indices=tf.stack([free_dof - 1, tf.zeros(free_dof.shape, dtype=tf.int64)], axis=1))
        sol_data['u_n1'] = tf.tensor_scatter_nd_update(sol_data['u_n1'],
                                                       tf.stack([free_dof - 1,
                                                                 tf.zeros(free_dof.shape, dtype=tf.int64)], axis=1),
                                                       u_n1_f+tf.squeeze(duf))
        u_n_f = tf.gather_nd(sol_data['u_n'],
                             indices=tf.stack([free_dof - 1, tf.zeros(free_dof.shape, dtype=tf.int64)], axis=1))
        sol_data['du_n1'] = tf.tensor_scatter_nd_update(sol_data['du_n1'],
                                                        tf.stack([free_dof - 1,
                                                                  tf.zeros(free_dof.shape, dtype=tf.int64)], axis=1),
                                                        u_n1_f - u_n_f)

        # assemble matrices: Fint, Fext
        isw = 2
        funcd(step_id, isw)
        '''F_intf = tf.gather_nd(sol_data['F_int'],
                              tf.stack((free_dof - 1, tf.zeros(free_dof.shape[0], dtype=tf.int64)), axis=1))
        R_force_f = F_intf - F_extf
        # R_force_f = sol_data['F_int'][free_dof-1, :].toarray() - sol_data['F_ext'][free_dof-1, :].toarray()

        # CHECK FOR CONVERGENCE
        tol_r = tf.norm(R_force_f)
        tol_e = abs(tf.tensordot(tf.squeeze(duf), R_force_f, axes=[[0], [0]]))

        if model_data['solution_control']['nr_param']['tol_Rforce'] == 1:
            tol = tol_r
        else:
            tol = tol_e

        if tol < tol_cr:
            if model_data['solution_control']['print_flag'] == 1:
                print(f'Energy Norm: {tol_e}')
                print(f'Residual Norm: {tol_r}')
            exit_flag = 1
        else:
            if model_data['solution_control']['print_flag'] == 1:
                print(f'Energy Norm: {tol_e}')
                print(f'Residual Norm: {tol_r}')
            exit_flag = 0'''

        exit_flag = 1

        return exit_flag

    @classmethod
    def myIsField(cls, inStruct, fieldName):
        # inStruct is assumed to be a dictionary
        isFieldResult = False

        for key in inStruct.keys():
            if key == fieldName:
                isFieldResult = True
                return isFieldResult
            elif isinstance(inStruct[key], dict):
                isFieldResult = cls.myIsField(inStruct[key], fieldName)
                if isFieldResult:
                    return isFieldResult

        return isFieldResult

    @classmethod
    def update_nodal_data(cls, step_id):

        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sol_data = fp.PreProcessing.sol_data
        max_node_dof = model_data['mesh_info']['max_node_dof']
        nnodes = model_data['mesh_info']['nnodes']
        ndof = model_data['dof_info']['ndof']
        out_data['step'][step_id-1]['nodal_disp'] = tf.transpose(tf.reshape(sol_data['u_n1'],
                                                                            (nnodes, max_node_dof)))

        out_data['step'][step_id-1]['nodal_react'] = np.zeros((ndof,))
        out_data['step'][step_id-1]['nodal_react'][model_data['dof_info']['free_dof']-1] = 0
        # out_data['step'][step_id-1]['nodal_react'][model_data['dof_info']['supp_dof']-1] = sol_data['F_int'][
        #     model_data['dof_info']['supp_dof']-1]
        indices = np.stack((model_data['dof_info']['supp_dof']-1,
                            np.zeros(model_data['dof_info']['nsupp'], dtype=np.int64)), axis=1)
        updates = tf.gather_nd(sol_data['F_int'], indices)
        out_data['step'][step_id - 1]['nodal_react'] = tf.zeros((model_data['dof_info']['ndof'], 1),
                                                                dtype=tf.float64)
        out_data['step'][step_id - 1]['nodal_react'] = (
            tf.tensor_scatter_nd_update(out_data['step'][step_id - 1]['nodal_react'], indices, updates))
        out_data['step'][step_id - 1]['nodal_react'] = tf.transpose(
            tf.reshape(out_data['step'][step_id - 1]['nodal_react'], (nnodes, max_node_dof)))

    @classmethod
    def assemble_system_matrices(cls, step_id, isw):
        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sol_data = fp.PreProcessing.sol_data

        nele = model_data['mesh_info']['nele']
        # large_disp_flag = model_data['solution_control']['large_disp_flag']
        # ele_type = model_data['ele_type']
        ndof = model_data['dof_info']['ndof']
        max_node_dof = model_data['mesh_info']['max_node_dof']
        max_ele_node = model_data['mesh_info']['max_ele_node']

        # Applied External Loads (configuration independent)
        free_dof = tf.constant(model_data['dof_info']['free_dof'], dtype=tf.int64)
        sol_data['F_ext'] = tf.zeros((ndof, 1), dtype=tf.float64)
        # sol_data['F_ext'][model_data['dof_info']['free_dof']-1] = out_data['step'][step_id-1]['Pf']
        sol_data['F_ext'] = tf.tensor_scatter_nd_update(sol_data['F_ext'],
                                                        tf.stack((free_dof - 1,
                                                                  tf.zeros(free_dof.shape, dtype=tf.int64)), axis=1),
                                                        tf.constant(model_data['loading']['Pf'].squeeze(), dtype=tf.float64))

        # Initialize matrices
        ele_ndof = max_ele_node * max_node_dof
        Fint_all = tf.zeros((ele_ndof * nele, 1), dtype=tf.float64)
        # Fint_ele = np.zeros(ele_ndof)
        # Kt_ele = np.zeros((ele_ndof, ele_ndof))
        K_all = tf.zeros((ele_ndof ** 2 * nele, 1), dtype=tf.float64)
        # M_all = np.zeros(ele_ndof ** 2 * nele) if model_data['solution_control']['dynamics'] == 1 else None

        # Iteration variables
        p = 0
        q = 0

        # Loop over elements
        # out_data['step'][step_id - 1]['strain_energy'] = tf.zeros((nele, ), dtype=tf.float64)
        nel = model_data['mesh_info']['max_ele_node']
        node_dof = model_data['mesh_info']['max_node_dof']
        pid = 1
        IEN = tf.constant(model_data['dof_info']['IEN'], dtype=tf.int64)
        LM = tf.constant(model_data['dof_info']['LM'], dtype=tf.int64)
        coord = tf.constant(model_data['mesh_info']['coord'], dtype=tf.float64)
        for i in tf.range(nele, dtype=tf.int64):
            '''nel = model_data['element'][i]['nnodes']
            node_dof = model_data['element'][i]['node_dof']
            ele_ndof = node_dof * nel

            # Properties
            pid = model_data['element'][i]['part_id']
            mat_id = model_data['part'][pid-1]['mat_id']
            sec_id = model_data['part'][pid-1]['sec_id']
            E = model_data['material'][mat_id-1]['E']
            v = model_data['material'][mat_id-1]['v']
            h = model_data['section'][sec_id-1]['thk']'''

            # Nodal Displacements (total trial)
            '''xl = model_data['element'][i]['coord'][:, 1:3]
            a = np.abs(xl[0, 0] - xl[1, 0]) / 2
            b = np.abs(xl[0, 1] - xl[3, 1]) / 2'''

            # Call appropriate function for element stiffness matrix
            # ix = model_data['element'][i]['conn']
            ix = IEN[i, :]
            # Nodal Displacements(total trial)
            ul = tf.gather_nd(sol_data['u_n1'],
                              tf.stack((LM[:, i] - 1, tf.zeros(ele_ndof, dtype=tf.int64)), axis=1))
            # ul = sol_data['u_n1'][LM[:, i]-1, 0]
            # Nodal Displacements (incremental t_n1 - tn)
            dul = tf.gather_nd(sol_data['du_n1'],
                               tf.stack((LM[:, i] - 1, tf.zeros(ele_ndof, dtype=np.int64)), axis=1))
            # dul = sol_data['du_n1'][LM[:, i]-1, 0]
            ndm = model_data['mesh_info']['space_dim']
            ul = tf.transpose(tf.reshape(ul, (nel, node_dof)))
            dul = tf.transpose(tf.reshape(dul, (nel, node_dof)))
            # xl = model_data['element'][i]['coord'][:, 1:3].T
            xl = tf.gather_nd(coord, tf.reshape(IEN[i, :]-1, (-1, 1)))
            xl = tf.transpose(xl[:, 1:3])
            Fint_ele, Kt_ele, strain_temp, stress_temp = (
                ms.MaterialSub.solid_2d_lib(ul, dul, xl, ix, ndm, nel, isw, pid, step_id, i+1))

            # Assemble Fint and K matrices
            Fint_all = tf.tensor_scatter_nd_update(Fint_all,
                                                   tf.stack((tf.range(q, q + ele_ndof, dtype=tf.int64),
                                                             tf.zeros(ele_ndof, dtype=tf.int64)), axis=1),
                                                   tf.squeeze(Fint_ele))
            # Fint_all[q:q + ele_ndof] = Fint_ele.squeeze()
            q += ele_ndof
            K_all = tf.tensor_scatter_nd_update(K_all,
                                                tf.stack((tf.range(p, p + ele_ndof ** 2, dtype=tf.int64),
                                                          tf.zeros(ele_ndof ** 2, dtype=tf.int64)), axis=1),
                                                tf.reshape(tf.transpose(Kt_ele), (-1,)))
            # K_all[p:p + ele_ndof ** 2] = Kt_ele.flatten(order='F')

            strain_indices = tf.constant(out_data['strain_indices'], dtype=tf.int64)
            strain_indices = tf.concat([strain_indices,
                                        i * tf.ones((strain_temp.shape[0]*strain_temp.shape[1], 1),
                                                    dtype=tf.int64),
                                        (step_id - 1) * tf.ones((strain_temp.shape[0]*strain_temp.shape[1], 1),
                                                                dtype=tf.int64)], axis=1)
            out_data['ele_strain'] = tf.tensor_scatter_nd_update(out_data['ele_strain'], strain_indices,
                                                                 tf.reshape(strain_temp, (-1,)))
            out_data['ele_stress'] = tf.tensor_scatter_nd_update(out_data['ele_stress'], strain_indices,
                                                                 tf.reshape(stress_temp, (-1,)))

            p += ele_ndof ** 2

        # Assemble matrices Kg, Mg, Kgeos, F_ext and Fint
        sol_data['F_int'] = tf.scatter_nd(tf.stack((tf.reshape(tf.transpose(LM), (-1, ))-1,
                                                    tf.zeros(ele_ndof * nele, dtype=tf.int64)), axis=1),
                                          tf.squeeze(Fint_all), tf.constant([ndof, 1], dtype=tf.int64))
        sol_data['Kg'] = tf.scatter_nd(tf.stack((model_data['dof_info']['loc_i_array']-1,
                                                 model_data['dof_info']['loc_j_array']-1), axis=1),
                                       tf.squeeze(K_all), tf.constant([ndof, ndof], dtype=tf.int64))
        '''if isw == 3:
            sol_data['Kgeos'] = csr_matrix(
                (K_all, (model_data['dof_info']['loc_i_array']-1, model_data['dof_info']['loc_j_array']-1)),
                shape=(ndof, ndof))
        if model_data['solution_control']['dynamics'] == 1:
            sol_data['Mg'] = csr_matrix(
                (M_all, (model_data['dof_info']['loc_i_array']-1, model_data['dof_info']['loc_j_array']-1)),
                shape=(ndof, ndof))'''

        # Update F_ext for frame elements or configuration dependent loading
        # sol_data['F_ext'] = ...

        # Compliant mechanism Design
        '''if large_disp_flag == 1:
            if model_data['solution_control']['finverter_flag'] == 1:
                num = len(model_data['solution_control']['finverter_info'])
                Lv = np.zeros(ndof)
                for k in range(num):
                    nid = model_data['solution_control']['finverter_info'][k][0]
                    dof = model_data['solution_control']['finverter_info'][k][1]
                    dir = model_data['solution_control']['finverter_info'][k][2]
                    dof_id = model_data['dof_info']['ID'][dof-1, nid-1]
                    Lv[dof_id-1] = dir * 1
                Lv = csr_matrix(Lv)
                sol_data['Lv'] = Lv

                # Add nodal springs (Force Inverter)
                if model_data['solution_control']['nodal_spring']['nspring'] > 0:
                    for k in range(model_data['solution_control']['nodal_spring']['nspring']):
                        nid = model_data['solution_control']['nodal_spring']['node_id'][k]
                        dof_dir = model_data['solution_control']['nodal_spring']['dof_dir'][k]
                        dof_id = model_data['dof_info']['ID'][dof_dir-1, nid-1]
                        sol_data['Kg'][dof_id-1, dof_id-1] += model_data['solution_control']['nodal_spring']['ks'][k]
                        sol_data['F_int'][dof_id-1, 0] += (model_data['solution_control']['nodal_spring']['ks'][k] *
                                                           sol_data['u_n1'][dof_id-1, 0])'''

    @classmethod
    def convert_sparse_to_dense(cls):
        fp.PreProcessing.model_data['support'] = fp.PreProcessing.model_data['support'].toarray()
        fp.PreProcessing.model_data['loading']['Pf'] = fp.PreProcessing.model_data['loading']['Pf'].toarray()
        fp.PreProcessing.model_data['loading']['Ps'] = fp.PreProcessing.model_data['loading']['Ps'].toarray()
        fp.PreProcessing.model_data['loading']['Us'] = fp.PreProcessing.model_data['loading']['Us'].toarray()
        fp.PreProcessing.model_data['material'][0]['E'] = np.float64(fp.PreProcessing.model_data['material'][0]['E'])
        fp.PreProcessing.model_data['material'][0]['v'] = np.float64(fp.PreProcessing.model_data['material'][0]['v'])

        '''fp.PreProcessing.out_data['step'][1]['Pf'] = fp.PreProcessing.out_data['step'][1]['Pf'].toarray()
        fp.PreProcessing.out_data['step'][1]['Us'] = fp.PreProcessing.out_data['step'][1]['Us'].toarray()
        fp.PreProcessing.out_data['step'][1]['Uf'] = fp.PreProcessing.out_data['step'][1]['Uf'].toarray()
        fp.PreProcessing.out_data['step'][1]['Ps'] = fp.PreProcessing.out_data['step'][1]['Ps'].reshape((-1, 1))
        fp.PreProcessing.out_data['step'][1]['tol_vec'] = fp.PreProcessing.out_data['step'][1]['tol_vec'].squeeze()'''
        num_com, lint = 6, 4
        indices1 = np.kron(np.arange(num_com, dtype=np.int64), np.ones((lint, ), dtype=np.int64))
        indices2 = np.kron(np.ones((num_com, ), dtype=np.int64), np.arange(lint, dtype=np.int64))
        fp.PreProcessing.out_data['strain_indices'] = np.stack((indices1, indices2), axis=1)
        return
