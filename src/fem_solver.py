# Imports
import numpy as np
import time
import fem_preprocess as fp
import scipy.io as sio
from scipy.sparse import csr_matrix, lil_matrix, linalg
import mat_subroutine as ms


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

        solver_type = fp.PreProcessing.model_data['solution_control']['solver']
        if solver_type == 1:
            cls.global_linear_solver()
        else:
            raise ValueError('Illegal solver switch parameter ...')

    @classmethod
    def global_linear_solver(cls):

        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sol_data = fp.PreProcessing.sol_data

        isFieldResult = cls.myIsField(model_data['solution_control'], 'assemblefunction')
        if isFieldResult == 1:
            funcd = model_data['solution_control']['assemblefunction']
        else:
            funcd = cls.assemble_system_matrices

        model_data['loading']['load_factor'] = 1
        out_data['step'].append({'Pf': model_data['loading']['Pf'],
                                 'Us': model_data['loading']['Us']})

        tic = time.time()
        tol_vec, exit_flag = cls.newt_raphson_linear(funcd, 2)

        if exit_flag == 1:
            step_id = 2
            out_data['step'][step_id-1]['Uf'] = sol_data['u_n1'][model_data['dof_info']['free_dof']-1]
            out_data['step'][step_id-1]['Ps'] = sol_data['F_int'][model_data['dof_info']['supp_dof']-1].toarray().flatten()

            # Update nodal data: displacement and reactions
            cls.update_nodal_data(step_id)

            out_data['step'][step_id-1]['tol_vec'] = tol_vec.flatten()
            out_data['step'][step_id-1]['iter_vec'] = np.array([[1]])
            out_data['step'][step_id-1]['load_ratio'] = np.array([[1.0]])

            if model_data['solution_control']['print_flag'] == 1:
                print('Analysis successfully Completed ...')
                print(f'Time: {time.time() - tic} sec')
        else:
            print('Analysis failed: Exiting')
            raise ValueError('Illegal exiting flag ...')

    @classmethod
    def newt_raphson_linear(cls, funcd, step_id):

        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sol_data = fp.PreProcessing.sol_data

        tol_cr = model_data['solution_control']['nr_param']['tol_cr']
        free_dof = model_data['dof_info']['free_dof']
        supp_dof = model_data['dof_info']['supp_dof']
        ndof = model_data['dof_info']['ndof']

        # Begin NR: Predictor Step
        sol_data['u_n'] = lil_matrix((ndof, 1))
        sol_data['u_n'][free_dof-1] = out_data['step'][step_id - 2]['Uf']
        sol_data['u_n'][supp_dof-1] = out_data['step'][step_id - 2]['Us']

        # step : n+1 (predictor)
        sol_data['u_n1'] = sol_data['u_n'].copy()
        sol_data['u_n1'][supp_dof-1] = out_data['step'][step_id-1]['Us']
        sol_data['du_n1'] = sol_data['u_n1'] - sol_data['u_n']

        # assemble matrices: Kt, Fint, Fext
        isw = 2
        funcd(step_id, isw)
        R_force = sol_data['F_int'] - sol_data['F_ext']

        duf = -linalg.spsolve(sol_data['Kg'][free_dof-1, :][:, free_dof-1], R_force[free_dof-1, :])

        # Updates
        sol_data['u_n1'][free_dof-1, 0] = sol_data['u_n1'][free_dof-1, 0] + duf.reshape(-1, 1)
        sol_data['du_n1'][free_dof-1] = sol_data['u_n1'][free_dof-1] - sol_data['u_n'][free_dof-1]

        # assemble matrices: Fint, Fext
        isw = 2
        funcd(step_id, isw)
        R_force = sol_data['F_int'] - sol_data['F_ext']

        # CHECK FOR CONVERGENCE
        tol_r = linalg.norm(R_force[free_dof-1, :])
        tol_e = abs(np.squeeze(duf @ R_force[free_dof-1, :]))

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
            exit_flag = 0

        return tol, exit_flag

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
        out_data['step'][step_id-1]['nodal_disp'] = sol_data['u_n1'].toarray().reshape(max_node_dof, nnodes, order='F')

        out_data['step'][step_id-1]['nodal_react'] = np.zeros((ndof,))
        out_data['step'][step_id-1]['nodal_react'][model_data['dof_info']['free_dof']-1] = 0
        out_data['step'][step_id-1]['nodal_react'][model_data['dof_info']['supp_dof']-1] = sol_data['F_int'][
            model_data['dof_info']['supp_dof']-1].toarray().squeeze()
        out_data['step'][step_id - 1]['nodal_react'] = out_data['step'][step_id - 1]['nodal_react'].reshape(
            max_node_dof, nnodes, order='F')

    @classmethod
    def assemble_system_matrices(cls, step_id, isw):
        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sol_data = fp.PreProcessing.sol_data

        LM = model_data['dof_info']['LM']
        nele = model_data['mesh_info']['nele']
        large_disp_flag = model_data['solution_control']['large_disp_flag']
        ele_type = model_data['ele_type']
        ndof = model_data['dof_info']['ndof']
        max_node_dof = model_data['mesh_info']['max_node_dof']
        max_ele_node = model_data['mesh_info']['max_ele_node']

        # Applied External Loads (configuration independent)
        sol_data['F_ext'] = lil_matrix((ndof, 1))
        sol_data['F_ext'][model_data['dof_info']['free_dof']-1] = out_data['step'][step_id-1]['Pf']

        # Initialize matrices
        ele_ndof = max_ele_node * max_node_dof
        Fint_all = np.zeros(ele_ndof * nele)
        Fint_ele = np.zeros(ele_ndof)
        Kt_ele = np.zeros((ele_ndof, ele_ndof))
        K_all = np.zeros(ele_ndof ** 2 * nele)
        M_all = np.zeros(ele_ndof ** 2 * nele) if model_data['solution_control']['dynamics'] == 1 else None

        # Energy
        if model_data['solution_control']['strain_energy_flag'] == 1:
            out_data['step'][step_id-1]['strain_energy'] = np.zeros(nele)

        # Iteration variables
        p = 0
        q = 0

        # Loop over elements
        out_data['step'][step_id - 1]['strain_energy'] = np.zeros((nele, ))
        for i in range(nele):
            nel = model_data['element'][i]['nnodes']
            node_dof = model_data['element'][i]['node_dof']
            ele_ndof = node_dof * nel

            # Properties
            pid = model_data['element'][i]['part_id']
            mat_id = model_data['part'][pid-1]['mat_id']
            sec_id = model_data['part'][pid-1]['sec_id']
            E = model_data['material'][mat_id-1]['E']
            v = model_data['material'][mat_id-1]['v']
            h = model_data['section'][sec_id-1]['thk']

            # Nodal Displacements (total trial)
            ul = sol_data['u_n1'][LM[:, i]-1, 0]
            xl = model_data['element'][i]['coord'][:, 1:3]
            a = np.abs(xl[0, 0] - xl[1, 0]) / 2
            b = np.abs(xl[0, 1] - xl[3, 1]) / 2

            # Call appropriate function for element stiffness matrix
            if ele_type == '1D':
                pass  # Implement beam, trusses, etc.
            elif ele_type == '2D':
                ix = model_data['element'][i]['conn']
                # Nodal Displacements(total trial)
                ul = sol_data['u_n1'][LM[:, i]-1, 0]
                # Nodal Displacements (incremental t_n1 - tn)
                dul = sol_data['du_n1'][LM[:, i]-1, 0]
                ndm = model_data['mesh_info']['space_dim']
                ul = np.reshape(ul, (node_dof, nel), order='F')
                dul = np.reshape(dul, (node_dof, nel), order='F')
                xl = model_data['element'][i]['coord'][:, 1:3].T
                Fint_ele, Kt_ele, m_ele = ms.MaterialSub.solid_2d_lib(ul, dul, xl, ix, ndm, nel, isw, pid, step_id, i+1)
                if model_data['solution_control']['strain_energy_flag'] == 1:
                    ul = np.reshape(ul, (node_dof * nel,), order='F')
                    out_data['step'][step_id-1]['strain_energy'][i] = 0.5 * ul.T @ Kt_ele @ ul

            # Assemble Fint and K matrices
            if isw == 1 or isw == 2:
                Fint_all[q:q + ele_ndof] = Fint_ele.squeeze()
                q += ele_ndof
            if isw == 2 or isw == 3:
                K_all[p:p + ele_ndof ** 2] = Kt_ele.flatten(order='F')
                p += ele_ndof ** 2

        # Assemble matrices Kg, Mg, Kgeos, F_ext and Fint
        if isw == 1 or isw == 2:
            sol_data['F_int'] = csr_matrix((Fint_all, (LM.flatten(order='F')-1, np.zeros(ele_ndof * nele))),
                                           shape=(ndof, 1)).tolil()
        if isw == 2:
            sol_data['Kg'] = csr_matrix(
                (K_all, (model_data['dof_info']['loc_i_array']-1, model_data['dof_info']['loc_j_array']-1)),
                shape=(ndof, ndof))
        if isw == 3:
            sol_data['Kgeos'] = csr_matrix(
                (K_all, (model_data['dof_info']['loc_i_array']-1, model_data['dof_info']['loc_j_array']-1)),
                shape=(ndof, ndof))
        if model_data['solution_control']['dynamics'] == 1:
            sol_data['Mg'] = csr_matrix(
                (M_all, (model_data['dof_info']['loc_i_array']-1, model_data['dof_info']['loc_j_array']-1)),
                shape=(ndof, ndof))

        # Update F_ext for frame elements or configuration dependent loading
        # sol_data['F_ext'] = ...

        # Compliant mechanism Design
        if large_disp_flag == 1:
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
                                                           sol_data['u_n1'][dof_id-1, 0])

