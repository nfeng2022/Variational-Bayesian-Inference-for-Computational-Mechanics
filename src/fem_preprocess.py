# imports
import numpy as np
import hdf5storage as h5
from scipy.sparse import lil_matrix
import model_property_cards as mp
import scipy.io as sio
import tensorflow as tf


class PreProcessing:

    # Define global constants
    constants = {
        'eight9': 0.888888888888889,
        'five9': 0.555555555555556,
        'one3': 0.333333333333333,
        'one9': 0.111111111111111,
        'sqtp6': 0.774596669241483,
        'sqt13': 0.577350269189626,
        'sqt4p8': 2.190890230020664,
        'thty29': 1.034482758620690
    }

    # Clear global variables
    model_data = {}
    out_data = {}
    topo_data = {}
    sol_data = {}
    topo_data_ele = {}
    out_data_cells = {}

    Pdevs = np.array([
        [0.666666666666667, 0, 0, 0, -0.333333333333333, 0, 0, 0, -0.333333333333333],
        [0, 0.500000000000000, 0, 0.500000000000000, 0, 0, 0, 0, 0],
        [0, 0, 0.500000000000000, 0, 0, 0, 0.500000000000000, 0, 0],
        [0, 0.500000000000000, 0, 0.500000000000000, 0, 0, 0, 0, 0],
        [-0.333333333333333, 0, 0, 0, 0.666666666666667, 0, 0, 0, -0.333333333333333],
        [0, 0, 0, 0, 0, 0.500000000000000, 0, 0.500000000000000, 0],
        [0, 0, 0.500000000000000, 0, 0, 0, 0.500000000000000, 0, 0],
        [0, 0, 0, 0, 0, 0.500000000000000, 0, 0.500000000000000, 0],
        [-0.333333333333333, 0, 0, 0, -0.333333333333333, 0, 0, 0, 0.666666666666667]
    ])
    Pvol = np.array([
        [0.333333333333333, 0, 0, 0, 0.333333333333333, 0, 0, 0, 0.333333333333333],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.333333333333333, 0, 0, 0, 0.333333333333333, 0, 0, 0, 0.333333333333333],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.333333333333333, 0, 0, 0, 0.333333333333333, 0, 0, 0, 0.333333333333333]
    ])
    sg2 = np.array([0, 0, 1])

    # Model initialization
    @classmethod
    def modeldata_initialization_topopt(cls, infile_name, model_file_name):

        # (1) Read Input data
        cls.get_input_data(infile_name)

        # Assign variables
        model_data = cls.model_data
        out_data = cls.out_data
        topo_data = cls.topo_data
        sol_data = cls.sol_data
        topo_data_ele = cls.topo_data_ele
        out_data_cells = cls.out_data_cells

        # Check loading conditions
        if not model_data['loading']['nodal_load'].any() and not model_data['loading']['nodal_disp'].any():
            raise ValueError('There is neither applied displacement nor load.')
        elif model_data['loading']['nodal_load'].any() and model_data['loading']['nodal_disp'].any():
            raise ValueError('Currently we don''t consider applying displacement and load simultaneously.')
        elif not model_data['loading']['nodal_load'].any():
            disp_control_flag = 1
            print('Displacement control is considered.')
        else:
            disp_control_flag = 0
            print('Load control is considered.')

        # (2) Read material data
        model_data['material'], model_data['section'], model_data['part'], \
            model_data['solution_control'], model_data['ele_type'] = \
            mp.model_property_cards_fun(disp_control_flag)

        # (3) Assign DOF
        cls.assign_dof_parfor_topopt()

        # (4) Update part_id
        nele = model_data['mesh_info']['nele']
        for i in range(nele):
            model_data['element'][i]['part_id'] = model_data['part'][0]['id']

        # (6) Assign Storage
        cls.assign_storage_topopt()

        # (7) Get Element Shape Data
        cls.get_element_data_topopt()

        # (8) Save file
        data_dic = {'model_data': model_data,
                    'out_data': out_data,
                    'sol_data': sol_data,
                    'constants': cls.constants,
                    'topo_data': topo_data,
                    'topo_data_ele': topo_data_ele,
                    'out_data_cells': out_data_cells,
                    'sg2': cls.sg2}
        # h5.write(data=data_dic, path='', filename=model_file_name)
        sio.savemat(model_file_name, data_dic)

    @classmethod
    def get_input_data(cls, file_name):

        # Open file for reading
        with open(file_name, 'r') as fid_in:
            fid_in.readline()
            tline = fid_in.readline().strip()
            mesh_data = np.array(tline.split(), dtype=np.float64)

            nnodes = int(mesh_data[0])
            nele = int(mesh_data[1])
            space_dim = int(mesh_data[3])
            max_node_dof = int(mesh_data[4])
            max_ele_node = int(mesh_data[5])

            # Reading coordinate data
            while True:
                tline = fid_in.readline()
                if tline.strip() == 'COORdinates ALL' or tline == '':
                    break

            if space_dim == 2:
                coord = np.loadtxt(fid_in, dtype=np.float64, max_rows=nnodes, usecols=(0, 1, 2, 3))
            elif space_dim == 3:
                coord = np.loadtxt(fid_in, dtype=np.float64, max_rows=nnodes, usecols=(0, 1, 2, 3, 4))

            # Reading element data
            while True:
                tline = fid_in.readline()
                if tline.strip() == 'ELEMents ALL' or tline == '':
                    break

            element_info = np.zeros((nele, max_ele_node + 3), dtype=np.uint32)
            for i in range(nele):
                element_info[i] = np.array(fid_in.readline().strip().split(), dtype=np.uint32)

            # Reading boundary conditions
            flag = False
            while True:
                tline = fid_in.readline()
                if tline.strip() == 'BOUNdary conditions':
                    flag = True
                    break
                if tline == '':
                    break

            boundary_data = []
            if flag:
                while True:
                    tline = fid_in.readline().strip()
                    if not tline:
                        break
                    else:
                        boundary_data.append(np.array(tline.split(), dtype=np.uint32))
                boundary_data = np.stack(boundary_data, axis=0)
            boundary_data = np.array(boundary_data)

            # Reading load data
            flag = False
            while True:
                tline = fid_in.readline()
                if tline.strip() == 'FORCe conditions':
                    flag = True
                    break
                if tline == '':
                    break

            load_data = []
            if flag:
                while True:
                    tline = fid_in.readline().strip()
                    if not tline:
                        break
                    else:
                        load_data.append(np.array(tline.split(), dtype=np.float64))
                load_data = np.stack(load_data, axis=0)
            load_data = np.array(load_data)

            # Reading displacement data
            flag = False
            while True:
                tline = fid_in.readline()
                if tline.strip() == 'DISPlacement conditions':
                    flag = True
                    break
                if tline == '':
                    break

            disp_data = []
            if flag:
                while True:
                    tline = fid_in.readline().strip()
                    if not tline:
                        break
                    else:
                        disp_data.append(np.array(tline.split(), dtype=np.float64))
                disp_data = np.stack(disp_data, axis=0)
            disp_data = np.array(disp_data)

        # Remove second column from arrays
        element_info = np.delete(element_info, 1, axis=1)
        coord = np.delete(coord, 1, axis=1)
        if boundary_data.size > 0:
            boundary_data = np.delete(boundary_data, 1, axis=1)
        if load_data.size > 0:
            load_data = np.delete(load_data, 1, axis=1)
        if disp_data.size > 0:
            disp_data = np.delete(disp_data, 1, axis=1)

        # Initialize element dictionary
        element = []
        for iter in range(nele):
            elem = {}
            elem['id'] = element_info[iter, 0]
            elem['part_id'] = None
            elem['conn'] = element_info[iter, 2:max_ele_node + 2]
            elem['nnodes'] = len(elem['conn'])

            elem['coord'] = []
            for k in range(elem['nnodes']):
                nid = elem['conn'][k]
                elem['coord'].append(coord[coord[:, 0] == nid][0, :])
            elem['coord'] = np.stack(elem['coord'], axis=0)

            if space_dim == 2:
                if max_node_dof == 2:
                    elem['node_dof'] = 2
                elif max_node_dof == 3:
                    elem['node_dof'] = 3
            elif space_dim == 3:
                elem['node_dof'] = 3

            element.append(elem)

        # Create support matrix
        support = lil_matrix((nnodes, boundary_data.shape[1]), dtype=np.uint32)
        support[:, 0] = np.arange(1, nnodes + 1)
        support[boundary_data[:, 0].astype(int) - 1, :] = boundary_data

        # Update support matrix with displacement data
        if disp_data.size > 0:
            temp_data = np.copy(disp_data)
            temp_data[temp_data != 0.0] = 1
            temp_data[:, 0] = disp_data[:, 0].astype(int)

            if space_dim == 2:
                support[temp_data[:, 0].astype(int) - 1, 1:3] += temp_data[:, 1:3]
            elif space_dim == 3:
                support[temp_data[:, 0].astype(int) - 1, 1:4] += temp_data[:, 1:4]

        # Store data in model_data dictionary
        mesh_info = {
            'nele': nele,
            'nnodes': nnodes,
            'coord': coord,
            'space_dim': space_dim,
            'max_node_dof': max_node_dof,
            'max_ele_node': max_ele_node
        }

        loading = {
            'nodal_load': load_data,
            'nodal_disp': disp_data
        }

        sets = {
            'nset': [{'id': boundary_data[:, 0].tolist()}]
        }

        cls.model_data = {
            'element': element,
            'support': support,
            'loading': loading,
            'mesh_info': mesh_info,
            'sets': sets
        }

    @classmethod
    def assign_dof_parfor_topopt(cls):

        model_data = cls.model_data
        nnodes = model_data['mesh_info']['nnodes']
        nele = model_data['mesh_info']['nele']
        max_node_dof = model_data['mesh_info']['max_node_dof']
        max_ele_node = model_data['mesh_info']['max_ele_node']
        support = model_data['support']
        nodal_load = model_data['loading']['nodal_load']
        nodal_disp = model_data['loading']['nodal_disp']

        # DOF Assignment
        ndof = nnodes * max_node_dof
        ID = np.arange(1, ndof + 1).reshape(nnodes, max_node_dof).T

        # Equality Constraints
        if model_data['solution_control']['eqconstraints'] > 0:
            for k in range(model_data['solution_control']['eqconstraints']):
                nid = model_data['solution_control']['eqcgroup'][k]['nid']
                dof = model_data['solution_control']['eqcgroup'][k]['dof']
                ID[dof - 1, nid - 1] = np.nan

            tempID = ID[~np.isnan(ID)]
            numdof = tempID.size
            tempID = np.arange(1, numdof + 1)
            ID[~np.isnan(ID)] = tempID

            for k in range(model_data['solution_control']['eqconstraints']):
                nid = model_data['solution_control']['eqcgroup'][k]['nid']
                dof = model_data['solution_control']['eqcgroup'][k]['dof']
                ID[dof - 1, nid - 1] = numdof + k + 1

            ndof = numdof + model_data['solution_control']['eqconstraints']

        IEN = np.zeros((max_ele_node, nele), dtype=int)
        all_dof = np.arange(1, ndof + 1)
        supp_dof = []

        if max_node_dof == 2:
            x_id = support[:, 1] == 1
            supp_x = support[np.squeeze(x_id.toarray())]
            y_id = support[:, 2] == 1
            supp_y = support[np.squeeze(y_id.toarray())]
            dof_idx = ID[0, supp_x.astype(int).toarray()-1]
            dof_idy = ID[1, supp_y.astype(int).toarray()-1]
            supp_dof = np.concatenate((dof_idx, dof_idy), axis=1).T
        elif max_node_dof == 3:
            x_id = support[:, 1] == 1
            supp_x = support[np.squeeze(x_id.toarray())]
            y_id = support[:, 2] == 1
            supp_y = support[np.squeeze(y_id.toarray())]
            z_id = support[:, 3] == 1
            supp_z = support[np.squeeze(z_id.toarray())]
            dof_idx = ID[0, supp_x.astype(int).toarray() - 1]
            dof_idy = ID[1, supp_y.astype(int).toarray() - 1]
            dof_idz = ID[2, supp_z.astype(int).toarray() - 1]
            supp_dof = np.concatenate((dof_idx, dof_idy, dof_idz), axis=1)

        supp_dof = np.unique(supp_dof)
        free_dof = np.setdiff1d(all_dof, supp_dof)
        LM = np.zeros((max_ele_node * max_node_dof, nele), dtype=int)

        if max_node_dof == 2:
            for i in range(nele):
                IEN[:, i] = model_data['element'][i]['conn']
                LM[:, i] = ID[:, IEN[:, i] - 1].flatten(order='F')
        elif max_node_dof == 3:
            for i in range(nele):
                IEN[:, i] = model_data['element'][i]['conn']
                LM[:, i] = ID[:, IEN[:, i] - 1].flatten(order='F')

        # Nodal loads and displacement
        D_all_dof = lil_matrix((ndof, 1))
        P_all_dof = lil_matrix((ndof, 1))

        if max_node_dof == 2:
            if nodal_load.size > 0:
                load_x = nodal_load[nodal_load[:, 1] != 0]
                load_y = nodal_load[nodal_load[:, 2] != 0]
                dof_idx = ID[0, load_x[:, 0].astype(int) - 1]
                dof_idy = ID[1, load_y[:, 0].astype(int) - 1]
                for k in range(dof_idx.size):
                    P_all_dof[dof_idx[k] - 1, 0] += load_x[k, 1]
                for k in range(dof_idy.size):
                    P_all_dof[dof_idy[k] - 1, 0] += load_y[k, 2]

            if nodal_disp.size > 0:
                for i in range(nodal_disp.shape[0]):
                    nid = nodal_disp[i, 0].astype(int)
                    if ID[0, nid - 1] in supp_dof:
                        D_all_dof[ID[0, nid - 1] - 1, 0] = nodal_disp[i, 1]
                    if ID[1, nid - 1] in supp_dof:
                        D_all_dof[ID[1, nid - 1] - 1, 0] = nodal_disp[i, 2]

        elif max_node_dof == 3:
            if nodal_load.size > 0:
                load_x = nodal_load[nodal_load[:, 1] != 0]
                load_y = nodal_load[nodal_load[:, 2] != 0]
                load_z = nodal_load[nodal_load[:, 3] != 0]
                dof_idx = ID[0, load_x[:, 0].astype(int) - 1]
                dof_idy = ID[1, load_y[:, 0].astype(int) - 1]
                dof_idz = ID[2, load_z[:, 0].astype(int) - 1]
                for k in range(dof_idx.size):
                    P_all_dof[dof_idx[k] - 1, 0] += load_x[k, 1]
                for k in range(dof_idy.size):
                    P_all_dof[dof_idy[k] - 1, 0] += load_y[k, 2]
                for k in range(dof_idz.size):
                    P_all_dof[dof_idz[k] - 1, 0] += load_z[k, 3]

            if nodal_disp.size > 0:
                for i in range(nodal_disp.shape[0]):
                    nid = nodal_disp[i, 0].astype(int)
                    if ID[0, nid - 1] in supp_dof:
                        D_all_dof[ID[0, nid - 1] - 1, 0] = nodal_disp[i, 1]
                    if ID[1, nid - 1] in supp_dof:
                        D_all_dof[ID[1, nid - 1] - 1, 0] = nodal_disp[i, 2]
                    if ID[2, nid - 1] in supp_dof:
                        D_all_dof[ID[2, nid - 1] - 1, 0] = nodal_disp[i, 3]

        # Sparse array indices
        ele_ndof = max_ele_node * max_node_dof
        loc_i_array = np.zeros(ele_ndof * ele_ndof * nele, dtype=int)
        loc_j_array = np.zeros(ele_ndof * ele_ndof * nele, dtype=int)
        p = 0
        onearray1 = np.ones(ele_ndof, dtype=int)
        onearray2 = np.ones(ele_ndof, dtype=int).reshape(1, -1)

        for i in range(nele):
            loc_j_array[p:p + ele_ndof * ele_ndof] = np.kron(LM[:, i], onearray1)
            loc_i_array[p:p + ele_ndof * ele_ndof] = (np.kron(LM[:, i].reshape(-1, 1), onearray2)).flatten(order='F')
            p += ele_ndof * ele_ndof

        # Store information
        model_data['boundary'] = {}
        model_data['dof_info'] = {}
        model_data['loading'] = {}
        model_data['boundary']['D_all_dof'] = D_all_dof
        model_data['boundary']['P_all_dof'] = P_all_dof
        model_data['dof_info']['LM'] = LM  # Column ordering
        model_data['dof_info']['ID'] = ID  # Column ordering
        model_data['dof_info']['IEN'] = IEN.T  # Row ordering
        model_data['dof_info']['all_dof'] = all_dof
        model_data['dof_info']['free_dof'] = free_dof
        model_data['dof_info']['supp_dof'] = supp_dof
        model_data['dof_info']['ndof'] = ndof
        model_data['dof_info']['nsupp'] = len(supp_dof)
        model_data['dof_info']['nfree'] = len(free_dof)
        model_data['dof_info']['loc_i_array'] = loc_i_array
        model_data['dof_info']['loc_j_array'] = loc_j_array
        model_data['loading']['Pf'] = P_all_dof[model_data['dof_info']['free_dof'] - 1, 0]
        model_data['loading']['Ps'] = P_all_dof[model_data['dof_info']['supp_dof'] - 1, 0]
        model_data['loading']['Us'] = D_all_dof[supp_dof - 1, 0]

    @classmethod
    def assign_storage_topopt(cls):
        model_data = cls.model_data
        out_data = cls.out_data

        nnodes = model_data['mesh_info']['nnodes']
        nele = model_data['mesh_info']['nele']
        large_disp_flag = model_data['solution_control']['large_disp_flag']
        sec_id = model_data['part'][0]['sec_id']
        mat_id = model_data['part'][0]['mat_id']
        estorage = model_data['section'][sec_id-1]['estorage']

        if estorage == 1:
            nalpha = model_data['section'][sec_id-1]['nalpha']

        int_pts = model_data['section'][sec_id-1]['intp']
        in_flag = model_data['material'][mat_id-1]['in_flag']
        if in_flag == 1:
            hsv = model_data['material'][mat_id-1]['hsv']
            ini_hsv = model_data['material'][mat_id-1]['ini_hsv']

        ele_type = model_data['ele_type']
        nel = model_data['element'][0]['nnodes']

        if ele_type == '1D':
            pass
        elif ele_type == '2D':
            _, nip = cls.quadr2d(int_pts, nel, 0)
        elif ele_type == '3D':
            _, nip = cls.quadr3d(int_pts, nel)

        numsteps = model_data['solution_control']['load_control']['numsteps']

        if nip > 0:
            if large_disp_flag == 1:
                out_data['eleF'] = np.zeros((4, nip, nele, numsteps + 1))
                out_data['ele_stressP'] = np.zeros((4, nip, nele, numsteps + 1))
                out_data['ele_A'] = np.zeros((16, nip, nele, numsteps + 1))
                out_data['ele_energy'] = np.zeros((1, nip, nele, numsteps + 1))
                out_data['flg'] = np.zeros((nele, 1))
                for i in range(nele):
                    out_data['eleF'][:, :, i, 1] = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
            else:
                out_data['ele_stress'] = np.zeros((6, nip, nele, numsteps + 1))
                out_data['ele_strain'] = np.zeros((6, nip, nele, numsteps + 1))
                if estorage == 1:
                    out_data['ele_alpha'] = np.zeros((nalpha, nele, numsteps + 1))
                if in_flag == 1:
                    out_data['ele_estrain'] = np.zeros((6, nip, nele, numsteps + 1))
                    out_data['ele_pstrain'] = np.zeros((6, nip, nele, numsteps + 1))
                    out_data['ele_hsv'] = np.zeros((hsv, nip, nele, numsteps + 1))
                    out_data['ele_nvec'] = np.zeros((6, nip, nele, numsteps + 1))
                    out_data['ele_gamma'] = np.zeros((1, nip, nele, numsteps + 1))
                    out_data['ele_info'] = np.zeros((6, nip, nele, numsteps + 1))  # Adjust the size accordingly
                    out_data['ele_eenergy'] = np.zeros((1, nip, nele, numsteps + 1))
                    out_data['ele_ienergy'] = np.zeros((1, nip, nele, numsteps + 1))
                    out_data['ele_penergy'] = np.zeros((1, nip, nele, numsteps + 1))

        out_data['step'] = []
        out_data['step'].append({'nodal_disp': np.zeros((3, nnodes)),
                                 'nodal_force': np.zeros((3, nnodes)),
                                 'Uf': np.zeros((model_data['dof_info']['nfree'], 1)),
                                 'Us': np.zeros((model_data['dof_info']['nsupp'], 1)),
                                 'Pf': np.zeros((model_data['dof_info']['nfree'], 1)),
                                 'Ps': np.zeros((model_data['dof_info']['nsupp'], 1))})

    @classmethod
    def quadr2d(cls, int_pts, nel, nodal_flag):
        l = min(5, int_pts)
        if nel == 4:
            if l == 0:
                l = 2
        elif nel <= 9:
            if l == 0:
                l = 3
        else:
            if l == 0:
                l = 4

        if nodal_flag:
            sg, lint = cls.int2dn(l)
        else:
            sg, lint = cls.int2d(l)
        return sg, lint

    @classmethod
    def int2d(cls, l):
        lint = l * l
        sg = np.zeros((3, lint))

        lr = np.array([-1, 1, 1, -1, 0, 1, 0, -1, 0])
        lz = np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])
        lw = np.array([25, 25, 25, 25, 40, 40, 40, 40, 64])

        if l == 0:
            lint = 5
            g = cls.constants['sqtp6']
            for i in range(4):
                sg[0, i] = g * lr[i]
                sg[1, i] = g * lz[i]
                sg[2, i] = cls.constants['five9']
            sg[0, 4] = 0.00
            sg[1, 4] = 0.00
            sg[2, 4] = 2.80 * cls.constants['eight9']
        elif l == 1:
            sg[0, 0] = 0.00
            sg[1, 0] = 0.00
            sg[2, 0] = 4.00
        elif l == 2:
            g = cls.constants['sqt13']
            for i in range(4):
                sg[0, i] = g * lr[i]
                sg[1, i] = g * lz[i]
                sg[2, i] = 1.00
        elif l == 3:
            g = cls.constants['sqtp6']
            h = 1.0 / 81.0
            for i in range(9):
                sg[0, i] = g * lr[i]
                sg[1, i] = g * lz[i]
                sg[2, i] = h * lw[i]
        elif l == 4:
            g = cls.constants['sqt4p8']
            h = cls.constants['one3'] / g
            ss = np.zeros(4)
            ww = np.zeros(4)
            ss[0] = np.sqrt((3.0 + g) / 7.0)
            ss[3] = -ss[0]
            ss[1] = np.sqrt((3.0 - g) / 7.0)
            ss[2] = -ss[1]
            ww[0] = 0.50 - h
            ww[1] = 0.50 + h
            ww[2] = 0.50 + h
            ww[3] = 0.50 - h
            i = 0
            for j in range(4):
                for k in range(4):
                    sg[0, i] = ss[k]
                    sg[1, i] = ss[j]
                    sg[2, i] = ww[j] * ww[k]
                    i += 1
        elif l == 5:
            g = np.sqrt(1120.00)
            ss = np.zeros(5)
            ww = np.zeros(5)
            ss[0] = np.sqrt((70.0 + g) / 126.0)
            ss[1] = np.sqrt((70.0 - g) / 126.0)
            ss[2] = 0.0
            ss[3] = -ss[1]
            ss[4] = -ss[0]
            ww[0] = (21.0 * g + 117.6) / (g * (70.0 + g))
            ww[1] = (21.0 * g - 117.6) / (g * (70.0 - g))
            ww[2] = 2.0 * (1.0 - ww[0] - ww[1])
            ww[3] = ww[1]
            ww[4] = ww[0]
            i = 0
            for j in range(5):
                for k in range(5):
                    sg[0, i] = ss[k]
                    sg[1, i] = ss[j]
                    sg[2, i] = ww[j] * ww[k]
                    i += 1
        else:
            print(" *ERROR* INT2D: Illegal quadrature order")

        return sg, lint

    @classmethod
    def int2dn(cls, l):
        lint = l
        sg = np.zeros((3, lint))

        x2 = np.array([-1, 1, 1, -1, 0, 1, 0, -1, 0])
        y2 = np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])
        w2 = np.array([1, 1, 1, 1, 4, 4, 4, 4, 16])
        x3 = np.array([-3, 3, 3, -3, -1, 1, 3, 3, 1, -1, -3, -3, -1, 1, 1, -1])
        y3 = np.array([-3, -3, 3, 3, -3, -3, -1, 1, 3, 3, 1, -1, -1, -1, 1, 1])
        w3 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9])

        if l == 4:
            for i in range(4):
                sg[0, i] = x2[i]
                sg[1, i] = y2[i]
                sg[2, i] = 1.00
        elif l == 9:
            h = cls.constants['one9']
            for i in range(9):
                sg[0, i] = x2[i]
                sg[1, i] = y2[i]
                sg[2, i] = w2[i] * h
        elif l == 16:
            g = cls.constants['one3']
            h = 0.06250
            for i in range(16):
                sg[0, i] = x3[i] * g
                sg[1, i] = y3[i] * g
                sg[2, i] = w3[i] * h
        else:
            print(" *ERROR* INT2DN: Illegal quadrature order")

        return sg, lint

    @classmethod
    def quadr3d(cls, int_pts, nel):
        sg, lint = cls.int3d(int_pts)
        return sg, lint

    @classmethod
    def int3d(cls, num_pt):
        lint = 0
        s = None

        ig = np.array([-1, 1, 1, -1])
        jg = np.array([-1, -1, 1, 1])

        if num_pt == 1:
            lint = 1
            s = np.zeros((4, 1))
            s[3, 0] = 8.0
        elif num_pt == 2:
            lint = 8
            g = cls.constants['sqt13']
            s = np.zeros((4, 8))
            for i in range(4):
                s[0, i] = ig[i] * g
                s[0, i + 4] = s[0, i]
                s[1, i] = jg[i] * g
                s[1, i + 4] = s[1, i]
                s[2, i] = g
                s[2, i + 4] = -g
                s[3, i] = 1.0
                s[3, i + 4] = 1.0
        elif num_pt == -9:
            lint = 9
            g = cls.constants['sqtp6']
            s = np.zeros((4, 9))
            for i in range(4):
                s[0, i] = ig[i] * g
                s[0, i + 4] = s[0, i]
                s[1, i] = jg[i] * g
                s[1, i + 4] = s[1, i]
                s[2, i] = g
                s[2, i + 4] = -g
                s[3, i] = cls.constants['five9']
                s[3, i + 4] = cls.constants['five9']
            s[3, 8] = cls.constants['thty29']
        elif num_pt == -4:
            lint = 4
            s = np.zeros((4, 4))
            g = cls.constants['sqt13']
            for i in range(4):
                s[0, i] = ig[i] * g
                s[1, i] = s[0, i]
                s[2, i] = jg[i] * g
                s[3, i] = 2.0
            s[1, 2] = -g
            s[1, 3] = g
        elif num_pt <= 5:
            sw = PreProcessing.int1d(num_pt)
            lint = 0
            s = np.zeros((4, num_pt * num_pt * num_pt))
            for k in range(num_pt):
                for j in range(num_pt):
                    for i in range(num_pt):
                        s[0, lint] = sw[0, i]
                        s[1, lint] = sw[0, j]
                        s[2, lint] = sw[0, k]
                        s[3, lint] = sw[1, i] * sw[1, j] * sw[1, k]
                        lint += 1
        else:
            print("Error -> int3d.m: Illegal quadrature order")

        return s, lint

    @staticmethod
    def int1d(num_pt):
        if num_pt == 1:
            points = np.array([0])
            weights = np.array([2])
        elif num_pt == 2:
            points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
            weights = np.array([1, 1])
        elif num_pt == 3:
            points = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
            weights = np.array([5 / 9, 8 / 9, 5 / 9])
        elif num_pt == 4:
            points = np.array([-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                               np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)])
            weights = np.array(
                [(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36])
        elif num_pt == 5:
            points = np.array([-np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3, -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 0,
                               np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3])
            weights = np.array([(322 - 13 * np.sqrt(70)) / 900, (322 + 13 * np.sqrt(70)) / 900, 128 / 225,
                                (322 + 13 * np.sqrt(70)) / 900, (322 - 13 * np.sqrt(70)) / 900])
        else:
            raise ValueError("Unsupported number of integration points")

        sw = np.array([points, weights])
        return sw

    @classmethod
    def get_element_data_topopt(cls):
        model_data = cls.model_data
        topo_data = cls.topo_data
        topo_data_ele = cls.topo_data_ele

        part_id = model_data['element'][0]['part_id']
        sec_id = model_data['part'][part_id-1]['sec_id']
        int_pts = model_data['section'][sec_id-1]['intp']

        etype = model_data['section'][sec_id-1]['etype']
        ndm = model_data['mesh_info']['space_dim']
        nel = model_data['mesh_info']['max_ele_node']
        ix = model_data['element'][0]['conn']
        xl = model_data['element'][0]['coord'][:, 1:3].T
        # Shape functions and derivatives at the centroid
        cls.sg2 = np.array([[0], [0], [1]])
        shp0, _ = cls.interp2d(0, xl, ix, ndm, nel, 0, etype)

        Bm0 = np.zeros((4, nel * 2))
        j = 0
        for i in range(nel):
            Bm0[0, j] = shp0[0, i]
            Bm0[1, j + 1] = shp0[0, i]
            Bm0[2, j] = shp0[1, i]
            Bm0[3, j + 1] = shp0[1, i]
            j += 2

        cls.sg2, lint = cls.quadr2d(int_pts, nel, 0)
        jac = np.zeros(lint)
        shp = np.zeros((3, nel, lint))
        Bm = np.zeros((4, nel * 2, lint))
        for ipt in range(lint):
            shp[:, :, ipt], jac[ipt] = cls.interp2d(ipt, xl, ix, ndm, nel, 0, etype)
            j = 0
            for i in range(nel):
                Bm[0, j, ipt] = shp[0, i, ipt]
                Bm[1, j + 1, ipt] = shp[0, i, ipt]
                Bm[2, j, ipt] = shp[1, i, ipt]
                Bm[3, j + 1, ipt] = shp[1, i, ipt]
                j += 2

        element_kdata = {}
        element_kdata['shp0'] = shp0
        element_kdata['Bm0'] = Bm0
        element_kdata['shp'] = shp
        element_kdata['Bm'] = Bm
        element_kdata['jac'] = jac
        element_kdata['lint'] = lint
        element_kdata['thk'] = model_data['section'][sec_id-1]['thk']
        element_kdata['dvol'] = model_data['section'][sec_id-1]['thk'] * jac

        v = model_data['material'][0]['v']
        kb = 1 / 3 / (1 - 2 * v)
        mu = 1 / (2 * (1 + v))
        Ce = 3 * kb * cls.Pvol + 2 * mu * cls.Pdevs
        Ce = Ce[[0, 1, 3, 4], :][:, [0, 1, 3, 4]]

        kl = np.zeros((nel * 2, nel * 2))
        BmL = np.zeros((4, nel * 2, lint))
        for ipt in range(lint):
            BmL[:, :, ipt] = Bm[:, :, ipt]
            Z = BmL[1:3, :, ipt].copy()
            Z[0, :] = 1 / 2 * (Z[0, :] + Z[1, :])
            Z[1, :] = Z[0, :]
            BmL[1:3, :, ipt] = Z
            dvol = element_kdata['dvol'][ipt]
            kl += dvol * (BmL[:, :, ipt].T @ Ce @ BmL[:, :, ipt])

        element_kdata['BmL'] = BmL
        element_kdata['LinE'] = {'Pdevs': cls.Pdevs, 'Pvol': cls.Pvol}
        element_kdata['Ktl_ele'] = kl
        element_kdata['CL'] = Ce

        topo_data['element_kdata'] = element_kdata
        topo_data_ele['element_kdata'] = element_kdata

    @classmethod
    def interp2d(cls, qid, xl, ix, ndm, nel, flg, etype):
        """
        Interpolation functions for 2-D elements

        Parameters:
        qid (int): Quadrature point
        xl (ndarray): Nodal coordinates
        ix (ndarray): Global nodal connections
        ndm (int): Mesh coordinate dimension
        nel (int): Number of element nodes
        flg (int): Global derivatives if false
        etype (int): Element type

        Returns:
        shp (ndarray): Shape functions
        jac (float): Jacobian determinant
        """
        if etype == 1:
            shp, jac = PreProcessing.shp2d(cls.sg2[0:2, qid], xl, ndm, nel, ix, flg)
            jac *= cls.sg2[2, qid]
        elif etype == 2:
            shp, jac = None, None
        return shp, jac

    @staticmethod
    def shp2d(ss, xl, ndm, nel, ix, flg):
        """
        Shape functions for quadrilateral elements

        Parameters:
        ss (ndarray): Natural coordinates for result point
        xl (ndarray): Nodal coordinates for result element
        ndm (int): Spatial dimension of mesh
        nel (int): Number of nodes on element
        ix (ndarray): Nodes attached to element
        flg (int): Compute global x/y derivatives if false, else derivatives w.r.t. natural coords

        Returns:
        shp (ndarray): Shape functions and derivatives at point
        xsj (float): Jacobian determinant at point
        """
        s = np.array([-0.5, 0.5, 0.5, -0.5])
        t = np.array([-0.5, -0.5, 0.5, 0.5])

        shp = np.zeros((3, nel))

        if nel == 4 and not flg:
            shp, xsj = PreProcessing.shapef(ss, xl, flg)
        elif nel == 16:
            shp, xsj = PreProcessing.shp2dc(ss, xl, 1, flg)
        elif nel == 12:
            shp, xsj = PreProcessing.shp2ds(ss, xl, 1, flg)
        else:
            for i in range(4):
                shp[2, i] = (0.5 + s[i] * ss[0]) * (0.5 + t[i] * ss[1])
                shp[0, i] = s[i] * (0.5 + t[i] * ss[1])
                shp[1, i] = t[i] * (0.5 + s[i] * ss[0])

            if nel == 3:
                for i in range(3):
                    shp[i, 2] += shp[i, 3]

            if nel > 4:
                shp = PreProcessing.shap2(ss[0], ss[1], shp, ix, nel)

            xs = np.zeros((2, 2))
            xs[0, 0] = np.dot(shp[0, :nel], xl[0, :nel])
            xs[0, 1] = np.dot(shp[1, :nel], xl[0, :nel])
            xs[1, 0] = np.dot(shp[0, :nel], xl[1, :nel])
            xs[1, 1] = np.dot(shp[1, :nel], xl[1, :nel])
            xsj = xs[0, 0] * xs[1, 1] - xs[0, 1] * xs[1, 0]

            if not flg:
                temp = 1.0 / xsj if xsj != 0.0 else 1.0
                sx = np.array([[xs[1, 1] * temp, -xs[0, 1] * temp],
                               [-xs[1, 0] * temp, xs[0, 0] * temp]])
                shp[0:2, :nel] = np.dot(sx.T, shp[0:2, :nel])

        return shp, xsj

    @staticmethod
    def shapef(s, xl, flg):
        """
        Quadrilateral shape functions

        Parameters:
        s (ndarray): Natural coordinates of point
        xl (ndarray): Nodal coordinates for element
        flg (int): Compute global derivatives if false, else derivatives w.r.t. natural coords

        Returns:
        shp (ndarray): Shape functions and derivatives at point
        xsj (float): Jacobian determinant at point
        """
        sh = 0.5 * s[0]
        th = 0.5 * s[1]
        sp = 0.5 + sh
        tp = 0.5 + th
        sm = 0.5 - sh
        tm = 0.5 - th
        xo = xl[0, 0] - xl[0, 1] + xl[0, 2] - xl[0, 3]
        xs = -xl[0, 0] + xl[0, 1] + xl[0, 2] - xl[0, 3] + xo * s[1]
        xt = -xl[0, 0] - xl[0, 1] + xl[0, 2] + xl[0, 3] + xo * s[0]
        yo = xl[1, 0] - xl[1, 1] + xl[1, 2] - xl[1, 3]
        ys = -xl[1, 0] + xl[1, 1] + xl[1, 2] - xl[1, 3] + yo * s[1]
        yt = -xl[1, 0] - xl[1, 1] + xl[1, 2] + xl[1, 3] + yo * s[0]
        xsj1 = xs * yt - xt * ys
        xsj = 0.0625 * xsj1

        if not flg:
            xsj1 = 1.0 / xsj1 if xsj1 != 0.0 else 1.0
            xs = (xs + xs) * xsj1
            xt = (xt + xt) * xsj1
            ys = (ys + ys) * xsj1
            yt = (yt + yt) * xsj1
            ytm = yt * tm
            ysm = ys * sm
            ytp = yt * tp
            ysp = ys * sp
            xtm = xt * tm
            xsm = xs * sm
            xtp = xt * tp
            xsp = xs * sp
            shp = np.zeros((3, 4))
            shp[0, 0] = -ytm + ysm
            shp[0, 1] = ytm + ysp
            shp[0, 2] = ytp - ysp
            shp[0, 3] = -ytp - ysm
            shp[1, 0] = xtm - xsm
            shp[1, 1] = -xtm - xsp
            shp[1, 2] = -xtp + xsp
            shp[1, 3] = xtp + xsm
        else:
            shp = np.zeros((3, 4))
            shp[0, 0] = -0.50 * tm
            shp[0, 1] = 0.50 * tm
            shp[0, 2] = 0.50 * tp
            shp[0, 3] = -0.50 * tp
            shp[1, 0] = -0.50 * sm
            shp[1, 1] = -0.50 * sp
            shp[1, 2] = 0.50 * sp
            shp[1, 3] = 0.50 * sm

        shp[2, 0] = sm * tm
        shp[2, 1] = sp * tm
        shp[2, 2] = sp * tp
        shp[2, 3] = sm * tp
        return shp, xsj

    @staticmethod
    def shp2dc(ss, xl, ord, flg):
        """
        Shape function routine for cubic (16-node) elements

        Parameters:
        ss (ndarray): Gauss point
        xl (ndarray): Element coordinates
        ord (int): Order to generate
        flg (int): Form global derivatives if false

        Returns:
        shp (ndarray): Shape functions and first derivatives
        xsj (float): Jacobian determinant
        """
        xi1 = np.array([1, 2, 2, 1, 3, 4, 2, 2, 4, 3, 1, 1, 3, 4, 4, 3])
        xi2 = np.array([1, 1, 2, 2, 1, 1, 3, 4, 2, 2, 4, 3, 3, 3, 4, 4])

        shp = np.zeros((3, 16))
        xi1s9 = 1 / 9 - ss[0] * ss[0]
        xi2s9 = 1 / 9 - ss[1] * ss[1]
        xi1s2 = 1.0 - ss[0] * ss[0]
        xi2s2 = 1.0 - ss[1] * ss[1]
        n1 = [-9.0 * (1.0 - ss[0]) * xi1s9 * 0.0625,
              -9.0 * (1.0 + ss[0]) * xi1s9 * 0.0625,
              27.0 * xi1s2 * (1.0 / 3.0 - ss[0]) * 0.0625,
              27.0 * xi1s2 * (1.0 / 3.0 + ss[0]) * 0.0625]
        n2 = [-9.0 * (1.0 - ss[1]) * xi2s9 * 0.0625,
              -9.0 * (1.0 + ss[1]) * xi2s9 * 0.0625,
              27.0 * xi2s2 * (1.0 / 3.0 - ss[1]) * 0.0625,
              27.0 * xi2s2 * (1.0 / 3.0 + ss[1]) * 0.0625]
        dn1 = [(1.0 + (18.0 - 27.0 * ss[0]) * ss[0]) * 0.0625,
               (-1.0 + (18.0 + 27.0 * ss[0]) * ss[0]) * 0.0625,
               (-27.0 - (18.0 - 81.0 * ss[0]) * ss[0]) * 0.0625,
               (27.0 - (18.0 + 81.0 * ss[0]) * ss[0]) * 0.0625]
        dn2 = [(1.0 + (18.0 - 27.0 * ss[1]) * ss[1]) * 0.0625,
               (-1.0 + (18.0 + 27.0 * ss[1]) * ss[1]) * 0.0625,
               (-27.0 - (18.0 - 81.0 * ss[1]) * ss[1]) * 0.0625,
               (27.0 - (18.0 + 81.0 * ss[1]) * ss[1]) * 0.0625]

        for k in range(16):
            shp[2, k] = n1[xi1[k] - 1] * n2[xi2[k] - 1]

        if ord >= 1:
            for k in range(16):
                shp[0, k] = dn1[xi1[k] - 1] * n2[xi2[k] - 1]
                shp[1, k] = n1[xi1[k] - 1] * dn2[xi2[k] - 1]

            xds = np.zeros((2, 2))
            for j in range(2):
                for i in range(2):
                    for k in range(16):
                        xds[i, j] += xl[i, k] * shp[j, k]

            xsj = xds[0, 0] * xds[1, 1] - xds[0, 1] * xds[1, 0]
            if not flg:
                for k in range(16):
                    mn1 = (xds[1, 1] * shp[0, k] - xds[1, 0] * shp[1, k]) / xsj
                    shp[1, k] = (-xds[0, 1] * shp[0, k] + xds[0, 0] * shp[1, k]) / xsj
                    shp[0, k] = mn1
        return shp, xsj

    @staticmethod
    def shp2ds(ss, xl, ord, flg):
        """
        Shape function routine for cubic (12-node) elements

        Parameters:
        ss (ndarray): Gauss point
        xl (ndarray): Element coordinates
        ord (int): Order to generate
        flg (int): Form global derivatives if false

        Returns:
        shp (ndarray): Shape functions and first derivatives
        xsj (float): Jacobian determinant
        """
        xi1a = np.array([-1.0, 1.0, 1.0, -1.0])
        xi2a = np.array([-1.0, -1.0, 1.0, 1.0])

        shp = np.zeros((3, 12))
        xi1s2 = 1.0 - ss[0] * ss[0]
        xi2s2 = 1.0 - ss[1] * ss[1]
        n1 = [9.0 * xi1s2 * (1.0 - 3.0 * ss[0]) * 0.0625,
              9.0 * xi1s2 * (1.0 + 3.0 * ss[0]) * 0.0625]
        n2 = [9.0 * xi2s2 * (1.0 - 3.0 * ss[1]) * 0.0625,
              9.0 * xi2s2 * (1.0 + 3.0 * ss[1]) * 0.0625]
        dn1 = [(-27.0 - (18.0 - 81.0 * ss[0]) * ss[0]) * 0.0625,
               (27.0 - (18.0 + 81.0 * ss[0]) * ss[0]) * 0.0625]
        dn2 = [(-27.0 - (18.0 - 81.0 * ss[1]) * ss[1]) * 0.0625,
               (27.0 - (18.0 + 81.0 * ss[1]) * ss[1]) * 0.0625]

        for i in range(4):
            shp[0, i] = 0.25 * xi1a[i] * (1.0 + xi2a[i] * ss[1])
            shp[1, i] = 0.25 * xi2a[i] * (1.0 + xi1a[i] * ss[0])
            shp[2, i] = 0.25 * (1.0 + xi1a[i] * ss[0]) * (1.0 + xi2a[i] * ss[1])

        xi1s2 = 0.5 * (1.0 - ss[0])
        xi2s2 = 0.5 * (1.0 - ss[1])
        shp[0, 4] = dn1[0] * xi2s2
        shp[1, 4] = -n1[0] * 0.5
        shp[2, 4] = n1[0] * xi2s2
        shp[0, 5] = dn1[1] * xi2s2
        shp[1, 5] = -n1[1] * 0.5
        shp[2, 5] = n1[1] * xi2s2
        shp[0, 11] = -n2[0] * 0.5
        shp[1, 11] = dn2[0] * xi1s2
        shp[2, 11] = n2[0] * xi1s2
        shp[0, 10] = -n2[1] * 0.5
        shp[1, 10] = dn2[1] * xi1s2
        shp[2, 10] = n2[1] * xi1s2

        xi1s2 = 0.5 * (1.0 + ss[0])
        xi2s2 = 0.5 * (1.0 + ss[1])
        shp[0, 9] = dn1[0] * xi2s2
        shp[1, 9] = n1[0] * 0.5
        shp[2, 9] = n1[0] * xi2s2
        shp[0, 8] = dn1[1] * xi2s2
        shp[1, 8] = n1[1] * 0.5
        shp[2, 8] = n1[1] * xi2s2
        shp[0, 6] = n2[0] * 0.5
        shp[1, 6] = dn2[0] * xi1s2
        shp[2, 6] = n2[0] * xi1s2
        shp[0, 7] = n2[1] * 0.5
        shp[1, 7] = dn2[1] * xi1s2
        shp[2, 7] = n2[1] * xi1s2

        for i in range(3):
            shp[i, 0] -= 2.0 * (shp[i, 4] + shp[i, 11]) + shp[i, 5] + shp[i, 10]
            shp[i, 1] -= 2.0 * (shp[i, 5] + shp[i, 6]) + shp[i, 4] + shp[i, 7]
            shp[i, 2] -= 2.0 * (shp[i, 7] + shp[i, 8]) + shp[i, 6] + shp[i, 9]
            shp[i, 3] -= 2.0 * (shp[i, 9] + shp[i, 10]) + shp[i, 8] + shp[i, 11]

        if ord >= 1:
            xds = np.zeros((2, 2))
            for j in range(2):
                for i in range(2):
                    for k in range(12):
                        xds[i, j] += xl[i, k] * shp[j, k]

            xsj = xds[0, 0] * xds[1, 1] - xds[0, 1] * xds[1, 0]
            if not flg:
                for k in range(12):
                    mn1 = (xds[1, 1] * shp[0, k] - xds[1, 0] * shp[1, k]) / xsj
                    shp[1, k] = (-xds[0, 1] * shp[0, k] + xds[0, 0] * shp[1, k]) / xsj
                    shp[0, k] = mn1

        return shp, xsj

    @staticmethod
    def shap2(s, t, shp, ix, nel):
        """
        Adds quadratic functions to quadrilaterals for any non-zero mid-side or central node

        Parameters:
        s (float): Natural coordinate s
        t (float): Natural coordinate t
        shp (ndarray): Shape functions and derivatives w.r.t natural coords
        ix (ndarray): List of nodes attached to element (0 = no node)
        nel (int): Maximum number of local node on element <= 9

        Returns:
        shp (ndarray): Updated shape functions and derivatives
        """
        s2 = (1.0 - s * s) * 0.5
        t2 = (1.0 - t * t) * 0.5
        shp[0:3, 4:nel] = 0

        if ix[4] != 0:
            shp[0, 4] = -s * (1.0 - t)
            shp[1, 4] = -s2
            shp[2, 4] = s2 * (1.0 - t)

        if ix[5] != 0:
            shp[0, 5] = t2
            shp[1, 5] = -t * (1.0 + s)
            shp[2, 5] = t2 * (1.0 + s)

        if ix[6] != 0:
            shp[0, 6] = -s * (1.0 + t)
            shp[1, 6] = s2
            shp[2, 6] = s2 * (1.0 + t)

        if ix[7] != 0:
            shp[0, 7] = -t2
            shp[1, 7] = -t * (1.0 - s)
            shp[2, 7] = t2 * (1.0 - s)

        if nel > 8:
            if ix[8] != 0:
                shp[0, 8] = -4.0 * s * t2
                shp[1, 8] = -4.0 * t * s2
                shp[2, 8] = 4.0 * s2 * t2
                shp[:, 0:4] -= 0.25 * shp[:, 8].reshape(-1, 1)
                shp[:, 4:8] -= 0.5 * shp[:, 8].reshape(-1, 1)

        shp[:, 0] -= 0.5 * (shp[:, 4] + shp[:, 7])
        shp[:, 1] -= 0.5 * (shp[:, 4] + shp[:, 5])
        shp[:, 2] -= 0.5 * (shp[:, 5] + shp[:, 6])
        shp[:, 3] -= 0.5 * (shp[:, 6] + shp[:, 7])

        return shp

    @classmethod
    def interp2d_tf(cls, qid, xl, ix, ndm, nel, flg, etype):
        """
        Interpolation functions for 2-D elements

        Parameters:
        qid (int): Quadrature point
        xl (ndarray): Nodal coordinates
        ix (ndarray): Global nodal connections
        ndm (int): Mesh coordinate dimension
        nel (int): Number of element nodes
        flg (int): Global derivatives if false
        etype (int): Element type

        Returns:
        shp (ndarray): Shape functions
        jac (float): Jacobian determinant
        """
        sg2 = tf.constant(cls.sg2, dtype=tf.float64)
        shp, jac = PreProcessing.shp2d_tf(sg2[0:2, qid], xl, ndm, nel, ix, flg)
        jac *= sg2[2, qid]
        return shp, jac

    @staticmethod
    def shp2d_tf(ss, xl, ndm, nel, ix, flg):
        """
        Shape functions for quadrilateral elements

        Parameters:
        ss (ndarray): Natural coordinates for result point
        xl (ndarray): Nodal coordinates for result element
        ndm (int): Spatial dimension of mesh
        nel (int): Number of nodes on element
        ix (ndarray): Nodes attached to element
        flg (int): Compute global x/y derivatives if false, else derivatives w.r.t. natural coords

        Returns:
        shp (ndarray): Shape functions and derivatives at point
        xsj (float): Jacobian determinant at point
        """
        '''s = np.array([-0.5, 0.5, 0.5, -0.5])
        t = np.array([-0.5, -0.5, 0.5, 0.5])'''

        shp, xsj = PreProcessing.shapef_tf(ss, xl, flg)

        return shp, xsj

    @staticmethod
    def shapef_tf(s, xl, flg):
        """
        Quadrilateral shape functions

        Parameters:
        s (ndarray): Natural coordinates of point
        xl (ndarray): Nodal coordinates for element
        flg (int): Compute global derivatives if false, else derivatives w.r.t. natural coords

        Returns:
        shp (ndarray): Shape functions and derivatives at point
        xsj (float): Jacobian determinant at point
        """
        sh = 0.5 * s[0]
        th = 0.5 * s[1]
        sp = 0.5 + sh
        tp = 0.5 + th
        sm = 0.5 - sh
        tm = 0.5 - th
        xo = xl[0, 0] - xl[0, 1] + xl[0, 2] - xl[0, 3]
        xs = -xl[0, 0] + xl[0, 1] + xl[0, 2] - xl[0, 3] + xo * s[1]
        xt = -xl[0, 0] - xl[0, 1] + xl[0, 2] + xl[0, 3] + xo * s[0]
        yo = xl[1, 0] - xl[1, 1] + xl[1, 2] - xl[1, 3]
        ys = -xl[1, 0] + xl[1, 1] + xl[1, 2] - xl[1, 3] + yo * s[1]
        yt = -xl[1, 0] - xl[1, 1] + xl[1, 2] + xl[1, 3] + yo * s[0]
        xsj1 = xs * yt - xt * ys
        xsj = 0.0625 * xsj1

        xsj1 = 1.0 / xsj1 if xsj1 != 0.0 else tf.constant(1.0, dtype=tf.float64)
        xs = (xs + xs) * xsj1
        xt = (xt + xt) * xsj1
        ys = (ys + ys) * xsj1
        yt = (yt + yt) * xsj1
        ytm = yt * tm
        ysm = ys * sm
        ytp = yt * tp
        ysp = ys * sp
        xtm = xt * tm
        xsm = xs * sm
        xtp = xt * tp
        xsp = xs * sp
        shp = tf.zeros((3, 4), dtype=tf.float64)
        '''shp[0, 0] = -ytm + ysm
        shp[0, 1] = ytm + ysp
        shp[0, 2] = ytp - ysp
        shp[0, 3] = -ytp - ysm
        shp[1, 0] = xtm - xsm
        shp[1, 1] = -xtm - xsp
        shp[1, 2] = -xtp + xsp
        shp[1, 3] = xtp + xsm

        shp[2, 0] = sm * tm
        shp[2, 1] = sp * tm
        shp[2, 2] = sp * tp
        shp[2, 3] = sm * tp'''

        indices = tf.constant([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0],
                               [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]], dtype=tf.int64)
        updates = tf.stack([-ytm + ysm, ytm + ysp, ytp - ysp, -ytp - ysm, xtm - xsm,
                            -xtm - xsp, -xtp + xsp, xtp + xsm, sm * tm, sp * tm, sp * tp, sm * tp])
        shp = tf.tensor_scatter_nd_update(shp, indices, updates)
        return shp, xsj