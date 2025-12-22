import matplotlib.pyplot as plt
import numpy as np
import fem_preprocess as fp
import scipy.io as sio
import xml.etree.ElementTree as ET
import tensorflow as tf
import h5py as h5


class PostProcessing:

    @staticmethod
    def plot_2d_mesh(step_id, mf, flag):
        # inputs: step_id
        # mf: magnification factor
        # flag: 1-> deformed only; 2-> both; flag: 3-> intial only
        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        nele = model_data['mesh_info']['nele']
        nel = model_data['mesh_info']['max_ele_node']

        fh = plt.figure()
        fh.clf()
        ha = fh.add_subplot(111, aspect='auto')
        ha.set_xticks([])
        ha.set_yticks([])
        ha.set_xlim(auto=True)
        ha.set_ylim(auto=True)
        ha.grid(False)
        ha.set_facecolor('w')

        if flag == 2 or flag == 3:
            for i in range(nele):
                X = model_data['element'][i]['coord'][:nel, 1]
                Y = model_data['element'][i]['coord'][:nel, 2]
                hp = plt.Polygon(np.vstack((X, Y)).T, edgecolor='k', facecolor='w', alpha=0.6)
                ha.add_patch(hp)

        if flag == 1 or flag == 2:
            for i in range(nele):
                X = model_data['element'][i]['coord'][:nel, 1]
                Y = model_data['element'][i]['coord'][:nel, 2]
                nid = model_data['element'][i]['conn'][:nel]
                ux = out_data['step'][step_id - 1]['nodal_disp'][0, nid - 1]
                uy = out_data['step'][step_id - 1]['nodal_disp'][1, nid - 1]
                x = X + mf * ux
                y = Y + mf * uy
                hp = plt.Polygon(np.vstack((x, y)).T, edgecolor='b', facecolor='w')
                ha.add_patch(hp)

        plt.axis('equal')
        plt.show()

    @staticmethod
    def check_keys(dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
                dict[key] = PostProcessing.todict(dict[key])
        return dict

    @staticmethod
    def todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                dict[strg] = PostProcessing.todict(elem)
            else:
                dict[strg] = elem
        return dict

    @staticmethod
    def loadmat(filename):
        """
        this function should be called instead of direct scipy.io .loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return PostProcessing.check_keys(data)

    @staticmethod
    def create_xdmf_file(xdfm_filename, h5_filename):
        # Initialization
        n_steps = len(fp.PreProcessing.out_data['step'])
        model_data = fp.PreProcessing.model_data
        nnodes = model_data['mesh_info']['nnodes']
        nele = model_data['mesh_info']['nele']
        nel = model_data['mesh_info']['max_ele_node']

        # Create the root element
        root = ET.Element("XDFM", type="MultiBlockDataSet", version="0.1", byte_order="LittleEndian")
        dataset = ET.SubElement(root, "MultiBlockDataSet",
                                Blocks=str(n_steps),
                                WholeExtent="0 0 0 0 0 0")

        # Define Blocks for each time step
        for t in range(n_steps):
            block = ET.SubElement(dataset, "DataSet",
                                  part="Block",
                                  index=str(t),
                                  type="PolyData")

            # Write points
            points = ET.SubElement(block, "Points")
            ET.SubElement(points, "DataArray",
                          type="Float64",
                          Name="points",
                          NumberOfComponents="2",
                          format="ascii").text = str(h5_filename)+"/step" + str(t) + "/nodes" + str(t) + "_coord"

            # Write polygons
            polygons = ET.SubElement(block, "Polygons")
            ET.SubElement(polygons, "DataArray",
                          type="Int32",
                          Name="polygons",
                          NumberOfComponents=str(nel),
                          format="ascii").text = str(h5_filename)+"/conn"

        tree = ET.ElementTree(root)
        with open(xdfm_filename, "wb") as fh:
            tree.write(fh)

    @staticmethod
    def xdmf_h5data_save(filename):
        # Write Datasets
        with h5.File(filename, "w") as f:
            # Get Date
            model_data = fp.PreProcessing.model_data
            conn = model_data['dof_info']['IEN']-1
            nodes1_coord = model_data['mesh_info']['coord'][:, 1:]
            das1 = f.create_dataset('conn', conn.shape, dtype='i')
            das1[...] = conn
            das2 = f.create_dataset('step0/nodes0_coord', nodes1_coord.shape, dtype='f')
            das2[...] = nodes1_coord

            out_data = fp.PreProcessing.out_data
            nele = model_data['mesh_info']['nele']
            nel = model_data['mesh_info']['max_ele_node']
            nstep = len(fp.PreProcessing.out_data['step'])
            for i in range(nstep - 1):
                nodes2_coord = np.zeros(nodes1_coord.shape)
                for e in range(nele):
                    X = model_data['element'][e]['coord'][:nel, 1]
                    Y = model_data['element'][e]['coord'][:nel, 2]
                    nid = model_data['element'][e]['conn'][:nel]
                    ux = out_data['step'][i + 1]['nodal_disp'][0, nid - 1]
                    uy = out_data['step'][i + 1]['nodal_disp'][1, nid - 1]
                    nodes2_coord[nid - 1, 0] = X + ux
                    nodes2_coord[nid - 1, 1] = Y + uy
                step_str = 'step' + str(i + 1) + '/nodes' + str(i + 1) + '_coord'
                das_step = f.create_dataset(step_str, nodes2_coord.shape, dtype='f')
                das_step[...] = nodes2_coord

    @staticmethod
    def von_mises_stress(step_id, ele_id, nipt_id):
        # Load Data
        ele_stress = fp.PreProcessing.out_data['ele_stress'][:, :, ele_id-1, step_id-1]
        ele_stress = ele_stress[:, nipt_id-1]
        Pdevs = fp.PreProcessing.Pdevs[[0, 4, 8, 3, 7, 2], :][:, [0, 4, 8, 3, 7, 2]]
        von_mises_stress = np.sqrt(0.5*np.sum((Pdevs @ ele_stress)**2, axis=0))
        return von_mises_stress

    @staticmethod
    def von_mises_stress_tf(step_id, ele_id, nipt_id):
        # Load Data
        ele_stress = fp.PreProcessing.out_data['ele_stress'][:, :, ele_id - 1, step_id - 1]
        indices = tf.reshape(tf.constant(nipt_id-1, dtype=tf.int64), (-1, 1))
        ele_stress = tf.transpose(tf.gather_nd(tf.transpose(ele_stress), indices))

        # Pdevs = fp.PreProcessing.Pdevs[[0, 4, 8, 3, 7, 2], :][:, [0, 4, 8, 3, 7, 2]]
        Pdevs_full = tf.constant(fp.PreProcessing.Pdevs, dtype=tf.float64)
        indices = tf.reshape(tf.constant([0, 4, 8, 3, 7, 2], dtype=tf.int64), (-1, 1))
        Pdevs = tf.transpose(tf.gather_nd(Pdevs_full, indices))
        Pdevs = tf.transpose(tf.gather_nd(Pdevs, indices))
        von_mises_stress = tf.sqrt(0.5 * tf.reduce_sum((Pdevs @ ele_stress) ** 2, axis=0))
        return von_mises_stress


