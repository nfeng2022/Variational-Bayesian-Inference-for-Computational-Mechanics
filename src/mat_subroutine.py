import numpy as np
import fem_preprocess as fp


class MaterialSub:

    @classmethod
    def solid_2d_lib(cls, ul, dul, xl, ix, ndm, nel, isw, part_id, step_id, ele_id):
        model_data = fp.PreProcessing.model_data
        sec_id = model_data['part'][part_id-1]['sec_id']
        eform = model_data['section'][sec_id-1]['eform']

        if eform == 1:
            return cls.solid_2d(ul, dul, xl, ix, ndm, nel, isw, part_id, step_id, ele_id)
        elif eform == 99:
            # Handle case 99 if needed
            pass
        else:
            raise ValueError(f"Unsupported eform value: {eform}")

    @classmethod
    def solid_2d(cls, ul, dul, xl, ix, ndm, nel, isw, part_id, step_id, ele_id):
        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        sec_id = model_data['part'][part_id-1]['sec_id']
        mat_id = model_data['part'][part_id-1]['mat_id']

        etype = model_data['section'][sec_id-1]['etype']
        stype = model_data['section'][sec_id-1]['stype']
        int_pts = model_data['section'][sec_id-1]['intp']
        thk = model_data['section'][sec_id-1]['thk']
        body = model_data['part'][part_id-1]['body']

        sg2, lint = fp.PreProcessing.quadr2d(int_pts, nel, 0)

        p = np.zeros((2 * nel, 1))
        kt = np.zeros((nel * 2, nel * 2))
        me = np.zeros((nel * 2, nel * 2)) if model_data['solution_control']['dynamics'] == 1 else []

        temp_strain_data = np.zeros((6, lint))
        temp_stress_data = np.zeros((6, lint))
        for ipt in range(lint):
            shp, jac = fp.PreProcessing.interp2d(ipt, xl, ix, ndm, nel, 0, etype)
            xref, xcur, eps = cls.strain_2d(xl, ul, dul, shp)

            if stype == 2:  # plane strain
                eps[2, :] = [0, 0, 0]

            sig, Ct, eps33 = cls.mat_driver_2d3d(eps, [], stype, mat_id, step_id, ele_id, ipt)

            if stype == 1:  # plane stress
                eps[2, 0] = eps33

            # Gat dvol
            dvol = 0
            if stype == 1 or stype == 2:  # Plane stress/strain
                Ct = Ct[[0, 1, 3], :][:, [0, 1, 3]]
                dvol = thk*jac
            elif stype == 3:  # Axisymmetric without torsion
                Ct = Ct[0:4, 0:4]
                dvol = 2.0 * np.pi * xref[0, 0] * jac
            elif stype == 4:  # Axisymmetric with torsion
                dvol = 2.0 * np.pi * xref[0, 0] * jac

            if isw == 1 or isw == 2:
                Bm, Nm = cls.calculate_Bm_Nm(shp, nel, stype, xref)

                p = cls.residual_2d(p, sig, body, dvol, stype, Bm, Nm)

                if isw == 2:
                    kt += dvol * (Bm.T @ Ct @ Bm)

                if model_data['solution_control']['dynamics'] == 1:
                    me += dvol * (Nm.T @ model_data['material'][mat_id]['den'] @ Nm)

            temp_strain_data[:, ipt] = eps[:, 0]
            temp_stress_data[:, ipt] = sig[:, 0]

        # Diagonalize the mass matrix (if applicable)
        out_data['ele_strain'][:, :, ele_id-1, step_id-1] = temp_strain_data
        out_data['ele_stress'][:, :, ele_id-1, step_id-1] = temp_stress_data

        return p, kt, me

    @classmethod
    def strain_2d(cls, xl, ul, dul, shp):
        eps = np.zeros((6, 3))

        ul = ul.toarray()
        dul = dul.toarray()

        xx = np.dot(xl[0, :], shp[2, :])
        yy = np.dot(xl[1, :], shp[2, :])

        uu = np.dot(ul[0, :], shp[2, :])
        vv = np.dot(ul[1, :], shp[2, :])

        eps[0, 0] = np.dot(shp[0, :], ul[0, :])
        eps[1, 0] = np.dot(shp[1, :], ul[1, :])
        eps[3, 0] = np.dot(shp[0, :], ul[1, :]) + np.dot(shp[1, :], ul[0, :])
        eps[0, 2] = np.dot(shp[0, :], dul[0, :])
        eps[1, 2] = np.dot(shp[1, :], dul[1, :])
        eps[3, 2] = np.dot(shp[0, :], dul[1, :]) + np.dot(shp[1, :], dul[0, :])

        xref = np.array([xx, yy, 0.0])
        xcur = np.array([xx + uu, yy + vv, 0.0])

        eps[0:6, 1] = eps[0:6, 0] - eps[0:6, 2]

        return xref, xcur, eps

    @classmethod
    def residual_2d(cls, p, sig, body, dvol, stype, Bm, Nm):

        if stype == 1 or stype == 2:  # Plane stress/strain
            p += dvol * (Bm.T @ sig[[0, 1, 3]]) - dvol * (Nm.T @ body[0:2])
        elif stype == 3:  # Axisymmetric without torsion
            p += dvol * (Bm.T @ sig[0:4]) - dvol * (Nm.T @ body[0:2])
        elif stype == 4:  # Axisymmetric with torsion
            p += dvol * (Bm.T @ sig) - dvol * (Nm.T @ body[0:3])

        return p

    @classmethod
    def calculate_Bm_Nm(cls, shp, nel, stype, xref):
        if stype == 1 or stype == 2:  # Plane stress/strain
            Bm = np.zeros((3, nel * 2))
            Nm = np.zeros((2, nel * 2))
            j = 0
            for i in range(nel):
                Bm[0, j] = shp[0, i]
                Bm[1, j + 1] = shp[1, i]
                Bm[2, j] = shp[1, i]
                Bm[2, j + 1] = shp[0, i]
                Nm[0, j] = shp[2, i]
                Nm[1, j + 1] = shp[2, i]
                j += 2
        elif stype == 3:  # Axisymmetric without torsion
            Bm = np.zeros((4, nel * 2))
            Nm = np.zeros((2, nel * 2))
            # Define or calculate Bm and Nm for this case
            j = 0
            for i in range(nel):
                Bm[0, j] = shp[0, i]
                Bm[1, j + 1] = shp[1, i]
                Bm[2, j] = shp[2, i]/xref[0]
                Bm[3, j] = shp[1, i]
                Bm[3, j + 1] = shp[0, i]
                Nm[0, j] = shp[2, i]
                Nm[1, j + 1] = shp[2, i]
                j += 2
        elif stype == 4:  # Axisymmetric with torsion
            Bm = np.zeros((5, nel * 2))
            Nm = np.zeros((2, nel * 2))
            # Define or calculate Bm and Nm for this case
            j = 0
            for i in range(nel):
                Bm[0, j] = shp[0, i]
                Bm[1, j + 1] = shp[1, i]
                Bm[2, j] = shp[2, i] / xref[0]
                Bm[3, j] = shp[1, i]
                Bm[3, j + 1] = shp[0, i]
                Bm[4, j + 2] = shp[1, i]
                Bm[5, j + 2] = shp[0, i] - shp[2, i]/xref[0]
                Nm[0, j] = shp[2, i]
                Nm[1, j + 1] = shp[2, i]
                Nm[2, j + 2] = shp[2, i]
                j += 2

        return Bm, Nm

    @classmethod
    def calculate_dvol(cls, stype, jac, xref):
        dvol = 0.0

        if stype == 1 or stype == 2:  # Plane stress/strain
            dvol = jac
        elif stype == 3:  # Axisymmetric without torsion
            dvol = 2.0 * np.pi * xref[0, 0] * jac
        elif stype == 4:  # Axisymmetric with torsion
            dvol = 2.0 * np.pi * xref[0, 0] * jac

        return dvol

    @classmethod
    def mat_driver_2d3d(cls, eps, isrt, stype, mat_id, step_id, ele_id, intp_id):
        model_data = fp.PreProcessing.model_data
        out_data = fp.PreProcessing.out_data
        mat_type = model_data['material'][mat_id-1]['type']
        eps33 = None

        # Plane Stress
        if stype == 1 and mat_type not in [1, 2]:
            eps33 = 0
            eps[2] = eps33

        # Plane stress iterations
        isrt = 0
        ps_conv_flag = 0
        ps_conv_flagcr = 0
        ps_iter = 1
        ps_iter_max = 100  # Define a maximum number of iterations

        while ps_conv_flagcr == 0:
            if ps_conv_flag == 1:
                ps_conv_flagcr = 1

            if mat_type == 1:  # Elastic Isotropic (Plane stress OK)
                sig, Ct, eps33 = cls.isotropic_elasticity(eps, stype, mat_id, ele_id)
            elif mat_type == 99:  # User Defined umat
                hist_data = {
                    'pstrain': out_data['step'][step_id - 1]['ele_pstrain'][ele_id]['data'][:, intp_id],
                    'hsv': out_data['step'][step_id - 1]['ele_hsv'][ele_id]['data'][:, intp_id]
                }
                # Update history variables (current step)

            else:
                raise ValueError('Error in mat_driver_2d3d: Illegal mat_type')

            # Plane Stress
            if mat_type not in [1, 2]:  # Plane stress is incorporated
                break

            if stype == 1:
                if not ps_conv_flag:
                    ps_iter += 1
                    eps33, ps_conv_flag = cls.plane_stress(eps, Ct[2, 2], eps33)
                if ps_conv_flagcr == 1 or ps_iter > ps_iter_max:
                    Ct = cls.plane_stress_Ct(Ct)
                if ps_iter > ps_iter_max:
                    raise ValueError('No convergence in plane stress iterations ...')
            else:
                ps_conv_flag = 1

        return sig, Ct, eps33

    @classmethod
    def plane_stress(cls, sig, Ct, eps33):
        tol = 1.0e-10
        dsig = sig[2]
        if Ct == 0 or dsig == 0:
            ps_conv_flag = 1
            return eps33, ps_conv_flag
        ct_inv = 1.0 / Ct
        deps33 = -dsig * ct_inv
        eps33 += deps33
        if abs(deps33) < tol * abs(eps33):
            ps_conv_flag = 1
        else:
            ps_conv_flag = 0
        return eps33, ps_conv_flag

    @classmethod
    def plane_stress_Ct(cls, Ct):
        ct_inv = 1.0 / Ct[2, 2]
        Ct[0, 0] -= Ct[0, 2] * ct_inv * Ct[2, 0]
        Ct[1, 0] -= Ct[1, 2] * ct_inv * Ct[2, 0]
        Ct[3, 0] -= Ct[3, 2] * ct_inv * Ct[2, 0]
        Ct[0, 1] -= Ct[0, 2] * ct_inv * Ct[2, 1]
        Ct[1, 1] -= Ct[1, 2] * ct_inv * Ct[2, 1]
        Ct[3, 1] -= Ct[3, 2] * ct_inv * Ct[2, 1]
        Ct[0, 3] -= Ct[0, 2] * ct_inv * Ct[2, 3]
        Ct[1, 3] -= Ct[1, 2] * ct_inv * Ct[2, 3]
        Ct[3, 3] -= Ct[3, 2] * ct_inv * Ct[2, 3]
        Ct[0, 2] = 0.0
        Ct[1, 2] = 0.0
        Ct[3, 2] = 0.0
        Ct[2, 0] = 0.0
        Ct[2, 1] = 0.0
        Ct[2, 3] = 0.0
        Ct[2, 2] = 0.0
        return Ct

    @classmethod
    def isotropic_elasticity(cls, eps, stype, mat_id, ele_id):
        model_data = fp.PreProcessing.model_data
        E = model_data['material'][mat_id-1]['E']
        v = model_data['material'][mat_id-1]['v']
        eps33 = None
        sig = np.zeros((6, 1))
        Ct = np.zeros((6, 6))

        if stype == 1:  # Plane stress
            Ce = np.array([[1, v, 0],
                           [v, 1, 0],
                           [0, 0, (1 - v) / 2]])
            Ce = E / (1 - v ** 2) * Ce
            sig[[0, 1, 3], 0] = Ce @ eps[[0, 1, 3], 0]
            eps33 = -v / (1 - v) * (eps[0] + eps[1])
            Ct[np.ix_([0, 1, 3], [0, 1, 3])] = Ce
        elif stype == 2:  # Plane strain
            l = v * E / ((1 + v) * (1 - 2 * v))
            mu = 0.5 * E / (1 + v)
            Ce = np.array([[l + 2 * mu, l, l, 0],
                           [l, l + 2 * mu, l, 0],
                           [l, l, l + 2 * mu, 0],
                           [0, 0, 0, mu]])
            sig[0:4, 0] = Ce @ eps[0:4, 0]
            Ct[np.ix_([0, 1, 3], [0, 1, 3])] = Ce[np.ix_([0, 1, 3], [0, 1, 3])]
        elif stype == 3:  # Axisymmetric strain
            # Implement axisymmetric strain case if needed
            pass
        elif stype == 4:  # 3-D case
            l = v * E / ((1 + v) * (1 - 2 * v))
            mu = 0.5 * E / (1 + v)
            Ct = np.array([[l + 2 * mu, l, l, 0, 0, 0],
                           [l, l + 2 * mu, l, 0, 0, 0],
                           [l, l, l + 2 * mu, 0, 0, 0],
                           [0, 0, 0, mu, 0, 0],
                           [0, 0, 0, 0, mu, 0],
                           [0, 0, 0, 0, 0, mu]])
            sig[:, 0] = Ct @ eps[0:6, 0]
        return sig, Ct, eps33
