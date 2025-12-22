# Imports
import numpy as np


def model_property_cards_fun(disp_control_flag):
    # Element type
    ele_type = '2D'  # 2D or 3D

    # Part Cards
    part = [{'id': 1, 'sec_id': 1, 'mat_id': 1, 'body': np.array([[0], [0], [0]])}]

    '''Material Cards Reference
    E = 9 * k * mu / (3 * k + mu);
    E = 2 * mu * (1 + v);
    mu = E / (2 * (1 + v));
    k = E / (3 * (1 - 2 * v));
    v = 0.5 * (3 * k - 2 * mu) / (3 * k + mu);'''

    # Material-1: HE-1 (stiff) by neo-Hookean
    '''stype = 1: plane stress; 2: plane strain; 3: Axisymmetric; 4: Axisymmetric torsion
    etype = 1: quad; 2: triangular
    eform = 1: d - standard; 2: A - assembly(d - standard); 3: F - bar element
    eform = 4: EAS
    eform = 5: 3 - EAS; eform = 6: 3 - EAS - F0; eform = 7: MEAN - DIL'''
    material = [{'id': 1, 'type': 1, 'E': 20.0, 'v': 0.3, 'hsv': 0, 'in_flag': 0}]

    # 2D section
    section = [{'id': 1, 'type': 1, 'int_scheme': 'guass', 'intp': 2, 'thk': 10, 'etype': 1,
                'stype': 2, 'eform': 1, 'estorage': 0, 'nalpha': 0, 'mode_type': 0}]

    # Equality Constraints
    eqcgroup = [{'nid': [], 'dof': []}]
    solution_control = {'eqconstraints': 0, 'eqcgroup': eqcgroup}

    # Output control
    # Print_flag: 1 -> print analysis progress; 0 -> no analysis hints
    solution_control.update({'output_flag': 1, 'strain_energy_flag': 0, 'print_flag': 0})

    # Solver options
    # Solver: 1-> linear | 2-> nonlinear
    # NR Solver Type: 1-> standard global | 2-> cubic global | 3-> adaptive
    # Cubic NR solver type: 1-> Midpoint method | 2-> Harmonic mean | 3-> Arithmetic mean
    solution_control.update({'solver': 1, 'newt_raphson_solver_type': 1, 'mastan2': 0,
                             'display_iter_flag': 1, 'dynamics': 0})

    # Nonlinear solver options
    solution_control.update({'scheme_type': 2, 'large_disp_flag': 0})

    # Load Control Parameters
    load_control = {'numsteps': 1, 'ini_incr': 0.05, 'min_incr': 1.0E-4, 'max_incr': 0.1,
                    'iter_d': 5, 'iter_buffer': 1, 'incr_factor': 1.5,
                    'macro_ini_incr': 0.05, 'macro_min_incr': 1.0E-4, 'macro_max_incr': 0.1}
    solution_control.update({'load_control': load_control})

    # NR parameters
    # tol_Rforce: 1 -> Residual norm; 0 -> Energy norm
    nr_param = {'max_iter': 10, 'tol_cr': 1.0e-10, 'tol_Rforce': 0, 'nr_tol_switch': 1.0}
    solution_control.update({'nr_param': nr_param})

    # Line search parameters
    line_search = {'line_search_flag': 0, 'type': 1, 'max_iter': 20, 'min_lambda': 0.1}
    solution_control.update({'line_search': line_search})

    # Arc Length Parameters
    arclen_control = {'scheme_type': 1, 'max_disp': 7, 'cdof': 2, 'cnode': 221,
                      'ini_incr': 1, 'min_incr': 1.0e-5, 'max_incr': 1000,
                      'max_steps': 500, 'max_iter': 20, 'tol_cr': 1.0e-12, 'iter_d': 8}
    solution_control.update({'arclen_control': arclen_control})

    # Topology Optimization flags
    solution_control.update({'topo_opt_flag': 0, 'fast_sol_flag': 1, 'topo_solver': 1,
                             'filter_radius': 0.0375, 'filter_ratio': 0, 'filter_type': 'hat',
                             'finverter_flag': 0})

    return material, section, part, solution_control, ele_type
