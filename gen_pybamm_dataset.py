"""
gen_pybamm_datasets.py

Generate simulation data from the PyBaMM PDE LMB cell model.

2024.10.28 | Created | Wesley Hileman <whileman@uccs.edu>
"""

import os
import pickle
import time

import util


def get_save_data(solution):
    return {
        'time': solution['time [s]'](),
        'iapp': solution['iapp [A]'](),
        'soc_pct': solution['soc [%]'](),
        'pos_thetas_avg': solution['pos_thetas_avg'](),
        'thetae0': solution['thetae'](x=0),
        'thetae2': solution['thetae'](x=2),
        'phie2': solution['phie [V]'](x=2),
        'thetass1': solution['thetass'](x=1),
        'thetass2': solution['thetass'](x=2),
        'Uss2': solution['Uss [V]'](x=2),
        'phise2': solution['phise [V]'](x=2),
        'etas2': solution['etas [V]'](x=2),
        'ifdl2': solution['ifdl [A]'](x=2),
        'if2': solution['pos_if [A]'](x=2),
        'neg_phis': solution['neg_phis [V]'](),
        'vcell': solution['vcell [V]'](),
    }


if __name__ == '__main__':
    # Load cell model from file.
    cell = util.load_mat_cell_model(os.path.join('MATLAB', 'CELLPARAMS', 'cellParams.mat'))

    # Define constants.
    TdegC = 25     # cell temperature [degC]
    ts = 1.0       # sampling interval [s]

    start = time.time()

    # TRAINING DATA ----------------------------------------------------------------------------------------------------

    #
    # Constant-current dis/charge.
    #

    C_rates = [-2, -1.5, -1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2]
    sim_cc = []
    for C_rate in C_rates:
        if C_rate < 0:
            # Charge the cell.
            soc0 = 0
            socf = 100
        else:
            # Discharge the cell.
            soc0 = 100
            socf = 0

        sim_cc.append({
            'TdegC': TdegC,
            'ts': ts,
            'soc0': soc0,
            'socf': socf,
            'C_rate': C_rate,
            'pybamm': get_save_data(util.sim_cc(
                cell,
                i_galv=C_rate*cell.cst.QAh,
                soc0=soc0, socf=socf,
                TdegC=TdegC, ts=ts,
            ).solution)
        })

    #
    # GITT.
    #

    t_galv_rest_vect = [(60*10, 60*20), (60*5, 60*10), (60*2, 60*5)]
    C_rates = [-1.0, 1.0]
    sim_gitt = []
    for t_galv, t_rest in t_galv_rest_vect:
        for C_rate in C_rates:
            if C_rate < 0:
                # Charge the cell.
                soc0 = 0
                socf = 100
            else:
                # Discharge the cell.
                soc0 = 100
                socf = 0

            sim_gitt.append({
                'TdegC': TdegC,
                'ts': ts,
                'soc0': soc0,
                'socf': socf,
                't_galv': t_galv,
                't_rest': t_rest,
                'C_rate': C_rate,
                'pybamm': get_save_data(util.sim_gitt(
                    cell,
                    t_galv=t_galv, t_rest=t_rest,
                    soc0=soc0, socf=socf,
                    i_galv=C_rate*cell.cst.QAh,
                    TdegC=TdegC, ts=ts,
                ).solution)
            })

    #
    # Drive cycles.
    #

    cycle_names = ['UDDS', 'LA92']
    soc0 = 90
    socf = 5
    sim_drive = []
    for cycle_name in cycle_names:
        sim_drive.append({
            'TdegC': TdegC,
            'ts': ts,
            'cycle_name': cycle_name,
            'soc0': soc0,
            'socf': socf,
            'pybamm': get_save_data(util.sim_drive(
                cell, cycle_name,
                soc0=soc0, socf=socf,
                TdegC=TdegC, ts=ts,
            ).solution)
        })

    train = {
        'cc': sim_cc,
        'gitt': sim_gitt,
        'drive': sim_drive,
    }

    # TESTING DATA -----------------------------------------------------------------------------------------------------

    #
    # Constant-current dis/charge.
    #

    C_rates = [-1.8, -0.5, 0.5, 1.8]
    sim_cc = []
    for C_rate in C_rates:
        if C_rate < 0:
            # Charge the cell.
            soc0 = 0
            socf = 100
        else:
            # Discharge the cell.
            soc0 = 100
            socf = 0

        sim_cc.append({
            'TdegC': TdegC,
            'ts': ts,
            'soc0': soc0,
            'socf': socf,
            'C_rate': C_rate,
            'pybamm': get_save_data(util.sim_cc(
                cell,
                i_galv=C_rate * cell.cst.QAh,
                soc0=soc0, socf=socf,
                TdegC=TdegC, ts=ts,
            ).solution)
        })

    #
    # GITT.
    #

    t_galv_rest_vect = [(60 * 7, 60 * 14)]
    C_rates = [-1.5, 1.5]
    sim_gitt = []
    for t_galv, t_rest in t_galv_rest_vect:
        for C_rate in C_rates:
            if C_rate < 0:
                # Charge the cell.
                soc0 = 0
                socf = 100
            else:
                # Discharge the cell.
                soc0 = 100
                socf = 0

            sim_gitt.append({
                'TdegC': TdegC,
                'ts': ts,
                'soc0': soc0,
                'socf': socf,
                't_galv': t_galv,
                't_rest': t_rest,
                'C_rate': C_rate,
                'pybamm': get_save_data(util.sim_gitt(
                    cell,
                    t_galv=t_galv, t_rest=t_rest,
                    soc0=soc0, socf=socf,
                    i_galv=C_rate * cell.cst.QAh,
                    TdegC=TdegC, ts=ts,
                ).solution)
            })

    #
    # Drive cycles.
    #

    cycle_names = ['US06']
    soc0 = 90
    socf = 5
    sim_drive = []
    for cycle_name in cycle_names:
        sim_drive.append({
            'TdegC': TdegC,
            'ts': ts,
            'cycle_name': cycle_name,
            'soc0': soc0,
            'socf': socf,
            'pybamm': get_save_data(util.sim_drive(
                cell, cycle_name,
                soc0=soc0, socf=socf,
                TdegC=TdegC, ts=ts,
            ).solution)
        })

    test = {
        'cc': sim_cc,
        'gitt': sim_gitt,
        'drive': sim_drive,
    }

    # Save results to file.
    filename = os.path.join("datasets", f"PyBaMM_{TdegC:d}degC.pickle")
    with open(filename, "wb") as f:
        pickle.dump({
            'cell': cell,
            'TdegC': TdegC,
            'ts': ts,
            'datasets': {
                'train': train,
                'test': test,
            },
        }, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    duration = end - start
    print('Simulation finished in ', duration/60, ' minutes')
