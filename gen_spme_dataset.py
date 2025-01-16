"""
gen_spm_datasets.py

Augment PyBaMM training/testing data with SPM simulation results.

2024.10.28 | Created | Wesley Hileman <whileman@uccs.edu>
"""

import os
import pickle

import pandas as pd
import scipy
from matplotlib import pyplot as plt

import util
from spme import SPMe

if __name__ == '__main__':
    # Load PyBaMM simulation data from file.
    with open(os.path.join('datasets', 'PyBaMM_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    # Simulate SPMe.
    cell = data['cell']
    spme = SPMe(cell, ns=10, ne_eff=5, ne_pos=10, ts=data['ts'], TdegC=data['TdegC'])
    for dataset_key, dataset in data['datasets'].items():  # train, test
        for series_key, series in dataset.items():    # cc, gitt, drive
            for sim_data in series:
                soc0Pct = sim_data['soc0']
                iapp = sim_data['pybamm']['iapp']
                # Place simulation results in another key in the data dictionary.
                sim_data['spme'] = spme.run(iapp, soc0Pct)

                # Save iapp profile to file.
                TdegC = sim_data['TdegC']
                time = sim_data['pybamm']['time']
                spec = util.sim_specs(series_key, sim_data)
                df = pd.DataFrame({'time': time, 'iapp': iapp, 'soc0Pct': soc0Pct, 'TdegC': TdegC,})
                df.to_excel(os.path.join('iapp_profiles', f"{dataset_key}_{series_key}_{spec}.xlsx"))


    # Store SPMe instance in data dictionary.
    data['spme'] = spme

    # Save data dictionary to file.
    with open(os.path.join('datasets', 'PyBaMM_SPMe_25degC.pickle'), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # Plot some comparisons between PyBaMM and SPMe simulations.
    fom_sim = data['datasets']['train']['cc'][-1]['pybamm']
    spme_sim = data['datasets']['train']['cc'][-1]['spme']
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['neg_phis'])
    # plt.plot(fom_sim['time'], spme_sim.neg_phis)
    # plt.title(r'$\phi_\mathrm{s}^\mathrm{n}$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['phise2'])
    # plt.plot(fom_sim['time'], spme_sim.pos_phise)
    # plt.title(r'$\phi_\mathrm{se}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['phie2'])
    # plt.plot(fom_sim['time'], spme_sim.phie2)
    # plt.title(r'$\phi_\mathrm{e}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['Uss2'])
    # plt.plot(fom_sim['time'], spme_sim.pos_Uss)
    # plt.title(r'$U_\mathrm{ss}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['thetass2'])
    # plt.plot(fom_sim['time'], spme_sim.thetass)
    # plt.title(r'$\theta_\mathrm{ss}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['etas2'])
    # plt.plot(fom_sim['time'], spme_sim.pos_etas)
    # plt.title(r'$\eta_\mathrm{s}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['ifdl2'])
    # plt.plot(fom_sim['time'], -spme_sim.iapp)
    # plt.title(r'$\i_\mathrm{f+dl}(\tilde{x}=2)$')
    # plt.subplots()
    # plt.plot(fom_sim['time'], fom_sim['pos_thetas_avg'])
    # plt.plot(fom_sim['time'], spme_sim.thetas_avg)
    # plt.title(r'$\theta_\mathrm{s,avg}$')
    fix, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(fom_sim['time'], fom_sim['vcell'], 'r', label='PyBaMM')
    plt.plot(fom_sim['time'], spme_sim.vcell, 'k', label='SPMe')
    plt.xlabel(r'Time, $t$ [s]')
    plt.ylabel(r'$v_\mathrm{cell}$')
    plt.title('Cell voltage vs time')
    plt.legend()
    plt.show()
