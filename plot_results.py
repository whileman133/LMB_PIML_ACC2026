import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import util

#tf.config.run_functions_eagerly(True)
matplotlib.set_loglevel("error")
plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])


if __name__ == "__main__":
    # Load data from file.
    with open(os.path.join('datasets', 'TrainTest_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    normalizer_thetass = data['datasets']['bulk_train']['thetass']['norm']
    normalizer_phie = data['datasets']['bulk_train']['phie']['norm']
    normalizer_if = data['datasets']['bulk_train']['if']['norm']
    adapter_thetass = data['fnn_adapter_dict']['thetass']
    adapter_phie = data['fnn_adapter_dict']['phie']
    adapter_if = data['fnn_adapter_dict']['if']
    datasets = data['datasets']
    spme = data['spme']

    # Load models from file.
    fnn_thetass = tf.keras.models.load_model(os.path.join('trained_models', 'thetass_25degC.keras'))
    fnn_phie = tf.keras.models.load_model(os.path.join('trained_models', 'phie_25degC.keras'))
    fnn_if = tf.keras.models.load_model(os.path.join('trained_models', 'if_25degC.keras'))

    predict_thetass = adapter_thetass.wrap(normalizer_thetass.wrap(fnn_thetass))
    predict_phie = adapter_phie.wrap(normalizer_phie.wrap(fnn_phie))
    predict_if = adapter_if.wrap(normalizer_if.wrap(fnn_if))
    predict = lambda spme_state: {**predict_thetass(spme_state), **predict_phie(spme_state), **predict_if(spme_state)}

    for dataset_key, dataset in datasets.items():     # train, test
        if dataset_key not in ['train', 'test']:
            continue

        # if dataset_key == 'train':
        #     continue

        series_labels = []
        spec_labels = []
        rmse_vcell_spme = []
        rmse_vcell_hybrid = []
        rmse_vcell_rom = []
        rmse_phie2_spme = []
        rmse_phie2_hybrid = []
        rmse_if2_spme = []
        rmse_if2_hybrid = []
        rmse_thetass2_spme = []
        rmse_thetass2_hybrid = []
        for series_key, series in dataset.items():    # cc, gitt, drive
            for sim_data in series:
                spec_string = util.sim_specs(series_key, sim_data)
                time = sim_data['pybamm']['time']
                iapp = sim_data['pybamm']['iapp']
                plain_sim = spme.run(iapp, sim_data['soc0'])
                hybrid_sim = spme.run(iapp, sim_data['soc0'], correction_fn=predict)

                thetass2_true = sim_data['pybamm']['thetass2']
                thetass2_hybrid = hybrid_sim.thetass
                thetass2_spme = plain_sim.thetass

                phie2_true = sim_data['pybamm']['phie2']
                phie2_hybrid = hybrid_sim.phie2
                phie2_spme = plain_sim.phie2

                if2_true = sim_data['pybamm']['if2']
                if2_hybrid = hybrid_sim.pos_if
                if2_spme = plain_sim.pos_if

                vcell_true = sim_data['pybamm']['vcell']
                vcell_hybrid = hybrid_sim.vcell
                vcell_spme = plain_sim.vcell

                soc_pct = sim_data['pybamm']['soc_pct']
                ind = soc_pct >= 3.0

                # Compute RMSE.
                series_labels.append(series_key)
                spec_labels.append(spec_string)
                rmse_vcell_spme.append(1000*np.sqrt(np.mean((vcell_true[ind] - vcell_spme[ind])**2, axis=0)))
                rmse_vcell_hybrid.append(1000*np.sqrt(np.mean((vcell_true[ind] - vcell_hybrid[ind])**2, axis=0)))
                rmse_phie2_spme.append(1000*np.sqrt(np.mean((phie2_true - phie2_spme)**2, axis=0)))
                rmse_phie2_hybrid.append(1000*np.sqrt(np.mean((phie2_true - phie2_hybrid)**2, axis=0)))
                rmse_if2_spme.append(1000*np.sqrt(np.mean((if2_true - if2_spme)**2, axis=0)))
                rmse_if2_hybrid.append(1000*np.sqrt(np.mean((if2_true - if2_hybrid)**2, axis=0)))
                rmse_thetass2_spme.append(1000*np.sqrt(np.mean((thetass2_true - thetass2_spme)**2, axis=0)))
                rmse_thetass2_hybrid.append(1000*np.sqrt(np.mean((thetass2_true - thetass2_hybrid)**2,axis=0)))

                print(f"{dataset_key} {series_key} {spec_string}: {rmse_vcell_spme[-1]:.4f}mV -> "
                      f" {rmse_vcell_hybrid[-1]:.4f}mV RMSE")

                # Compare to ROM if available.
                compare_file = os.path.join('MATLAB', "ROMSIM", f"{dataset_key}_{series_key}_{spec_string}.mat")
                compare_rom = False
                if os.path.exists(compare_file):
                    compare_rom = True
                    compare_data = util.load_mat_rom_sim(compare_file)
                    vcell_rom = compare_data.vcell
                    diff = len(time) - len(vcell_rom)
                    if diff > 0:
                        vcell_rom = np.concatenate((vcell_rom, np.full(diff, np.nan)))
                    rmse_vcell_rom.append(1000 * np.sqrt(np.nanmean((vcell_true[ind] - vcell_rom[ind]) ** 2, axis=0)))
                else:
                    rmse_vcell_rom.append(np.nan)

                # Make plot directory.
                plot_dir = os.path.join('plots', 'trained', dataset_key, series_key, spec_string)
                if not os.path.isdir(plot_dir):
                    os.makedirs(plot_dir)

                # Restrict plotted time domain for drive-cycle plots for improved readability.
                if series_key == 'drive':
                    time_ind = np.logical_and(1000 <= time, time <= 1200)
                else:
                    time_ind = np.ones_like(time, dtype=bool)

                # Plot comparisons.
                fig, ax = plt.subplots(constrained_layout=True)
                #fig.patch.set_facecolor('#EFE5C3')
                #ax.set_facecolor('white')
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
                ax.set_box_aspect(1 / scipy.constants.golden)
                plt.plot(time[time_ind], thetass2_true[time_ind], label=f'Newman', zorder=9)
                plt.plot(time[time_ind], thetass2_hybrid[time_ind], '--', label=f'PIML', zorder=10)
                plt.plot(time[time_ind], thetass2_spme[time_ind], label=f'SPMe')
                plt.xlabel(r'Time [s]')
                plt.ylabel(r'$\theta_\mathrm{ss}(\tilde{x}=2)$')
                plt.title(r'$\theta_\mathrm{ss}(\tilde{x}=2)$: ' f"{dataset_key.capitalize()} {spec_string}"
                          f" {series_key.upper()}")
                plt.legend().set_zorder(100)
                plt.savefig(os.path.join(plot_dir, f'thetass.png'), bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir, f'thetass.eps'), bbox_inches='tight')
                plt.close()

                fig, ax = plt.subplots(constrained_layout=True)
                #fig.patch.set_facecolor('#EFE5C3')
                #ax.set_facecolor('white')
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
                ax.set_box_aspect(1 / scipy.constants.golden)
                plt.plot(time[time_ind], 1000*phie2_true[time_ind], label=f'Newman', zorder=9)
                plt.plot(time[time_ind], 1000*phie2_hybrid[time_ind], '--', label=f'PIML', zorder=10)
                plt.plot(time[time_ind], 1000*phie2_spme[time_ind], label=f'SPMe')
                plt.xlabel(r'Time [s]')
                plt.ylabel(r'$\phi_\mathrm{e}(\tilde{x}=2)$ [mV]')
                plt.title(r'$\phi_\mathrm{e}(\tilde{x}=2)$: ' f"{dataset_key.capitalize()} {spec_string}"
                          f" {series_key.upper()}")
                plt.legend().set_zorder(100)
                plt.savefig(os.path.join(plot_dir, f'phie.png'), bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir, f'phie.eps'), bbox_inches='tight')
                plt.close()

                fig, ax = plt.subplots(constrained_layout=True)
                #fig.patch.set_facecolor('#EFE5C3')
                #ax.set_facecolor('white')
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
                ax.set_box_aspect(1 / scipy.constants.golden)
                plt.plot(time[time_ind], if2_true[time_ind], label=f'Newman', zorder=9)
                plt.plot(time[time_ind], if2_hybrid[time_ind], '--', label=f'PIML', zorder=10)
                plt.plot(time[time_ind], if2_spme[time_ind], label=f'SPMe')
                plt.xlabel(r'Time [s]')
                plt.ylabel(r'$i_\mathrm{f}(\tilde{x}=2)$ [A]')
                plt.title(r'$i_\mathrm{f}(\tilde{x}=2)$: ' f"{dataset_key.capitalize()} {spec_string}"
                          f" {series_key.upper()}")
                plt.legend().set_zorder(100)
                plt.savefig(os.path.join(plot_dir, f'if.png'), bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir, f'if.eps'), bbox_inches='tight')
                plt.close()

                fig, ax = plt.subplots(constrained_layout=True)
                #fig.patch.set_facecolor('#EFE5C3')
                #ax.set_facecolor('white')
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
                ax.set_box_aspect(1 / scipy.constants.golden)
                plt.plot(time[time_ind]/60, vcell_true[time_ind], label=f'Newman', zorder=9)
                plt.plot(time[time_ind]/60, vcell_hybrid[time_ind], '--', label=f'PIML', zorder=10)
                plt.plot(time[time_ind]/60, vcell_spme[time_ind], label=f'SPMe')
                if compare_rom:
                    plt.plot(time[time_ind]/60, vcell_rom[time_ind], label=f'HRA/outBlend')
                plt.xlabel(r'Time, $t$ [min]')
                plt.ylabel(r'Cell Voltage, $v_\mathrm{cell}$ [V]')
                plt.title(r'Cell Voltage: ' f"{dataset_key.capitalize()} {spec_string}"
                          f" {series_key.upper()}")
                plt.legend().set_zorder(100)
                plt.savefig(os.path.join(plot_dir, f'vcell.png'), bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir, f'vcell.eps'), bbox_inches='tight')
                plt.close()

        # Make data directory.
        data_dir = os.path.join('performance_data', 'trained', dataset_key)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        metrics = pd.DataFrame({
            'series': series_labels,
            'spec': spec_labels,
            'rmse_vcell_spme [mV]': rmse_vcell_spme,
            'rmse_vcell_hybrid [mV]': rmse_vcell_hybrid,
            'rmse_vcell_rom [mV]': rmse_vcell_rom,
            'rmse_thetass2_spme [milli]': rmse_thetass2_spme,
            'rmse_thetass2_hybrid [milli]': rmse_thetass2_hybrid,
            'rmse_phie2_spme [mV]': rmse_phie2_spme,
            'rmse_phie2_hybrid [mV]': rmse_phie2_hybrid,
            'rmse_if2_spme [mA]': rmse_if2_spme,
            'rmse_if2_hybrid [mA]': rmse_if2_hybrid,
         })
        metrics.to_excel(os.path.join(data_dir, 'metrics.xlsx'))
