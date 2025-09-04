"""
gen_traintest_dataset.py

Generate training/testing data for ML model.

2024.11.03 | Created | Wesley Hileman <whileman@uccs.edu>
"""
import os
import pickle

import numpy as np

import util
from util import MuSigmaNormalizer, iir_1p


if __name__ == '__main__':
    # Load simulation data from file.
    with open(os.path.join('datasets', 'PyBaMM_SPMe_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    # Define constants.
    tau = 410
    ts = data['ts']

    # Data I/O adapters.
    fnn_adapter_dict = {
        'thetass': util.ThetassFNNAdapter(tau, ts),
        'phie': util.PhieFNNAdapter(tau, ts),
        'if': util.IfFNNAdapter(tau, ts),
        # FNNs to be trained without single-pole filter output.
        'thetass_without_delta': util.ThetassFNNAdapter(tau, ts, include_delta=False),
        'phie_without_delta': util.PhieFNNAdapter(tau, ts, include_delta=False),
        'if_without_delta': util.IfFNNAdapter(tau, ts, include_delta=False),
    }

    # Prepare training data --------------------------------------------------------------------------------------------

    data['datasets']['bulk_train'] = {}
    for fnn_label, adapter in fnn_adapter_dict.items():
        X_train = []
        Y_train = []
        for series_key, series in data['datasets']['train'].items():    # cc, gitt, drive
            for sim_data in series:
                x = adapter.pack_x(sim_data['spme'])
                y = adapter.pack_y(sim_data['pybamm'])
                sim_data.setdefault('x', {})[fnn_label] = x
                sim_data.setdefault('y', {})[fnn_label] = y
                for ktime in range(x.shape[0]):
                    X_train.append(x[ktime, :])
                    Y_train.append(y[ktime, :])

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Normalize the features.
        norm = MuSigmaNormalizer(X_train, Y_train)
        X_train, Y_train = norm.get_values()

        # Save to data dict.
        data['datasets']['bulk_train'][fnn_label] = {
            'X': X_train,
            'Y': Y_train,
            'norm': norm,
        }

    # Prepare testing data ---------------------------------------------------------------------------------------------

    data['datasets']['bulk_test'] = {}
    for fnn_label, adapter in fnn_adapter_dict.items():
        X_test = []
        Y_test = []
        norm = data['datasets']['bulk_train'][fnn_label]['norm']
        for series_key, series in data['datasets']['test'].items():    # cc, gitt, drive
            for sim_data in series:
                x = adapter.pack_x(sim_data['spme'])
                y = adapter.pack_y(sim_data['pybamm'])
                sim_data.setdefault('x', {})[fnn_label] = x
                sim_data.setdefault('y', {})[fnn_label] = y
                for ktime in range(x.shape[0]):
                    X_test.append(x[ktime, :])
                    Y_test.append(y[ktime, :])

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Normalize the features.
        X_test = norm.pack_x(X_test)
        Y_test = norm.pack_y(Y_test)

        # Save to data dict.
        data['datasets']['bulk_test'][fnn_label] = {
            'X': X_test,
            'Y': Y_test,
            'norm': norm,
        }

    # Save simulation data to a file.
    data['tau'] = tau
    data['fnn_adapter_dict'] = fnn_adapter_dict
    with open(os.path.join('datasets', 'TrainTest_25degC.pickle'), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

