import os
import pickle
import timeit

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sympy import simplify, latex, N
import tensorflow as tf

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
    normalizer_thetass_without_delta = data['datasets']['bulk_train']['thetass_without_delta']['norm']
    normalizer_phie_without_delta = data['datasets']['bulk_train']['phie_without_delta']['norm']
    normalizer_if_without_delta = data['datasets']['bulk_train']['if_without_delta']['norm']
    adapter_thetass = data['fnn_adapter_dict']['thetass']
    adapter_phie = data['fnn_adapter_dict']['phie']
    adapter_if = data['fnn_adapter_dict']['if']
    adapter_thetass_without_delta = data['fnn_adapter_dict']['thetass_without_delta']
    adapter_phie_without_delta = data['fnn_adapter_dict']['phie_without_delta']
    adapter_if_without_delta = data['fnn_adapter_dict']['if_without_delta']
    datasets = data['datasets']
    spme = data['spme']

    fnn_thetass = tf.keras.models.load_model(os.path.join('trained_models', 'thetass_25degC.keras'))
    fnn_phie = tf.keras.models.load_model(os.path.join('trained_models', 'phie_25degC.keras'))
    fnn_if = tf.keras.models.load_model(os.path.join('trained_models', 'if_25degC.keras'))
    sr_thetass = PySRRegressor.from_file(run_directory=os.path.join('trained_sr_models', 'thetass'))
    sr_phie = PySRRegressor.from_file(run_directory=os.path.join('trained_sr_models', 'phie'))
    sr_if = PySRRegressor.from_file(run_directory=os.path.join('trained_sr_models', 'if'))
    print('thetass:', latex(N(simplify(sr_thetass.sympy()),3)))
    print('phie:', latex(N(simplify(sr_phie.sympy()),3)))
    print('if:', latex(N(simplify(sr_if.sympy()),3)))

    X_test_thetass = data['datasets']['bulk_test']['thetass']['X']
    X_test_phie = data['datasets']['bulk_test']['phie']['X']
    X_test_if = data['datasets']['bulk_test']['if']['X']
    profile_thetass_sr = timeit.repeat(lambda: sr_thetass.predict(X_test_thetass), number=1000)
    profile_phie_sr = timeit.repeat(lambda: sr_phie.predict(X_test_phie), number=1000)
    profile_if_sr = timeit.repeat(lambda: sr_if.predict(X_test_if), number=1000)
    profile_thetass_nn = timeit.repeat(lambda: fnn_thetass(X_test_thetass), number=1000)
    profile_phie_nn = timeit.repeat(lambda: fnn_phie(X_test_phie), number=1000)
    profile_if_nn = timeit.repeat(lambda: fnn_if(X_test_if), number=1000)
    print('thetass:', np.mean(profile_thetass_sr)/X_test_thetass.shape[0], np.mean(profile_thetass_nn)/X_test_thetass.shape[0])
    print('phie:', np.mean(profile_phie_sr)/X_test_phie.shape[0], np.mean(profile_phie_nn)/X_test_phie.shape[0])
    print('if:', np.mean(profile_if_sr)/X_test_if.shape[0], np.mean(profile_if_nn)/X_test_if.shape[0])
