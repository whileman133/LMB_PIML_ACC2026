"""
train_model.py
"""
import os
import pickle

from pysr import PySRRegressor


if __name__ == "__main__":
    with open(os.path.join('datasets', 'TrainTest_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    for fnn_label in data['fnn_adapter_dict']:
        if fnn_label.endswith('_without_delta'):
            continue

        X_train = data['datasets']['bulk_train'][fnn_label]['X']
        Y_train = data['datasets']['bulk_train'][fnn_label]['Y']
        X_test = data['datasets']['bulk_test'][fnn_label]['X']
        Y_test = data['datasets']['bulk_test'][fnn_label]['Y']
        n_input = X_train.shape[-1]
        n_output = Y_train.shape[-1]

        model = PySRRegressor(
            maxsize=45,       # maximum expression complexity
            niterations=50,
            population_size=100,
            binary_operators=["+","*","^"],
            constraints={'^': (-1, 3)},  # limit complexity of exponents
            elementwise_loss="L2DistLoss()",
            output_directory='trained_sr_models',
            run_id=fnn_label,
            progress=False,
        )
        model.fit(X_train[::10,:], Y_train[::10,:])

