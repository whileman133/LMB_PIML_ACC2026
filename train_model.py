"""
train_model.py
"""
import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Masking, TimeDistributed, Concatenate, Dropout
from tensorflow.keras import Sequential

tf.random.set_seed(42)

if __name__ == "__main__":
    # Load data from file.
    with open(os.path.join('datasets', 'TrainTest_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    # Build and train models.
    for fnn_label in data['fnn_adapter_dict']:
        if not fnn_label.endswith('_mtau'):
            continue
        # if fnn_label.startswith('thetass'):
        #     continue

        X_train = data['datasets']['bulk_train'][fnn_label]['X']
        Y_train = data['datasets']['bulk_train'][fnn_label]['Y']
        X_test = data['datasets']['bulk_test'][fnn_label]['X']
        Y_test = data['datasets']['bulk_test'][fnn_label]['Y']
        n_input = X_train.shape[-1]
        n_output = Y_train.shape[-1]

        # Build model.
        n_hidden_nodes = 32 if fnn_label.startswith('thetass') else 128
        n_layers = 3 if fnn_label.startswith('thetass') else 5
        n_epochs = 1000
        # if fnn_label.endswith('_mtau'):
        #     n_hidden_nodes = 32 if fnn_label.startswith('thetass') else 64
        #     n_layers = 3
        #     n_epochs = 1000
        # else:
        #     n_hidden_nodes = 32 if fnn_label.startswith('thetass') else 128
        #     n_layers = 3 if fnn_label.startswith('thetass') else 5
        #     n_epochs = 1000
        model = Sequential(name='FNN')
        model.add(Input(shape=(n_input,), name='input'))
        for k in range(n_layers):
            model.add(Dense(n_hidden_nodes, activation='relu', name=f'Dense_{k}'))
        model.add(Dense(n_output, activation='linear', name='Output'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        # Train model.
        model.fit(
            X_train, Y_train,
            batch_size=1024,
            epochs=n_epochs,
            validation_data=(X_test, Y_test)
        )

        # Save trained model.
        model.save(os.path.join("trained_models", f"{fnn_label}_25degC.keras"))