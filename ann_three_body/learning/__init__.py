import tensorflow as tf
from tensorflow import keras
import numpy as np


# https://keras.io/getting-started/sequential-model-guide/

def create_model(N, num_layers_hidden=10) -> keras.Model:
    # For number of masses N
    # The input is:
    # Mass_1
    # ...
    # Mass_N
    # R_init_x_1
    # R_init_y_1
    # R_init_z_1
    # ...
    # R_init_x_N
    # R_init_y_N
    # R_init_z_N
    # V_init_x_1
    # V_init_y_1
    # V_init_z_1
    # ...
    # V_init_x_N
    # V_init_y_N
    # V_init_z_N
    # The total input size is then N+3*N+3*N = 7N

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=7 * N))
    model.add(keras.layers.Activation('relu'))

    for i in range(num_layers_hidden):
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(6 * N))
    model.add(keras.layers.Activation('relu'))

    model.compile(loss='mean_squared_error')

    return model


def format_data(N, sets):
    num_sets = len(sets)
    input = np.zeros((num_sets, 7 * N))
    output = np.zeros((num_sets, 6 * N))

    # Distribute data into numpy array
    for i, set in enumerate(sets):
        [m, r_init, v_init, T_max, r_out, v_out] = set
        input[i, 0:N] = m.flatten()
        input[i, N:4 * N] = r_init.flatten()
        input[i, 4 * N:7 * N] = v_init.flatten()
        output[i, :3 * N] = r_out.flatten()
        output[i, 3 * N:] = v_out.flatten()
    return input, output


def fit_model(N, sets):
    input, output = format_data(N, sets)

    model = create_model(N)
    model.fit(input, output)
