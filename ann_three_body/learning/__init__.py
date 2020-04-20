import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


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
    # model.add(keras.layers.Activation('relu'))

    for i in range(num_layers_hidden):
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(6 * N))
    # model.add(keras.layers.Activation('relu'))

    # keras.losses.

    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=[])

    model.summary()

    return model


def format_data(N, sets):
    num_sets = len(sets)
    input = np.zeros((num_sets, 7 * N))
    output = np.zeros((num_sets, 6 * N))

    # Distribute data into numpy array
    for i, set in enumerate(sets):
        [m, r_init, v_init, T_max, r_out, v_out] = set
        input[i, :] = format_input(N, m, r_init, v_init)
        output[i, ] = format_output(N, r_out, v_out)
    return input, output


def format_input(N, m, r_init, v_init):
    input = np.zeros((7 * N))
    input[0:N] = m.flatten()
    input[N:4 * N] = r_init.flatten()
    input[4 * N:7 * N] = v_init.flatten()
    return input


def get_data_from_input(data, N=None):
    if N is None:
        N = data.shape[-1] // 7
    m = data[0:N]
    r = data[N:4 * N].reshape((N, 3))
    v = data[4 * N:7 * N].reshape((N, 3))
    return m, r, v


def get_data_from_output(data, N=None):
    if N is None:
        N = data.shape[-1] // 6
    if len(data.shape) == 1:
        r = data[:3 * N].reshape((N, 3))
        v = data[3 * N:6 * N].reshape((N, 3))
    else:
        r = data[:, :3 * N].reshape((data.shape[0], N, 3))
        v = data[:, 3 * N:6 * N].reshape((data.shape[0], N, 3))
    return r, v


def format_output(N, r_out, v_out):
    output = np.zeros((6 * N))
    output[:3 * N] = r_out.flatten()
    output[3 * N:] = v_out.flatten()
    return output


def split_data(input, output, validation_percentage=0.1):
    num_sets = output.shape[0]
    num_validation = int(num_sets * validation_percentage)
    return (input[:-num_validation], output[:-num_validation]), (input[-num_validation:], output[-num_validation:])


def load_model(path):
    model = create_model(3)
    # model = keras.models.load_model(path)

    # # Loads the weights
    model.load_weights(os.path.join(path, "cp.ckpt"))

    # Check its architecture
    model.summary()
    return model


def fit_model(sets, N=3,
              model=None,
              save=True,
              output_dir="output/training_1"):
    input, output = format_data(N, sets)
    (input, output), (input_validation, output_validation) = split_data(input, output)

    checkpoint_path = os.path.join(output_dir, "cp.ckpt")

    if model is None:
        model = load_model(output_dir)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    try:
        model.fit(input,
                  output,
                  validation_data=(input_validation, output_validation),
                  epochs=10_000,
                  callbacks=[cp_callback])
    except KeyboardInterrupt:
        print("\n\nThe model fit got interrupted. Saving the model regardless.")
    model.save(output_dir)

    loss, acc = model.evaluate(input_validation, output_validation, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def predict_timesteps(model: keras.models.Model,
                      N, m, r_init, v_init,
                      num_timesteps=10):
    output_format = format_output(N, r_init, v_init).size
    output = np.zeros((num_timesteps+1, output_format))
    output[0, :] = format_output(N, r_init, v_init)

    for i in range(num_timesteps):
        r, v = get_data_from_output(output[i, :], N=N)
        input = format_input(N, m, r, v)
        output[i+1, :] = model.predict(input[None, :])
    return output
