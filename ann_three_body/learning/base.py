from tensorflow import keras
import numpy as np
import os
from .util import *


class BaseModel(object):
    """
    Wrapper class for a Keras (TensorFlow) model for making it easier to train an Artificial Neural Network for
    N-body simulations.
    """
    _model: keras.models.Model
    storage_folder: str
    CHECKPOINT_FILE = "cp.ckpt"
    N = 3

    def __init__(self, model, storage_folder=None):
        self._model = model
        if storage_folder is None:
            storage_folder = self.get_default_storage_folder()
        self.storage_folder = storage_folder

        # Check its architecture
        model.summary()

    @classmethod
    def get_default_storage_folder(cls):
        return os.path.join("output", cls.__name__)

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = cls.get_default_storage_folder()
        print(f"Loading model from {path}")

        model = keras.models.load_model(path)
        # model = model.compile()

        return cls(model, storage_folder=path)

    @staticmethod
    def exists(path):
        return os.path.isdir(path)

    def load_weights(self, path=None):
        if path is None:
            path = self.storage_folder
        # Loads the weights
        self._model.load_weights(os.path.join(path, self.CHECKPOINT_FILE))

    @classmethod
    def get_loss_function(cls):
        def loss(y_true, y_pred):
            return keras.losses.mean_absolute_error(y_true, y_pred)

        return loss

    @classmethod
    def new(cls, N,
            num_layers_hidden=10, num_nodes=128,
            storage_folder=None):
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
        model.add(keras.layers.Dense(num_nodes, input_dim=cls._get_input_dimension(N)))
        # model.add(keras.layers.Activation('relu'))

        for i in range(num_layers_hidden):
            model.add(keras.layers.Dense(num_nodes, activation='relu'))
            # model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(cls._get_output_dimension(N)))
        # model.add(keras.layers.Activation('relu'))

        model.compile(loss='mean_absolute_error', #cls.get_loss_function()
                      optimizer='adam',
                      metrics=[])

        model.summary()

        return cls(model, storage_folder=storage_folder)

    def save(self, output_dir):
        self._model.save(output_dir)

    def fit_model(self,
                  sets, N=3,
                  save=True):
        input, output = self.format_data(N, sets)
        (input, output), (input_validation, output_validation) = split_data(input, output)

        checkpoint_path = os.path.join(self.storage_folder, self.CHECKPOINT_FILE)

        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
        try:
            self._model.fit(input,
                            output,
                            validation_data=(input_validation, output_validation),
                            epochs=10_000,
                            callbacks=[cp_callback])
        except KeyboardInterrupt:
            print("\n\nThe model fit got interrupted. Saving the model regardless.")
        self.save(self.storage_folder)

    # @classmethod
    def format_data(self, N, sets):
        num_sets = len(sets)
        input = np.zeros((num_sets, self._get_input_dimension(N)))
        output = np.zeros((num_sets, self._get_output_dimension(N)))

        # Distribute data into numpy array
        for i, set in enumerate(sets):
            [m, r_init, v_init, T_max, r_out, v_out] = set
            input[i, :] = self.format_input(N, T_max, m, r_init, v_init)
            output[i, :] = self.format_output(N, r_out, v_out)
        return input, output

    @staticmethod
    def _get_input_dimension(N):
        return 7 * N

    @staticmethod
    def _get_output_dimension(N):
        return 6 * N

    @staticmethod
    def format_input(N, t, m, r_init, v_init):
        input = np.zeros((7 * N))
        input[0:N] = m.flatten()
        input[N:4 * N] = r_init.flatten()
        input[4 * N:7 * N] = v_init.flatten()
        return input

    @staticmethod
    def format_output(N, r_out, v_out):
        output = np.zeros((6 * N))
        output[:3 * N] = r_out.flatten()
        output[3 * N:] = v_out.flatten()
        return output

    @staticmethod
    def get_data_from_input(data, N=None):
        if N is None:
            N = data.shape[-1] // 7
        m = data[0:N]
        r = data[N:4 * N].reshape((N, 3))
        v = data[4 * N:7 * N].reshape((N, 3))
        return m, r, v

    @staticmethod
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

    def predict_timesteps(self,
                          N, m, r_init, v_init,
                          t,
                          num_timesteps=10):
        output = np.zeros((num_timesteps + 1, self._get_output_dimension(N)))
        output[0, :] = self.format_output(N, r_init, v_init)

        for i in range(num_timesteps):
            r, v = self.get_data_from_output(output[i, :], N=N)
            input = self.format_input(N, t, m, r, v)
            output[i + 1, :] = self._model.predict(input[None, :])
        return output
