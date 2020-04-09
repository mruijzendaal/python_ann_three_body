import pickle
import os


def store_simulation(m, r_init, v_init, T_max, r_out, v_out, filename, folder="output"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    filename += ".pkl"
    file = os.path.join(folder, filename)
    pickle.dump([m, r_init, v_init, T_max, r_out, v_out], open(file, "wb"))


def load_simulation(filename, folder="output"):
    filename += ".pkl"
    file = os.path.join(folder, filename)
    m, r_init, v_init, T_max, r_out, v_out = pickle.load(open(file, "rb"))
    return m, r_init, v_init, T_max, r_out, v_out
