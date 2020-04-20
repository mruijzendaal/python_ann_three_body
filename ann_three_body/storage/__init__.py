import pickle
import os


def store_simulation(m, r_init, v_init, t, r_out, v_out, filename, folder="output"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    filename += ".pkl"
    file = os.path.join(folder, filename)
    pickle.dump([m, r_init, v_init, t, r_out, v_out], open(file, "wb"))


def load_simulation(filename, folder="output"):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    file = os.path.join(folder, filename)
    m, r_init, v_init, T_max, r_out, v_out = pickle.load(open(file, "rb"))
    return m, r_init, v_init, T_max, r_out, v_out


def load_sets_from_folder(folder):
    sets = []
    for f in os.listdir(folder):
        if not os.path.isfile(os.path.join(folder, f)) or not f.endswith(".pkl"):
            continue
        sets.append(load_simulation(f, folder=folder))
    return sets
