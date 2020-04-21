import pickle
import os
import physics

def save_object(obj, filename, folder="output"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    filename += ".pkl"
    file = os.path.join(folder, filename)
    pickle.dump(obj, open(file, "wb"))


def load_object(filename, folder="output"):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    file = os.path.join(folder, filename)
    obj = pickle.load(open(file, "rb"))
    return obj


def store_simulation(m, r_init, v_init, t, r_out, v_out, filename, folder="output"):
    save_object([m, r_init, v_init, t, r_out, v_out], filename, folder)


def load_simulation(filename, folder="output"):
    m, r_init, v_init, T_max, r_out, v_out = load_object(filename, folder)
    return m, r_init, v_init, T_max, r_out, v_out


def augment_simulation(m, r_init, v_init, t, r_out, v_out):
    yield m, -r_init, -v_init, t, -r_out, -v_out
    r_init[:, 0] = -r_init[:, 0]
    v_init[:, 0] = -v_init[:, 0]
    r_out[:, 0] = -r_out[:, 0]
    v_out[:, 0] = -v_out[:, 0]
    yield m, r_init, v_init, t, r_out, v_out
    yield m, -r_init, -v_init, t, -r_out, -v_out


def centralize_simulation(m, r_init, v_init, t, r_out, v_out):
    # Set center of mass at (0, 0, 0)
    r_cm = (m * r_init).sum(axis=1, keepdims=True) / m.sum()
    r_init = r_init - r_cm
    r_out = r_out - r_cm

    # Set system momentum to (0, 0, 0)
    p_cm = (m * v_init).sum(axis=1, keepdims=True)
    v_init = v_init - m / m.sum() * p_cm
    v_out = v_out - m / m.sum() * p_cm

    return m, r_init, v_init, t, r_out, v_out


def load_sets_from_folder(folder):
    print(f"Loading data sets from {folder}")
    sets = []
    savefile = os.path.join(folder, "all_sets.pkl")
    if os.path.isfile(savefile):
        sets = load_object("all_sets", folder)
        return sets

    for f in os.listdir(folder):
        if not os.path.isfile(os.path.join(folder, f)) or not f.endswith(".pkl"):
            continue
        try:
            res = load_simulation(f, folder=folder)
        except:
            continue
        sets.append(res)
        # for augres in augment_simulation(*res):
        #     sets.append(augres)
    save_object(sets, "all_sets", folder)

    return sets
