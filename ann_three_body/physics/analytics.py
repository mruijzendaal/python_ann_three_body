import numpy as np
from . import energy


def assess(m, r_1, v_1, r_2, v_2):
    relative_change = energy.conservation(m, r_1, v_1, r_2, v_2)
    print(f"Relative change in energy: {relative_change}")


def get_relative_error(energy: np.ndarray) -> np.ndarray:
    return (energy - energy[0]) / energy[0]
