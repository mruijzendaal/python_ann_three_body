import numpy as np


def get_relative_error(energy: np.ndarray) -> np.ndarray:
    return (energy - energy[0]) / energy[0]
