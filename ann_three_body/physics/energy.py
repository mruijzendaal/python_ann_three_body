import numpy as np
from . import constants


def kinetic(v, m) -> np.ndarray:
    E_kin = (0.5 * m * v ** 2).sum(axis=(2)) * (constants.m_nd * constants.v_nd**2)**-1
    E_kin = E_kin.reshape([E_kin.shape[0], m.size, 1])
    return E_kin


def gravitational(r, m) -> np.ndarray:
    E_pot = np.zeros([r.shape[0], m.size, 1])

    for i, m_i in enumerate(m):
        for j, m_j in enumerate(m):
            if i == j:
                continue
            E_pot[:, i, 0] += -constants.G * (constants.m_nd**2 / constants.r_nd**2)**-1 *\
                              m_i * m_j / np.linalg.norm(r[:, i, :] - r[:, j, :], axis=1) ** 2
    return E_pot


def total(r, v, m) -> np.ndarray:
    return kinetic(v, m) + gravitational(r, m)


def system_total(r, v, m) -> np.ndarray:
    return total(r, v, m).sum(axis=1)