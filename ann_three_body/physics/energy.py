import numpy as np

from . import constants


def kinetic(v, m) -> np.ndarray:
    E_kin = (0.5 * m * v ** 2).sum(axis=-1, keepdims=True) * (constants.m_nd * constants.v_nd**2)
    # E_kin = E_kin.reshape([E_kin.shape[0], m.size, 1])
    return E_kin


def gravitational(r, m) -> np.ndarray:
    if len(r.shape) == 3:
        # Include time-dependence
        E_pot = np.zeros([r.shape[0], m.size, 1])
    else:
        E_pot = np.zeros((m.size, 1))

    for i, m_i in enumerate(m):
        for j, m_j in enumerate(m):
            if i == j:
                continue
            rij = np.linalg.norm(r[..., i, :] - r[..., j, :], axis=-1, keepdims=True)
            E_pot[..., i, :] += - constants.G * m_i * m_j / rij
    E_pot *= (constants.m_nd**2 / constants.r_nd)
    return E_pot


def total(r, v, m) -> np.ndarray:
    return kinetic(v, m) + gravitational(r, m)


def system_total(r, v, m) -> np.ndarray:
    return total(r, v, m).sum(axis=-2)


def conservation(m, r_1, v_1, r_2, v_2):
    E_2 = system_total(r_2, v_2, m)
    E_1 = system_total(r_1, v_1, m)
    return (E_2 - E_1)/E_1