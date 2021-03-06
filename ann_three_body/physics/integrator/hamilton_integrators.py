import numpy as np
import numba
import time

from ..constants import *
from .base import PhysicsIntegrator
from ..system import MechanicalSystem


@numba.jit
def drdp(dt, r, p, N, m, K1, K2):
    dp = np.zeros((N, 3))

    for i in range(N):
        mask = np.zeros(m.size, dtype=np.bool_)
        mask[i] = True
        rij = r[i, :] - r[~mask, :]
        rij_len = np.sqrt((rij ** 2).sum(axis=1)).reshape(N - 1, 1)
        dp[i, :] = -K1 * m[i] * (m[~mask] * 2 * rij / rij_len ** 3).sum(axis=0)

    p_new = p + dt * dp
    r_new = r + dt * (K2 * p_new / m)

    return p_new, r_new


@numba.jit
def hamilton_non_symplectic(r0, p0, t, N, m, K1, K2):
    dt = t[1] - t[0]
    Nt = t.size
    r = np.zeros((Nt, N, 3))
    p = np.zeros((Nt, N, 3))
    r[0, :, :], p[0, :, :] = r0, p0

    for i in range(Nt - 1):
        p_i, r_i = p[i, :, :], r[i, :, :]
        p[i + 1, :, :], r[i + 1, :, :] = drdp(dt, r_i, p_i, N, m, K1, K2)
    return r, p


#
#   Hamiltonian equations of motion, integrators
#
class HamiltonianIntegrator(PhysicsIntegrator):
    K3: float
    K4: float

    def __init__(self):
        super().__init__()
        # Constants for the equation of motion, according to the renormalization
        self.K3 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
        self.K4 = v_nd * t_nd / r_nd

    def integrate(self, N_t, system: MechanicalSystem):
        t = self.get_timesteps(N_t)
        r, p = hamilton_non_symplectic(system.r_init,
                                       system.v_init * system.m,
                                       t,
                                       system.N, system.m, self.K3, self.K4)
        # r, p = hamilton1st(self.r_init, self.v_init * self.m, t, self.N, self.m, self.K3, self.K4)
        return r, p / self.m


cd_euler = np.array([[1],  # c
                     [1]])  # d
cd_verlet = np.array([[0, 1],
                      [1 / 2, 1 / 2]])
cd_ruth3 = np.array([[1, -2 / 3, 2 / 3],
                     [-1 / 24, 3 / 4, 7 / 24]])
cd_ruth4 = np.array([[1 / (2 * (2 - 2 ** (1 / 3))), (1 - 2 ** (1 / 3)) / (2 * (2 - 2 ** (1 / 3))),
                      (1 - 2 ** (1 / 3)) / (2 * (2 - 2 ** (1 / 3))), 1 / (2 * (2 - 2 ** (1 / 3)))],
                     [1 / (2 - 2 ** (1 / 3)), -2 ** (1 / 3) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3)), 0]])

for c, d in [cd_euler, cd_verlet, cd_ruth3, cd_ruth4]:
    assert c.sum() == 1
    assert d.sum() - 1 < 1e-10


@numba.jit
def dpdr_symplectic(c, d, dt, r, p, N, m, K1, K2):
    p_new = p.copy()
    r_new = r.copy()

    # Loop through the sub-timesteps and update q and p using coefficients c_i and d_i each sub-timestep
    for ci, di in zip(c, d):
        # For each sub-timestep, the calculations need to be performed on the state at the previous timestep
        # p_old, r_old --> Use in calculation
        # p_new, r_new --> Store result
        p_old, r_old = p_new.copy(), r_new.copy()

        dp = np.zeros((N, 3))
        r_new += (ci * dt) * (K2 * p_old / m)

        for i in range(N):
            mask = np.zeros(m.size, dtype=np.bool_)
            mask[i] = True
            rij = r_new[i, :] - r_new[~mask, :]
            rij_len = np.sqrt((rij ** 2).sum(axis=1)).reshape(N - 1, 1)
            dp[i, :] = -K1 * m[i] * (m[~mask] * 2 * rij / rij_len ** 3).sum(axis=0)
        p_new += (di * dt) * dp

    return p_new, r_new


@numba.jit
def hamilton_sympletic(cd, r0, p0, t, N, m, K1, K2):
    dt = t[1] - t[0]
    Nt = t.size
    r = np.zeros((Nt, N, 3))
    p = np.zeros((Nt, N, 3))
    r[0, :, :], p[0, :, :] = r0, p0

    if cd is None:
        for i in range(Nt - 1):
            p_i, r_i = p[i, :, :], r[i, :, :]
            p[i + 1, :, :], r[i + 1, :, :] = drdp(dt, r_i, p_i, N, m, K1, K2)
    else:
        c, d = cd[0, :], cd[1, :]

        for i in range(Nt - 1):
            p_i, r_i = p[i, :, :], r[i, :, :]
            p[i + 1, :, :], r[i + 1, :, :] = dpdr_symplectic(c, d, dt, r_i, p_i, N, m, K1, K2)
    return r, p


class SymplecticIntegrator(PhysicsIntegrator):
    K3: float
    K4: float

    def __init__(self):
        super().__init__()
        # Constants for the equation of motion, according to the renormalization
        self.K3 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
        self.K4 = v_nd * t_nd / r_nd

    def integrate(self, N_t, system: MechanicalSystem):
        t = self.get_timesteps(N_t)
        r, p = hamilton_sympletic(cd_ruth4, system.r_init,
                                  system.v_init * system.m,
                                  t,
                                  system.N, system.m, self.K3, self.K4)
        return r, p / system.m


@numba.jit
def dpdr_brutus(c, d, dt, r, p, N, m, K1, K2):
    p_new = p.copy()
    r_new = r.copy()

    # Loop through the sub-timesteps and update q and p using coefficients c_i and d_i each sub-timestep
    for ci, di in zip(c, d):
        # For each sub-timestep, the calculations need to be performed on the state at the previous timestep
        # p_old, r_old --> Use in calculation
        # p_new, r_new --> Store result
        p_old, r_old = p_new.copy(), r_new.copy()

        dp = np.zeros((N, 3))
        r_new += (ci * dt) * (K2 * p_old / m)

        for i in range(N):
            mask = np.zeros(m.size, dtype=np.bool_)
            mask[i] = True
            rij = r_new[i, :] - r_new[~mask, :]
            rij_len = np.sqrt((rij ** 2).sum(axis=1)).reshape(N - 1, 1)
            dp[i, :] = -K1 * m[i] * (m[~mask] * 2 * rij / rij_len ** 3).sum(axis=0)
        p_new += (di * dt) * dp

    return p_new - p, r_new - r


@numba.jit
def brutus_timestep(N, m, r, p, dr, dp):
    eta = 1 / 100

    a = dp / m

    dt = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        rij = np.sqrt(((r[i, :] - r[mask, :]) ** 2).sum(axis=1)).reshape(N - 1, 1)
        aij = np.sqrt(((a[i, :] - a[mask, :]) ** 2).sum(axis=1)).reshape(N - 1, 1)
        dt[i] = np.sqrt(rij / aij).min()
    dt = eta * dt.min()
    # print(1/dt)
    return dt


@numba.jit(nopython=True, cache=True)
def hamilton_sympletic_brutus(cd, r0, p0, tmax, N, m, K1, K2):
    N_resize = 1000
    a_resize = np.zeros((N_resize, N, 3), dtype=np.float64)
    r = np.zeros((N_resize, N, 3), dtype=np.float64)
    p = np.zeros((N_resize, N, 3), dtype=np.float64)
    t = np.zeros(N_resize, dtype=np.float64)
    r[0, :, :], p[0, :, :] = r0, p0

    t_i = 0
    i = 0

    # Initial timestep very small compared to timescale
    dt = 1 / 1000000
    if cd is None:
        raise (ValueError("WTF"))
        # while t < tmax:
        #     p_i, r_i = p[i, :, :], r[i, :, :]
        #     dp, dr = drdp(dt, r_i, p_i, N, m, K1, K2)
        #     dt = brutus_timestep(N, r_i, p_i, dr, dp)
        #     p[i + 1, :, :], r[i + 1, :, :] = dp * dt, dr * dt
    else:
        c, d = cd[0, :], cd[1, :]

        while t_i < tmax:
            if i + 1 == r.shape[0]:
                r = np.concatenate((r, a_resize.copy()))
                p = np.concatenate((p, a_resize.copy()))
                t = np.concatenate((t, np.zeros(N_resize)))
            p_i, r_i = p[i, :, :], r[i, :, :]
            p_new, r_new = dpdr_symplectic(c, d, dt, r_i, p_i, N, m, K1, K2)
            dt = brutus_timestep(N, m, r_i, p_i, r_new / dt, p_new / dt)
            dt = min(dt, tmax - t_i)
            p[i + 1, :, :], r[i + 1, :, :] = p_new, r_new
            t_i += dt
            t[i+1] = t_i
            i += 1
    return r[:i, :, :], p[:i, :, :], t[:i]


class BrutusIntegrator(PhysicsIntegrator):
    K3: float
    K4: float

    def __init__(self):
        super().__init__()
        # Constants for the equation of motion, according to the renormalization
        self.K3 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
        self.K4 = v_nd * t_nd / r_nd

    def integrate(self, N_t, system: MechanicalSystem):
        time_start = time.time()
        r, p, t = hamilton_sympletic_brutus(cd_ruth4, system.r_init,
                                            system.v_init * system.m,
                                            self.get_timespan()[-1],
                                            system.N, system.m, self.K3, self.K4)
        print(f"Number of Brutus timesteps: {r.shape[0]}")
        print(f"Elapsed time: {time.time() - time_start:.1f}s")
        return r, p / system.m, t
