import numpy as np
import scipy.integrate
import scipy

from ..constants import *
from ..system import MechanicalSystem
from .base import PhysicsIntegrator


#
#   Newtonian equations of motion, integrators
#
# A function defining the equations of motion
# @numba.jit
def newtonian_n_body_equations(t, state, N, m, K1, K2):
    # print(N, m, K1, K2)
    # Create r and v vectors
    # r = [[x_1, y_1, z_1],
    #      [..., ..., ...],
    #      [x_N, y_N, z_N]]
    # v = [[vx_1, vy_1, vz_1],
    #      [... , ..., ...],
    #      [vx_N, vy_N, vz_N]]
    r = state[:3 * N].reshape((N, 3))
    v = state[3 * N:6 * N].reshape((N, 3))

    # dvi = K_1 / m_i SUM(j=/=i) m_i m_j / r_ij^3 * r_ij
    dv = np.zeros((N, 3))

    # dr  = K_2 * v_i
    dr = K2 * v

    for i in range(N):
        dvi = np.zeros(3, dtype=np.float64)

        for j in range(N):
            if j == i:
                continue
            m_j = m[j]
            r_ij = r[j, :] - r[i, :]
            r_ij_magn = np.linalg.norm(r_ij)

            dvi += m_j / r_ij_magn ** 3 * r_ij
        dv[i, :] = K1 * dvi

    return np.concatenate((dr, dv)).flatten()


class NewtonianIntegrator(PhysicsIntegrator):
    K1: float
    K2: float

    def __init__(self, system: MechanicalSystem):
        super().__init__(system)
        # Constants for the equation of motion, according to the renormalization
        self.K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
        self.K2 = v_nd * t_nd / r_nd

    def n_body_equations(self, state, t):
        return newtonian_n_body_equations(t, state, self.physics_system.N, self.physics_system.m, self.K1, self.K2)


class OdeIntegrator(NewtonianIntegrator):
    def __init__(self, system: MechanicalSystem):
        super().__init__(system)

    def integrate(self, N_T):
        init_params = np.array([self.physics_system.r_init, self.physics_system.v_init])  # create array of initial params
        init_params = init_params.flatten()  # flatten array to make it 1D

        # Run the ODE solver
        solution = scipy.integrate.odeint(self.n_body_equations,
                                          init_params,
                                          self.get_timesteps(N_T))

        solution = solution.reshape([N_T, 2 * self.physics_system.N, 3])
        r = solution[:, :self.physics_system.N, :]
        v = solution[:, self.physics_system.N:, :]
        r2d = r[:, :, :2]
        v2d = v[:, :, :2]

        return r, v


class SciPyIvpIntegrator(NewtonianIntegrator):
    def __init__(self, system: MechanicalSystem, method='LSODA'):
        super().__init__(system)
        self.method = method

    def integrate(self, N_T):
        init_params = np.array([self.physics_system.r_init, self.physics_system.v_init])  # create array of initial params
        init_params = init_params.flatten()  # flatten array to make it 1D

        timespan = self.get_timespan()
        time = self.get_timesteps(N_T)

        # Run the ODE solver
        solution = scipy.integrate.solve_ivp(lambda t, y: self.n_body_equations(y, t),
                                             y0=init_params,
                                             t_span=timespan,
                                             t_eval=time,
                                             method=self.method,
                                             dense_output=True)
        N_T = solution.t.size
        solution = solution.y.T.reshape([N_T, 2 * self.physics_system.N, 3])
        r = solution[:, :3, :]
        v = solution[:, 3:6, :]
        r2d = r[:, :, :2]
        v2d = v[:, :, :2]

        return r, v
