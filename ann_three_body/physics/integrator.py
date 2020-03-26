import numpy as np
import scipy as sci
import scipy.integrate
from .constants import *


class MechanicalSystem:
    def __init__(self, r_init, v_init, m):
        self.N = np.size(m)
        self.r_init = r_init
        self.v_init = v_init
        self.m = m

        assert r_init.shape == (self.N, 3)
        assert v_init.shape == (self.N, 3)

    def integrate(self, N_t):
        raise Exception("Method `integrate` not inherited by subclass.")


class SciPyIntegrator(MechanicalSystem):
    def __init__(self, r_init, v_init, m):
        super().__init__(r_init, v_init, m)
        # Constants for the equation of motion, according to the renormalization
        self.K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
        self.K2 = v_nd * t_nd / r_nd

    # A function defining the equations of motion
    def n_body_equations(self, t, state):
        # Create r and v vectors
        # r = [[x_1, y_1, z_1],
        #      [... , ..., ...],
        #      [x_N, y_N, z_N]]
        # v = [[vx_1, vy_1, vz_1],
        #      [... , ..., ...],
        #      [vx_N, vy_N, vz_N]]
        r = state[:3 * self.N].reshape([self.N, 3])
        v = state[3 * self.N:6 * self.N].reshape([self.N, 3])

        # dvi = K_1 / m_i SUM(j=/=i) m_i m_j / r_ij^3 * r_ij
        dv = np.zeros([self.N, 3])

        # dr  = K_2 * v_i
        dr = self.K2 * v

        for i in range(self.N):
            dvi = np.zeros(3, dtype="float64")

            for j in range(self.N):
                if j == i:
                    continue
                m_j = self.m[j]
                r_ij = r[j, :] - r[i, :]
                r_ij_magn = np.linalg.norm(r_ij)

                dvi += m_j / r_ij_magn ** 3 * r_ij
            dv[i, :] = self.K1 * dvi

        return np.concatenate([dr, dv]).flatten()


class OdeIntegrator(SciPyIntegrator):
    def __init__(self, r_init, v_init, m):
        super().__init__(r_init, v_init, m)

    def integrate(self, N_T):
        init_params = np.array([self.r_init, self.v_init])  # create array of initial params
        init_params = init_params.flatten()  # flatten array to make it 1D
        time_span = np.linspace(0, 4, N_T)  # 8 orbital periods and 500 points

        # Run the ODE solver
        solution = scipy.integrate.odeint(self.n_body_equations, init_params, time_span)

        solution = solution.reshape([N_T, 2 * self.N, 3])
        r = solution[:, :3, :]
        v = solution[:, 3:6, :]
        r2d = r[:, :, :2]
        v2d = v[:, :, :2]

        return r, v


class SciPyIvpIntegrator(SciPyIntegrator):
    def __init__(self, r_init, v_init, m, method='LSODA'):
        super().__init__(r_init, v_init, m)
        self.method = method

    def integrate(self, N_T):
        init_params = np.array([self.r_init, self.v_init])  # create array of initial params
        init_params = init_params.flatten()  # flatten array to make it 1D

        timespan = (0., 6.)
        time_span = np.linspace(*timespan, N_T)  # 8 orbital periods and 500 points

        # Run the ODE solver
        solution = scipy.integrate.solve_ivp(self.n_body_equations,
                                             y0=init_params,
                                             t_span=timespan,
                                             t_eval=time_span,
                                             method=self.method,
                                             dense_output=True)
        N_T = solution.t.size
        solution = solution.y.T.reshape([N_T, 2 * self.N, 3])
        r = solution[:, :3, :]
        v = solution[:, 3:6, :]
        r2d = r[:, :, :2]
        v2d = v[:, :, :2]

        return r, v
