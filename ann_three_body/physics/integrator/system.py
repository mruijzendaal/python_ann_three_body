import numpy as np


class MechanicalSystem:
    N: int

    def __init__(self, r_init, v_init, m):
        self.N = np.size(m)
        self.r_init = r_init
        self.v_init = v_init
        self.m = m

        assert r_init.shape == (self.N, 3)
        assert v_init.shape == (self.N, 3)

    def integrate(self, N_t):
        raise Exception("Method `integrate` not inherited by subclass.")

    @staticmethod
    def get_timespan():
        return 0., 1.

    def get_timesteps(self, N_t):
        return np.linspace(*self.get_timespan(), N_t)
