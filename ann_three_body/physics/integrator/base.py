import numpy as np

from .. import system


class PhysicsIntegrator:
    T = 0.1

    def __init__(self):
        pass

    def integrate(self, N_t, system: system.MechanicalSystem):
        raise Exception("Method `integrate` not inherited by subclass.")

    def get_timespan(self):
        return 0., self.T

    def get_timesteps(self, N_t):
        return np.linspace(*self.get_timespan(), N_t)
