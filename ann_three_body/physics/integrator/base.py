import numpy as np

from .. import system


class PhysicsIntegrator:
    def __init__(self):
        pass

    def integrate(self, N_t, system: system.MechanicalSystem):
        raise Exception("Method `integrate` not inherited by subclass.")

    @staticmethod
    def get_timespan():
        return 0., 0.1

    def get_timesteps(self, N_t):
        return np.linspace(*self.get_timespan(), N_t)
