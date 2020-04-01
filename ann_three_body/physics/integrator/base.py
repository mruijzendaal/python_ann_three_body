import numpy as np

from .. import system


class PhysicsIntegrator:
    physics_system: system.MechanicalSystem

    def __init__(self, physics_system: system.MechanicalSystem):
        self.physics_system = physics_system

    def integrate(self, N_t):
        raise Exception("Method `integrate` not inherited by subclass.")

    @staticmethod
    def get_timespan():
        return 0., 2.

    def get_timesteps(self, N_t):
        return np.linspace(*self.get_timespan(), N_t)