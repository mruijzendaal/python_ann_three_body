import numpy as np

import physics.system
import physics.integrator
import visualization

# The dimensions of data in this project are as follows:
#                       Example: velocity
# Dimension 0: data.    [v_x, v_y, vz]
# Dimension 1: body nr  [data_of_mass_0, data_1, data_2, ...]
# Dimension 2: time     [data_t_0, data_t_1, data_t_2, ...]

# Adjust the initial conditions to whatever problem you want to solve

# problem = physics.system.AlphaCentauriSystem()
# problem = physics.system.EearthSunSystem()
# problem = physics.system.EearthSunMoonSystem()
problem = physics.system.RandomNbodySystem(N=3)

# Adjust to compare different integrators

# integrator = physics.integrator.OdeIntegrator(r_init, v_init, m)
# integrator = physics.integrator.SciPyIvpIntegrator(r_init, v_init, m, method='LSODA')
# integrator = physics.integrator.HamiltonianIntegrator(r_init, v_init, m)
# integrator = physics.integrator.SymplecticIntegrator(problem)
integrator = physics.integrator.BrutusIntegrator(problem)


def main():
    r, v = integrator.integrate(10_000)

    m = problem.m
    N = problem.N

    visualization.show_trajectory(r, v, N)
    visualization.show_energy(r, v, m)
    visualization.animate_trajectory_2d(r, v, N, m)


if __name__ == "__main__":
    main()
