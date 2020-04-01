import numpy as np

import physics.integrator
import visualization


# The dimensions of data in this project are as follows:
#                       Example: velocity
# Dimension 0: data.    [v_x, v_y, vz]
# Dimension 1: body nr  [data_of_mass_0, data_1, data_2]
# Dimension 2: time     [data_t_0, data_t_1, data_t_2]

def initial_conditions_threebody():
    # Define masses
    m = np.array([1.1,  # Alpha Centauri A
                  0.5,  # Alpha Centauri C
                  0.907],  # Alpha Centauri B
                 dtype="float64")
    N = np.size(m)
    m = m.reshape([N, 1])

    # Define initial position vectors
    r_init = np.array([[-0.5, 0, 0],
                       [0, 0.5, 0],
                       [0.5, 0, 0]],
                      dtype="float64")

    # Find Centre of Mass
    r_com = m * r_init / m.sum()
    # Define initial velocities
    v_init = np.array([[0.01, 0.01, 0],  # Alpha Centauri B
                       [0.01, 0.01, 0],  # Alpha Centauri C
                       [-0.01, 0, 0]],  # Alpha Centauri A
                      dtype="float64")
    return N, m, r_init, v_init


def initial_conditions_random(N=3):
    m = np.random.rand(N, 1) * 0.2 + 0.9  # Masses in [0.9, 1.1]
    m = np.ones((N, 1))
    r = np.random.rand(N, 3) - 0.5  # Initial positions between [-0.5, 0.5]
    r[2, :] = 0.0

    v = np.random.rand(N, 3) - 0.5  # Velocities in [-0.5, 0.5]
    v[2, :] = 0.0
    return N, m, r, v


def initial_conditions_kepler():
    # Define masses
    m = np.array([1,  # Sun
                  3e-6],  # Earth
                 dtype="float64")
    N = np.size(m)
    m = m.reshape([N, 1])

    # Define initial position vectors
    r_init = np.array([[0, 0, 0],
                       [1, 0, 0]],
                      dtype="float64")

    # Find Centre of Mass
    r_com = m * r_init / m.sum()
    # Define initial velocities
    v_init = np.array([[0.00, 0.0, 0],  # Sun
                       [0, 1, 0]],  # Earth
                      dtype="float64")

    return N, m, r_init, v_init


N, m, r_init, v_init = initial_conditions_random(N=3)
# N, m, r_init, v_init = initial_conditions_threebody()
# N, m, r_init, v_init = initial_conditions_kepler()


# integrator = physics.integrator.OdeIntegrator(r_init, v_init, m)
# integrator = physics.integrator.SciPyIvpIntegrator(r_init, v_init, m, method='LSODA')
# integrator = physics.integrator.HamiltonianIntegrator(r_init, v_init, m)
integrator = physics.integrator.SymplecticIntegrator(r_init, v_init, m)


def main():
    r, v = integrator.integrate(10_000)

    visualization.show_trajectory(r, v, N)
    visualization.show_energy(r, v, m)
    visualization.animate_trajectory_2d(r, v, N)


if __name__ == "__main__":
    main()
