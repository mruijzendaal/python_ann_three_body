import numpy as np

import physics.integrator
import visualization

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
                   [-0.01, 0, 0]],   # Alpha Centauri A
                  dtype="float64")

integrator = physics.integrator.SciPyIvpIntegrator(r_init, v_init, m, method='LSODA')


def main():
    r, v = integrator.integrate(10_000)

    visualization.show_trajectory(r, v, N)
    visualization.show_energy(r, v, m)
    visualization.animate_trajectory_2d(r, v, N)


if __name__ == "__main__":
    main()
