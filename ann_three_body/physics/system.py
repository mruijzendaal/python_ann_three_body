import numpy as np


class MechanicalSystem:
    N: int
    r_init: np.ndarray
    v_init: np.ndarray
    m: np.ndarray

    def __init__(self, r_init, v_init, m):
        self.N = np.size(m)
        self.r_init = r_init
        self.v_init = v_init
        self.m = m

        assert r_init.shape == (self.N, 3)
        assert v_init.shape == (self.N, 3)

        print(f"Defined the following {self.N}-body system:")
        for i, m_i in enumerate(m.flatten()):
            print(f"m_{i} = {m_i}")


class EearthSunSystem(MechanicalSystem):
    def __init__(self):
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

        # Define initial velocities
        v_init = np.array([[0.00, 0.0, 0],  # Sun
                           [0, 1, 0]],  # Earth
                          dtype="float64")
        super().__init__(r_init, v_init, m)


class EearthSunMoonSystem(MechanicalSystem):
    def __init__(self):
        # Define masses
        m = np.array([1,  # Sun
                      0.012300 * 3e-6,
                      3e-6],  # Earth
                     dtype="float64")
        N = np.size(m)
        m = m.reshape([N, 1])

        # Define initial position vectors
        r_init = np.array([[0, 0, 0],
                           [1 + 0.00257986577, 0, 0],
                           [1, 0, 0]],
                          dtype="float64")

        # Define initial velocities
        v_init = np.array([[0.00, 0.0, 0],  # Sun
                           [0, 1 - 0.0335795836131632, 0],
                           [0, 1, 0]],  # Earth
                          dtype="float64")
        super().__init__(r_init, v_init, m)


class AlphaCentauriSystem(MechanicalSystem):
    def __init__(self):
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
        super().__init__(r_init, v_init, m)


class RandomNbodySystem(MechanicalSystem):
    def __init__(self, N=3):
        m = np.random.rand(N, 1) + 1  # Masses in [0.9, 1.1]
        # m = np.ones((N, 1))
        r = np.random.rand(N, 3) - 0.5  # Initial positions between [-0.5, 0.5]
        r[2, :] = 0.0

        v = np.random.rand(N, 3) - 0.5  # Velocities in [-0.5, 0.5]
        v[2, :] = 0.0
        super().__init__(r, v, m)
