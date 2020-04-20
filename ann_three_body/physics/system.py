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

        # print(f"Defined the following {self.N}-body system:")
        # for i, m_i in enumerate(m.flatten()):
        #     print(f"m_{i} = {m_i}")


class EarthSunSystem(MechanicalSystem):
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
    def __init__(self, N=3,
                 mass_mean=1, mass_dev=0.5,
                 position_mean=0, position_dev=0.5,
                 v_mean=0, v_dev=0.5):
        self.N = N
        self.mass_mean = mass_mean
        self.mass_dev = mass_dev

        self.position_mean = position_mean
        self.position_dev = position_dev

        self.v_mean = v_mean
        self.v_dev = v_dev

        r, v, m = self._new()
        super().__init__(r, v, m)

    def _new(self):
        m = np.random.rand(self.N, 1) * (2 * self.mass_dev) + (self.mass_mean - self.mass_dev)
        r = np.random.rand(self.N, 3) * (2 * self.position_dev) + (self.position_mean - self.position_dev)
        v = np.random.rand(self.N, 3) * (2 * self.v_dev) + (self.v_mean - self.v_dev)

        # Set equal mass
        m = np.ones((self.N, 1))

        # Set z = 0, v_z = 0
        r[2, :] = 0.0
        v[2, :] = 0.0

        # Set center of mass at (0, 0, 0)
        r_cm = (m * r).sum(axis=0) / m.sum()
        r = r - r_cm

        # Set system momentum to (0, 0, 0)
        p_cm = (m * v).sum(axis=0)
        v = v - m / m.sum() * p_cm

        # Check system momentum
        p_cm = (m * v).sum(axis=0)

        return r, v, m

    def set_new(self):
        self.r_init, self.v_init, self.m = self._new()
