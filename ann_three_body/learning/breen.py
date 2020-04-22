from .base import *


class BreenModel(BaseModel):
    """
    This system describes a randomly generated system according to
    Newton vs the machine: solving the chaotic three-body problem using deep neural networks, Breen (2019)

    The original model has input and output
                            t               (x1, y1)
                            (x2, y2)  -->   (x2, y2)

    Here, we are going to implement
                            t               (x1, y1)
                            (x1, y1)        (x2, y2)
                            (x2, y2)  -->   (x3, y3)
                            (x3, y3)        (vx1, vy1)
                                            (vx2, vy2)
                                            (vx3, vy3)
    """

    def format_input(self, N, t, m, r_init, v_init):
        # Mask out masses, rz and vz
        input = np.zeros((N*2+1))
        input[..., 0] = t
        input[..., 1:] = r_init.flatten()[[True, True, False]*N]
        return input

    def format_output(self, N, r, v):
        output = np.zeros((N*2*2))
        output[..., 0:N * 2] = r.flatten()[[True, True, False]*N]
        output[..., N * 2:N*2*2] = v.flatten()[[True, True, False]*N]
        return output

    @staticmethod
    def _get_input_dimension(N):
        # t, (rx, ry) for every mass
        return N * 2 + 1

    @staticmethod
    def _get_output_dimension(N):
        # (rx, ry, ux, uy) for every mass
        return N * 2 * 2

    @staticmethod
    def get_data_from_input(data, N=None):
        # Transform back to (m1,..., mN), (rx1, ry1, rz1), ..., (rxN, ryN, rzN), (ux1, uy1, uz1),  ..., (uxN, uyN, uzN)

        if N is None:
            N = (data.shape[-1] - 1) // 6
        m = data[0:0]
        r = np.zeros((N, 3))
        v = np.zeros((N, 3))
        r[:, 0:2] = data[0:2 * N].reshape((N, 3))
        v[:, 0:2] = data[2 * N:].reshape((N, 3))
        return m, r, v

    @staticmethod
    def get_data_from_output(data, N=None):
        if N is None:
            N = data.shape[-1] // 6

        mask = [True, True, False]*N
        if len(data.shape) == 1:
            r = np.zeros((N * 3))
            v = np.zeros((N * 3))
            r[mask] = data[:2 * N]
            v[mask] = data[2 * N:]
            r = r.reshape((N, 3))
            v = v.reshape((N, 3))
        else:
            r = np.zeros((data.shape[0], N * 3))
            v = np.zeros((data.shape[0], N * 3))
            r[:, mask] = data[:, :2 * N]
            v[:, mask] = data[:, 2 * N:]
            r = r.reshape((data.shape[0], N, 3))
            v = v.reshape((data.shape[0], N, 3))
        return r, v
