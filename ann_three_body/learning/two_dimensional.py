from .base import *


class TwoDimensionalModel(BaseModel):
    @staticmethod
    def _get_space_mask(N):
        return [*[True]*N, *[True]*N, *[False]*N]

    def format_input(self, N, m, r_init, v_init):
        # Mask out masses, rz and vz
        mask = np.array([False] * N + [True, True, False]*N*2)
        input = super().format_input(N, m, r_init, v_init)
        if len(input.shape) == 1:
            return input[mask]
        else:
            return input[:, mask]

    def format_output(self, N, r, v):
        mask = [True, True, False]*N*2
        output = super().format_output(N, r, v)
        if len(output.shape) == 1:
            return output[mask]
        else:
            return output[:, mask]

    @staticmethod
    def _get_input_dimension(N):
        # (rx, ry, ux, uy) for every mass
        return N * 2 * 2

    @staticmethod
    def _get_output_dimension(N):
        # (rx, ry, ux, uy) for every mass
        return N * 2 * 2

    @staticmethod
    def get_data_from_input(data, N=None):
        # Transform back to (m1,..., mN), (rx1, ry1, rz1), ..., (rxN, ryN, rzN), (ux1, uy1, uz1),  ..., (uxN, uyN, uzN)

        if N is None:
            N = data.shape[-1] // 7
        m = data[0:0]
        r = data[N:4 * N].reshape((N, 3))
        v = data[4 * N:7 * N].reshape((N, 3))
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
