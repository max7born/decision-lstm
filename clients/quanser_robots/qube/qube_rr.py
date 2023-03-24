import numpy as np
from ..common import QSocket
from .base import QubeBase
from .ctrl import CalibrCtrl


class Qube(QubeBase):
    def __init__(self, ip, fs_ctrl):
        super(Qube, self).__init__(fs=fs_ctrl, fs_ctrl=fs_ctrl)
        self._qsoc = QSocket(ip, x_len=self.state_space.shape[0],
                             u_len=self.action_space.shape[0])
        self._sens_offset = None

    def _calibrate(self):
        # Reset calibration
        n_pos = int(self.state_space.shape[0] / 2)
        self._sens_offset = np.zeros(n_pos, dtype=np.float32)

        # Record alpha offset if alpha == k * 2pi (happens upon reconnect)
        x = self._zero_sim_step()
        if np.abs(x[1]) > np.pi:
            diff = 1.0
            while diff > 0.0:
                xn = self._zero_sim_step()
                diff = np.linalg.norm(xn - x)
                x = xn
            self._sens_offset[1] = x[1]

        # Find theta offset by going to joint limits
        x = self._zero_sim_step()
        act = CalibrCtrl(1. / self.timing.dt_ctrl)
        while not act.done:
            x = self._sim_step(act(x))
        self._sens_offset[0] = (act.go_right.th_lim + act.go_left.th_lim) / 2

        # Set current state
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        x = self._qsoc.snd_rcv(a)
        x[:self._sens_offset.shape[0]] -= self._sens_offset
        return x

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._calibrate()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
