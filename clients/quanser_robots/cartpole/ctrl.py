import numpy as np
import time
from quanser_robots.cartpole.base import CartpoleDynamics
from pynput.mouse import Controller


class PDCtrl:
    """
    PD controller for the cartpole environment.
    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, s_des=np.zeros(4), tol=5e-4):
        self.K = K if K is not None else np.array([20.0, 0.0, 0.0, 0.0])

        self.done = False
        self.s_des = s_des
        self.tol = tol

    def __call__(self, s):

        # Compute the voltage:
        err = self.s_des - s
        v = np.dot(self.K.transpose(), err)

        # Check for completion:
        if np.sum(err**2) <= self.tol:
            self.done = True

        return np.array([v], dtype=np.float32)


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, s_init, positive=True):
        self.done = False
        self.success = False
        self.x_init = s_init[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1

        self.sign = 1 if positive else -1
        self.u_max = self.sign * np.array([1.5])

        self._t_init = False
        self._t0 = 0.0
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, s):
        x, _, xd, _ = s

        # Initialize the time:
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

        # Compute voltage:
        if (time.time() - self._t0) < self._t_min:
            u = self.u_max

        elif np.abs(xd) < self.xd_max:
            u = np.zeros(1)
            self.success = True
            self.done = True

        elif (time.time() - self._t0) > self._t_max:
            u = np.zeros(1)
            self.success = False
            self.done = True

        else:
            u = self.u_max

        return u


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class SwingUpCtrl:
    """Swing up and balancing controller"""

    def __init__(self, long=False, mu=18.5, u_max=18, v_max=24):
        self.dynamics = CartpoleDynamics(long=long)
        self.pd_control = False
        self.pd_activated = False
        self.done = False

        self.u_max = u_max
        self.v_max = v_max

        if long:
            self.K_pd = np.array([-41.833, 189.8393, -47.8483, 28.0941])
        else:

            # Simulation:
            self.k_p = 8.5                                              # Standard Value: 08.5
            self.k_e = 19.5                                             # Standard Value: 19.5 (500Hz)
            self.k_d = 0.0                                              # Standard Value: XXXX
            # self.K_pd = np.array([+41.83, -173.44, +46.14, -16.27])   # Standard Value: [+41.8, -173.4, +46.1, -16.2]
            self.K_pd = np.array([+41.0, -200.0, +55.0, -16.00])        # Standard Value: [+41.8, -173.4, +46.1, -16.2]

    def __call__(self, state):
        x, sin_theta, cos_theta, x_dot, theta_dot = state
        alpha, theta = get_angles(sin_theta, cos_theta)

        dyna = self.dynamics
        Mp = self.dynamics.mp
        pl = self.dynamics.pl
        Jp = self.dynamics.Jp

        Ek = Jp/2. * theta_dot**2
        Ep = Mp * dyna.g * pl * (1 - np.cos(theta))     # E(0) = 0., E(pi) = E(-pi) = 2 mgl
        Er = 2. * Mp * dyna.g * pl                      # = 2 mgl

        # since we use theta zero in the rest position, we have -theta dot and
        if np.abs(alpha) < 0.1745 or self.pd_control:

            if not self.pd_activated:
                pass
                # print("PD Control")

            self.pd_activated = True
            u = np.matmul(self.K_pd, (np.array([x, alpha, x_dot, theta_dot])))

        else:
            self.u_max = 180
            u = np.clip(self.k_e * (Ek + Ep - Er) * np.sign(theta_dot * np.cos(theta))
                        + self.k_p * (0.0 - x)
                        + self.k_d * (0.0 - x_dot), -self.u_max, self.u_max)

            if self.pd_activated:
                self.done = True
                self.pd_activated = False

        Vm = (dyna.Jeq * dyna.Rm * dyna.r_mp * u)/(dyna.eta_g * dyna.Kg * dyna.eta_m * dyna.Kt)\
              + dyna.Kg * dyna.Km * x_dot / dyna.r_mp

        Vm = np.clip(Vm, -self.v_max, self.v_max)
        return [Vm, u]


class SwingDownCtrl:
    """Swing down and keep in center and resting position."""

    def __init__(self, long=False, mu=14., epsilon=1E-4):
        self.dynamics = CartpoleDynamics(long=long)
        self.mu = mu
        self.pd_control = False
        self.pd_activated = False
        self.done = False
        self.epsilon = epsilon

        self.K = np.array([0., 0.1, 0.1, 0.1])

    def __call__(self, state):
        x, sin_theta, cos_theta, x_dot, theta_dot = state
        alpha, theta = get_angles(sin_theta, cos_theta)

        dyna = self.dynamics
        Mp = self.dynamics.mp
        pl = self.dynamics.pl
        Jp = pl**2 * Mp / 3.

        Ek = Jp/2. * theta_dot**2
        Ep = Mp * dyna.g * pl * (1 - np.cos(theta))
        Er = 2*Mp*dyna.g*pl     # ==0

        # since we use theta zero in the rest position, we have -theta dot and
        if np.abs(theta) < 0.025 or self.pd_control:
            if not self.pd_activated:
                self.pd_activated = True

            u = np.matmul(self.K, (-np.array([x, theta, x_dot, theta_dot])))

        else:
            umax = 10
            mu = self.mu * np.sqrt(np.abs(np.clip(theta_dot, -1, 1)))
            u = -np.clip(mu/20. * (Ek+Ep-Er) * np.sign(theta_dot * np.cos(theta)), -umax, umax)
            if self.pd_activated:
                print("energy")
                self.done = True
                self.pd_activated = False

        error = np.mean(np.square(np.array([x,theta,x_dot,theta_dot])))
        if error < self.epsilon:
            print("Resting position")
            self.done = True

        Vm = (dyna.Jeq * dyna.Rm * dyna.r_mp*u)/(dyna.eta_g * dyna.Kg * dyna.eta_m * dyna.Kt)\
              + dyna.Kg * dyna.Km * x_dot / dyna.r_mp

        Vm = np.clip(Vm, -24, 24)
        return [Vm, u]


class MouseCtrl:

    def __init__(self):
        input("Set the mouse more or less in the center of your screen and press Enter to activate the controller")
        self.mouse = Controller()
        self.x = self.mouse.position[0]

    def __call__(self, *args, **kwargs):
        Vm = np.clip((self.mouse.position[0] - self.x)/2000.*24., -24., +24.)
        return [Vm, 0.]
