import numpy as np
import warnings
from quanser_robots.common import Base, Timing, LabeledBox

np.set_printoptions(precision=6, suppress=True)
X_LIM = 0.814


class CartpoleBase(Base):

    def __init__(self, fs, fs_ctrl, stabilization=False):

        super(CartpoleBase, self).__init__(fs, fs_ctrl)
        self._state = None
        self._vel_filt = None

        self.safe_range = 0.1

        self.stabilization = stabilization
        self._x_lim = X_LIM / 2.      # [m]
        self.stabilization_th = 0.25  # [rad]

        act_max = np.array([24.0])
        state_max = np.array([self._x_lim, np.inf, np.inf, np.inf])
        sens_max = np.array([np.inf, np.inf])
        obs_max = np.array([self._x_lim, 1.0, 1.0, np.inf, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('x', 'theta'),
            low=-sens_max, high=sens_max, dtype=np.float32)

        self.state_space = LabeledBox(
            labels=('x', 'theta', 'x_dot', 'theta_dot'),
            low=-state_max, high=state_max, dtype=np.float32)

        self.observation_space = LabeledBox(
            labels=('x',  'sin_th', 'cos_th', 'x_dot', 'th_dot'),
            low=-obs_max, high=obs_max, dtype=np.float32)

        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)

        if self.stabilization:
            self.reward_range = (1 - np.cos(np.pi - self.stabilization_th), 2.)
        else:
            self.reward_range = (0., 2.)

        # Function to ensure that state and action constraints are satisfied:
        self._lim_act = ActionLimiter()

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _limit_act(self, action):
        if np.abs(action) > 24.:
            warnings.warn("Control signal a = {0:.2f} should be between -24V and 24V.".format(action))
        return np.clip(action, -24., 24.)

    def _rwd(self, x, a):

        x_c, th, _, _ = x
        rwd = -np.cos(th)

        # Normalize theta to [-pi, +pi]
        th = np.mod(th + np.pi, 2. * np.pi) - np.pi

        done = False
        if self.stabilization:
            done = np.abs(th - np.sign(th) * np.pi) > self.stabilization_th
        done = done or np.abs(x_c) > self._x_lim - self.safe_range

        return np.float32(rwd) + 1., done

    def _observation(self, state):
        """
        A observation is provided given the internal state.

        :param state: (x, theta, x_dot, theta_dot)
        :type state: np.array
        :return: (x, sin(theta), cos(theta), x_dot, theta_dot)
        :rtype: np.array
        """
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])


class CartpoleDynamics:

    def __init__(self, long=False):

        self.g = 9.81              # Gravitational acceleration [m/s^2]
        self.eta_m = 1.            # Motor efficiency  []
        self.eta_g = 1.            # Planetary Gearbox Efficiency []
        self.Kg = 3.71             # Planetary Gearbox Gear Ratio
        self.Jm = 3.9E-7           # Rotor inertia [kg.m^2]
        self.r_mp = 6.35E-3        # Motor Pinion radius [m]
        self.Rm = 2.6              # Motor armature Resistance [Ohm]
        self.Kt = .00767           # Motor Torque Constant [N.zz/A]
        self.Km = .00767           # Motor Torque Constant [N.zz/A]
        self.mc = 0.37             # Mass of the cart [kg]

        if long:
            self.mp = 0.23         # Mass of the pole [kg]
            self.pl = 0.641 / 2.   # Half of the pole length [m]

            self.Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
            self.Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.3
            self.scale = np.array([0.45, 1.])
        else:
            self.mp = 0.127        # Mass of the pole [kg]
            self.pl = 0.3365 / 2.  # Half of the pole length [m]
            self.Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
            self.Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.5
            self.scale = np.array([1., 1.])

        # Compute Inertia:
        self.Jp = self.pl ** 2 * self.mp / 3.   # Pole inertia [kg.m^2]
        self.Jeq = self.mc + (self.eta_g * self.Kg ** 2 * self.Jm) / (self.r_mp ** 2)

    def __call__(self, s, v_m):
        x, theta, x_dot, theta_dot = s

        # Compute force acting on the cart:
        F = np.asscalar((self.eta_g * self.Kg * self.eta_m * self.Kt) / (self.Rm * self.r_mp) *
                        (-self.Kg * self.Km * x_dot / self.r_mp + self.eta_m * v_m))

        # Compute acceleration:
        A = np.array([[self.mp + self.Jeq, +self.mp * self.pl * np.cos(theta)],
                      [+self.mp * self.pl * np.cos(theta), self.Jp + self.mp * self.pl ** 2]])

        b = np.array([F - self.Beq * x_dot - self.mp * self.pl * np.sin(theta) * theta_dot ** 2,
                      0. - self.Bp * theta_dot - self.mp * self.pl * self.g * np.sin(theta)])

        s_ddot = np.linalg.solve(A, b)
        return s_ddot


class CartpoleInverseDynamics:
    def __init__(self, long_pole=False):
            # Construct the dynamics model:
            self._dyn = CartpoleDynamics(long=long_pole)

    def __call__(self, s, u):

            if s.size == 5:
                # Observation as input:
                x, sin_theta, cos_theta, x_dot, theta_dot = s

            elif s.size == 4:
                # State as input:
                x, theta, x_dot, theta_dot = s
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)

            else:
                raise ValueError

            # Compute dynamics model:
            h_11 = self._dyn.Jeq + self._dyn.mp
            h_12 = self._dyn.mp * self._dyn.pl * cos_theta
            h_22 = self._dyn.Jp + self._dyn.mp * self._dyn.pl ** 2

            c_1 = + self._dyn.mp * self._dyn.pl * sin_theta * theta_dot ** 2
            g_2 = + self._dyn.mp * self._dyn.pl * self._dyn.g * sin_theta
            f_1 = - self._dyn.Beq * x_dot
            f_2 = - self._dyn.Bp * theta_dot

            # Compute inverse dynamics model:
            h_tilde = h_11 - h_12 / h_22 * h_12
            c_tilde = c_1
            g_tilde = - h_12 / h_22 * g_2
            f_tilde = f_1 - h_12 / h_22 * f_2

            # Compute the desired force acting on the cart
            force_cart = h_tilde * u + c_tilde + g_tilde - f_tilde

            # Compute motor torque and corresponding voltage:
            tau = self._dyn.r_mp / (self._dyn.eta_g * self._dyn.Kg) * force_cart
            v = self._dyn.Rm / (self._dyn.eta_m * self._dyn.Kt) * tau + self._dyn.Km * self._dyn.Kg / self._dyn.r_mp * x_dot

            # Add an additional voltage to overcome static friction:
            # v = v + np.sign(v) * 0.5 * np.exp(-np.abs(x_dot/0.01)
            return v


class ActionLimiter:

    def __init__(self):
        pass

    def _joint_lim_violation_force(self, x):
        pass

    def __call__(self, x, a):
        return a






