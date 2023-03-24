import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox, Timing

np.set_printoptions(precision=6, suppress=True)


class QubeBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(QubeBase, self).__init__()
        self._state = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        safety_th_lim = 1.0
        act_max = np.array([5.0])
        state_max = np.array([2.0, 4.0 * np.pi, 30.0, 40.0])
        sens_max = np.array([2.3, np.inf])
        obs_max = np.array([1.0, 1.0, 1.0, 1.0, state_max[2], state_max[3]])

        # Spaces
        self.state_space = LabeledBox(
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Function to ensure that state and action constraints are satisfied
        self._lim_act = ActionLimiter(self.state_space,
                                      self.action_space,
                                      safety_th_lim)

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _sim_step(self, a):
        """
        Update internal state of simulation and return an estimate thereof.

        :param a: action
        :return: state
        """
        raise NotImplementedError

    def _ctrl_step(self, a):
        x = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = self._lim_act(x, a)
            x = self._sim_step(a_cmd)
        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, x, a):
        th, al, thd, ald = x
        al_mod = al % (2 * np.pi) - np.pi
        cost = al_mod**2 + 5e-3*ald**2 + 1e-1*th**2 + 2e-2*thd**2 + 3e-3*a[0]**2
        done = not self.state_space.contains(x)
        rwd = np.exp(-cost) * self.timing.dt_ctrl
        return np.float32(rwd), done

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        assert a is not None, "Action should be not None"
        assert isinstance(a, np.ndarray), "The action should be a ndarray"
        assert np.all(not np.isnan(a)), "Action NaN is not a valid action"
        assert a.ndim == 1, "The action = {1} must be 1d but the input is {0:d}d".format(a.ndim, a)

        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        obs = np.float32([np.cos(self._state[0]), np.sin(self._state[0]),
                          np.cos(self._state[1]), np.sin(self._state[1]),
                          self._state[2], self._state[3]])
        return obs, rwd, done, {'s': self._state, 'a': act}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):
        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = \
            action_space.high[0] / (self._th_lim_max - self._th_lim_min)
        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)

    def _joint_lim_violation_force(self, x):
        th, _, thd, _ = x
        up = self._relu(th-self._th_lim_max) - self._relu(th-self._th_lim_min)
        dn = -self._relu(-th-self._th_lim_max)+self._relu(-th-self._th_lim_min)
        if (th > self._th_lim_min and thd > 0.0 or
                th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0
        return force

    def __call__(self, x, a):
        force = self._joint_lim_violation_force(x)
        return self._clip(force if force else a)


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(self):
        # Gravity
        self.g = 9.81

        # Motor
        self.Rm = 8.4    # resistance
        self.km = 0.042  # back-emf constant (V-s/rad)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Dr = 5e-6   # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.024  # mass (kg)
        self.Lp = 0.129  # length (m)
        self.Dp = 1e-6   # viscous damping (N-m-s/rad), original: 0.0005

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr ** 2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp ** 2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr ** 2
        self._c[1] = 0.25 * self.Mp * self.Lp ** 2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop('_c')
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, s, u):
        th, al, thd, ald = s
        voltage = u[0]

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * np.sin(al) ** 2
        b = self._c[2] * np.cos(al)
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * np.sin(2 * al) * thd * ald \
            - self._c[2] * np.sin(al) * ald * ald
        c1 = -0.5 * self._c[1] * np.sin(2 * al) * thd * thd \
            + self._c[4] * np.sin(al)
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd


class Parameterized(gym.Wrapper):
    """
    Allow passing new dynamics parameters upon environment reset.
    """
    def params(self):
        return self.unwrapped.dyn.params

    def step(self, action):
        return self.env.step(action)

    def reset(self, params=None):
        if params:
            self.unwrapped.dyn.params = params
        return self.env.reset()
