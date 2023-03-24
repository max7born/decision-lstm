import socket
import struct
import numpy as np
from scipy import signal
import gym
from gym import spaces
from gym.utils import seeding


class QSocket:
    """
    Class for communication with Quarc.
    """
    def __init__(self, ip, x_len, u_len):
        """
        Prepare socket for communication.
        :param ip: IP address of the Windows PC
        :param x_len: number of measured state variables to receive
        :param u_len: number of control variables to send
        """
        self._x_fmt = '>' + x_len * 'd'
        self._u_fmt = '>' + u_len * 'd'
        self._buf_size = x_len * 8  # 8 bytes for each double
        self._port = 9095  # fixed in Simulink model
        self._ip = ip
        self._soc = None

    def snd_rcv(self, u):
        """
        Send u and receive x.
        :param u: control vector
        :return: x: vector of measured states
        """
        self._soc.send(struct.pack(self._u_fmt, *u))
        data = self._soc.recv(self._buf_size)
        return np.array(struct.unpack(self._x_fmt, data), dtype=np.float32)

    def open(self):
        if self._soc is None:
            self._soc = socket.socket()
            self._soc.connect((self._ip, self._port))

    def close(self):
        if self._soc is not None:
            self._soc.close()
            self._soc = None

    def is_open(self):
        open = True
        if self._soc is None:
            open = False

        return open




class SymmetricBoxSpace:
    """
    Generic real-valued box space with symmetric boundaries.
    """
    def __init__(self, bound: np.ndarray, labels: tuple):
        self.bound_lo = -bound
        self.bound_up = bound
        self.labels = labels
        self.dim = len(labels)

    def project(self, ele: np.ndarray):
        return np.clip(ele, self.bound_lo, self.bound_up)


class VelocityFilter:
    """
    Discrete velocity filter derived from a continuous one.
    """
    def __init__(self, x_len, num=(50, 0), den=(1, 50), dt=0.002, x_init=None):
        """
        Initialize discrete filter coefficients.
        :param x_len: number of measured state variables to receive
        :param num: continuous-time filter numerator
        :param den: continuous-time filter denominator
        :param dt: sampling time interval
        :param x_init: initial observation of the signal to filter
        """
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        if x_init is None:
            self.z = np.zeros((max(len(self.a), len(self.b)) - 1, x_len),
                              dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init):
        """
        This method can be used to set the initial state of the velocity filter.
        This is useful when the initial (position) observation
        has been retrieved and it is non-zero.
        Otherwise the filter would assume a very high velocity.
        :param x_init: initial observation
        """
        assert isinstance(x_init, np.ndarray)
        # Get the initial condition of the filter
        zi = signal.lfilter_zi(self.b, self.a)  # dim = order of the filter = 1
        # Set the filter state
        self.z = np.outer(zi, x_init)

    def __call__(self, x):
        xd, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return xd.ravel()


class LabeledBox(spaces.Box):
    """
    Adds `labels` field to gym.spaces.Box to keep track of variable names.
    """
    def __init__(self, labels, **kwargs):
        super(LabeledBox, self).__init__(**kwargs)
        assert len(labels) == self.high.size
        self.labels = labels


class GentlyTerminating(gym.Wrapper):
    """
    This env wrapper sends zero command to the robot when an episode is done.
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            self.env.step(np.zeros(self.env.action_space.shape))
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class Timing:
    def __init__(self, fs, fs_ctrl):
        fs_ctrl_min = 50.0  # minimal control rate
        assert fs_ctrl >= fs_ctrl_min, \
            "control frequency must be at least {}".format(fs_ctrl_min)
        self.n_sim_per_ctrl = int(fs / fs_ctrl)
        assert fs == fs_ctrl * self.n_sim_per_ctrl, \
            "sampling frequency must be a multiple of the control frequency"
        self.dt = 1.0 / fs
        self.dt_ctrl = 1.0 / fs_ctrl
        self.render_rate = int(fs_ctrl)


class PhysicSystem:

    def __init__(self, dt, **kwargs):
        self.dt = dt
        for k in kwargs:
            setattr(self, k, kwargs[k])
            setattr(self, k + "_dot", 0.)

    def add_acceleration(self, **kwargs):
        for k in kwargs:
            setattr(self, k + "_dot", getattr(self, k + "_dot") + self.dt * kwargs[k])
            setattr(self, k, getattr(self, k) + self.dt * getattr(self, k + "_dot"))

    def get_state(self, entities_list):
        ret = []
        for k in entities_list:
            ret.append(getattr(self, k))
        return np.array(ret)


class Base(gym.Env):

    def __init__(self, fs, fs_ctrl):
        """

        :param fs: frequency of observation
        :type fs: float
        :param fs_ctrl: frequency of control
        :type fs_ctrl: float
        """
        super(Base, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Spaces
        self.sensor_space = None
        self.state_space = None
        self.observation_space = None
        self.action_space = None
        self.reward_range = None

        # Function to ensure that state and action constraints are satisfied
        self._lim_act = None

        self.reward_range = None
        # Function to ensure that state and action constraints are satisfied:

        self.done = False
        self.seed()

    def _limit_act(self, action):
        raise NotImplementedError

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _sim_step(self, a):
        """
        Update internal state of simulation and return an estimate thereof.

        :param a: action
        :type a: np.array
        :return: state
        :rtype: np.array
        """
        raise NotImplementedError

    def _ctrl_step(self, a):
        """
        Control for a number of steps accordingly to the step frequency.

        :param a: action
        :type a: np.array
        :return: observed state, and action used
        :rtype: tuple
        """
        x = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = self._lim_act(x, a)
            x = self._sim_step(a_cmd)
        return x, a_cmd

    def _rwd(self, x, a):
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Set the random seed.

        :param seed: random seed
        :type seed: int
        :return: list
        """
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _observation(self, state):
        raise NotImplementedError

    def step(self, a):
        assert a is not None, "Action should be not None"
        assert isinstance(a, np.ndarray), "The action should be a ndarray but is of type '{0}'".format(type(a))
        assert np.all(not np.isnan(a)), "Action NaN is not a valid action"
        assert a.ndim == 1, "The action = {1} must be 1d but the input is {0:d}d".format(a.ndim, a)

        rwd, done = self._rwd(self._state, a)

        self.done = self.done or done #TODO: use in the future this information to prevent action after one done=True
        if not done:
            self._state, act = self._ctrl_step(a)
        else:
            self._state, act = self._ctrl_step(a * 0.)

        obs = self._observation(self._state)
        return obs, rwd, done, {'s': self._state, 'a': act}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class Simulation(Base):

    def __init__(self, fs, fs_ctrl, dynamics, entities, filters, initial_distr):

        super(Simulation, self).__init__(fs, fs_ctrl)
        self.entities = entities
        self.entities_dot = [e + "_dot" for e in self.entities]
        self.filters = {}
        self.filter_init = filters
        self.initial_distr = initial_distr

        self._sim_state = None
        self.viewer = None
        self.physics = None
        self._dynamics = dynamics

    def _calibrate(self):

        initial_values = {}
        for e in self.entities:
            v = self.initial_distr[e]()
            initial_values[e] = v
            self.filters[e] = self.filter_init[e](v)

        self._state = np.array([initial_values[e] for e in self.entities]   # system's variable
                               + [0. for _ in self.entities])               # system's velocities

        self.physics = PhysicSystem(self.timing.dt, **initial_values)
        self._sim_state = self.physics.get_state(self.entities + self.entities_dot)

    def _sim_step(self, a):
        # Add a bit of noise to action for robustness
        a_noisy = a + 1e-6 * np.float32(np.random.randn(self.action_space.shape[0]))
        acceleration = {e: v for e, v in zip(self.entities, self._dynamics(self._sim_state, a_noisy))}

        self.physics.add_acceleration(**acceleration)

        self._sim_state = self.physics.get_state(self.entities + self.entities_dot)
        current_state = self.physics.get_state(self.entities)
        velocities = np.array([self.filters[e](self._sim_state[i:i+1]) for i, e in enumerate(self.entities)]).ravel()
        return np.concatenate([current_state, velocities])

    def reset(self):
        self._calibrate()
        return self.step(np.array([0.0]))[0]


class Logger:
    """
    For debugging purposes. Saves a numpy files with a trajectory.
    """
    def __init__(self, env):
        self.env = env
        self.obs_log = []
        self.act_log = []

    def reset(self):
        s = self.env.reset()
        self.obs_log.append(s)
        return s

    def step(self,a):
        s, r, d, i = self.env.step(a)
        self.obs_log.append(s)
        self.act_log.append(a)
        return s, r, d, i

    def save(self, path=""):
        np.save(path + "act_log.npy", self.act_log)
        np.save(path + "obs_log.npy", self.obs_log)
        self.obs_log = []
        self.act_log = []

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class NoFilter:

    def __init__(self, x_init=0., dt=0.002):
        self.x = x_init
        self.dt = dt

    def __call__(self, *args, **kwargs):
        ret = (np.array(args) - self.x)/self.dt
        self.x = np.array(args)
        return ret
