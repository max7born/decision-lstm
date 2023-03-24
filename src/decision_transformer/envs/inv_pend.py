import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, offset=.1):
        self.offset = offset
        self.step_nr = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        


    def step(self, a):
        reward = 1.0
        self.offset
        self.step_nr += 1
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = (not notdone) or (self.step_nr >=1000) 
        return ob, reward, done, {}

    def reset_model(self):
        self.step_nr = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-self.offset, high=self.offset)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-self.offset, high=self.offset)
        self.set_state(qpos, qvel)
        #print(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
