"""
Analytic real-robot swing-up controller with trajectory visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from quanser_robots import GentlyTerminating
from quanser_robots.qube import SwingUpCtrl
import time

plt.style.use('seaborn')


env = GentlyTerminating(gym.make('QubeRR-100-v0'))

ctrl = SwingUpCtrl()
obs = env.reset()
s_all, a_all = [], []
done = False
t0 = time.perf_counter()
n = 0
while not done:
    env.render()
    act = ctrl(obs)
    obs, rwd, done, info = env.step(act)
    s_all.append(info['s'])
    a_all.append(info['a'])
    n += 1
t1 = time.perf_counter()
print("freq = {}, time = {}".format(n / (t1-t0), t1-t0))
env.close()


fig, axes = plt.subplots(5, 1, figsize=(5, 8), tight_layout=True)

s_all = np.stack(s_all)
a_all = np.stack(a_all)

n_points = s_all.shape[0]
t = np.linspace(0, n_points * env.unwrapped.timing.dt_ctrl, n_points)
for i in range(4):
    state_labels = env.unwrapped.state_space.labels[i]
    axes[i].plot(t, s_all.T[i], label=state_labels, c='C{}'.format(i))
    axes[i].legend(loc='lower right')
action_labels = env.unwrapped.action_space.labels[0]
axes[4].plot(t, a_all.T[0], label=action_labels, c='C{}'.format(4))
axes[4].legend(loc='lower right')

axes[0].set_ylabel('ang pos [rad]')
axes[1].set_ylabel('ang pos [rad]')
axes[2].set_ylabel('ang vel [rad/s]')
axes[3].set_ylabel('ang vel [rad/s]')
axes[4].set_ylabel('voltage [V]')
axes[4].set_xlabel('time [seconds]')
plt.show()
