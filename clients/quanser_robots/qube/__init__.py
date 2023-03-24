from gym.envs.registration import register
from .ctrl import SwingUpCtrl, PDCtrl
from .base import Parameterized

# Simulation
register(
    id='Qube-100-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=600,
    kwargs={'fs': 200.0, 'fs_ctrl': 100.0}
)

register(
    id='Qube-250-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=1500,
    kwargs={'fs': 250.0, 'fs_ctrl': 250.0}
)

register(
    id='Qube-500-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=3000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0}
)

# Real robot
register(
    id='QubeRR-100-v0',
    entry_point='quanser_robots.qube.qube_rr:Qube',
    max_episode_steps=600,
    kwargs={'ip': '192.168.2.17', 'fs_ctrl': 100.0}
)

register(
    id='QubeRR-250-v0',
    entry_point='quanser_robots.qube.qube_rr:Qube',
    max_episode_steps=1500,
    kwargs={'ip': '192.168.2.17', 'fs_ctrl': 250.0}
)

register(
    id='QubeRR-500-v0',
    entry_point='quanser_robots.qube.qube_rr:Qube',
    max_episode_steps=3000,
    kwargs={'ip': '192.168.2.17', 'fs_ctrl': 500.0}
)
