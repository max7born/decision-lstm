"""
This example shows how to change physics parameters upon environment reset.
"""

import gym
from quanser_robots import GentlyTerminating
from quanser_robots.qube import Parameterized

env = Parameterized(GentlyTerminating(gym.make('Qube-100-v0')))

# Show all adjustable physics parameters
print(env.params())

# Pass a dictionary of modified physics parameters upon environment reset
env.reset({'g': 10.0})
print(env.params())  # only the provided parameters are modified

# Upon reset, previous parameters are used and not the default ones
env.reset({'Rm': 9.0})
print(env.params())
