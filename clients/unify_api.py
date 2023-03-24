import numpy as np

import gym
import quanser_robots

# env = gym.make('Pendulum-v0')
# state = env.reset()
# print(state.shape)
# print("List 1d = {0}".format(env.step([1])[0]))
# print("List 2d = {0}".format(env.step([[1]])[0]))
# print("List 3d = {0}".format(env.step([[[1]]])[0]))
# print(env.step([[[1, 2]]])[0])

env_list = ['DoublePendulum-v0', 'Qube-v0', 'BallBalancerSim-v0', 'CartpoleSwingShort-v0', 'Levitation-v0']

for name in env_list:

    print("\n\n{0}".format(name))
    env = gym.make(name)
    env.reset()

    try:
        print("1d Input:", end="\t")
        tmp = env.step(np.array([1]))[0]
        print("type = {0}\t\t{1}".format(type(tmp), tmp))

    except AssertionError:
        print("Assertion Error")

    except ValueError:
        print("Value Error ")

    try:
        print("2d Input:", end="\t")
        tmp = env.step(np.array([[1]]))[0]
        print("type = {0}\t\t{1}".format(type(tmp), tmp))

    except AssertionError:
        print("Assertion Error")

    except ValueError:
        print("Value Error ")

    try:
        print("3d Input:", end="\t")
        tmp = env.step(np.array([[[1]]]))[0]
        print("type = {0}\t\t{1}".format(type(tmp), tmp))

    except AssertionError:
        print("Assertion Error")

    except ValueError:
        print("Value Error ")

    try:
        print("List Input:", end="\t")
        tmp = env.step([1])[0]
        print("type = {0}\t\t{1}".format(type(tmp), tmp))

    except AssertionError:
        print("Assertion Error")

    except ValueError:
        print("Value Error ")


