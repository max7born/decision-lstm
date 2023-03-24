import time
import numpy as np
import gym
from quanser_robots import GentlyTerminating


class MetronomCtrl:
    """Rhythmically swinging metronome."""

    def __init__(self, u_max=10.0, f=0.5, dur=5.0):
        """
        Constructor

        :param u_max: maximum voltage
        :param f: frequency in Hz
        :param dur: task finishes after `dur` seconds

        """
        self.done = False
        self.u_max = u_max
        self.f = f
        self.dur = dur
        self.start_time = None

    def __call__(self, _):
        """
        Calculates the actions depending on the elapsed time.

        :return: scaled sinusoidal voltage
        """
        if self.start_time is None:
            self.start_time = time.time()
        t = time.time() - self.start_time
        if not self.done and t > self.dur:
            self.done = True
            u = 0.0
        else:
            u = 0.25 * self.u_max * np.sin(2 * np.pi * self.f * t)
        return [u]


def main():
    env = GentlyTerminating(gym.make('CartpoleRR-v0'))

    print("\n\nMetronom Example:")
    ctrl = MetronomCtrl()

    print("\tCalibrate the System:", end="")
    obs = env.reset()
    print("\tDone")

    print("\tSwing Pendulum:", end="")
    while not ctrl.done:

        # env.render()
        act = ctrl(obs)
        obs, _, _, _ = env.step(act)

    print("\t\t\tDone")

    print("\tReset the System:", end="")
    obs = env.reset()
    print("\t\tDone")

    env.close()


if __name__ == "__main__":
    main()
