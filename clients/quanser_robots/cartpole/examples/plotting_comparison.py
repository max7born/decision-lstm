import numpy as np
import matplotlib as mp
mp.use("Qt5Agg")
mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load Data:
    sim_data = np.load('saved_real_data.npz')

    fig = plt.figure(1, figsize=(11, 15))
    fig.subplots_adjust(left=0.07, bottom=0.045, right=0.98, top=0.955, wspace=0.01, hspace=0.55)

    state_descriptor = ["Cart Position", "Pendulum Angle", "Cart Velocity", "Pendulum Velocity"]
    unit = [r"x [m]", r"$\theta$ [rad]", r"$\dot{x}$ [m/s]", r"$\dot{\theta}$ [rad/s]"]
    time = np.linspace(0.0, float(sim_data['s'].shape[1]) / 500., sim_data['s'].shape[1])

    # Plot state:
    for i in range(4):
        ax = fig.add_subplot(5, 1, i+1)
        # ax.set_title(state_descriptor[i])
        ax.set_ylabel(unit[i])
        ax.get_yaxis().set_label_coords(-0.05, 0.5)
        ax.set_xlabel("Time [s]")
        ax.plot(time, sim_data['s'][:, :, i].transpose(), c="r")

    # Plot Action:
    ax = fig.add_subplot(5, 1, 5)
    # ax.set_title("Action")
    ax.set_ylabel("Motor Voltage [V]")
    ax.get_yaxis().set_label_coords(-0.05, 0.5)
    ax.set_xlabel("Time [s]")
    ax.plot(time[:-1], sim_data['a'][:, :, 0].transpose(), c="r")


    plt.show()

