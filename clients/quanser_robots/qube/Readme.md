Qube environment
================

Simulation and control environment for the Quanser Qube (Furuta Pendulum).

To see Qube in action, start the simulated swing-up demo

    python3 quanser_robots/qube/examples/swing-up.py


Package contents
----------------
1. `model.pdf` physical model description
2. `base.py` common functionality
3. `qube.py` simulated environment
4. `qube_rr.py` real robot environment
5. `ctrl.py` energy-based swing-up controller
6. `examples` example scripts


Controlling the real robot
--------------------------
To control the real robot,
you must be in the local network with the Qube control PC.
There are two ways to get into the same network:

1. Ethernet connection.
   Assume Qube control PC has the IP address `192.168.2.17`
   (see `quanser_robots/qube/__init__.py`);
   then your PC should have the IP e.g. `192.168.2.1` to be in the same network.
   Try `ping 192.168.2.1` on the Windows PC to check the setup.
   
2. Wireless connection. If your laptop doesn't have an Ethernet port,
   you can connect to the Windows PC via WiFi.
   - For that, connect the Windows PC to the Internet
     (you might need to modify Network adapter settings for IPv4,
     set everything to automatic there).
   - Go to Start -> Settings -> Network & Internet -> Mobile hotspot
     and turn on the mobile hotspot.
   - Connect to the WiFi network that you just created.
     Adjust the IP in `quanser_robots/qube/__init__.py`.


To run the swing-up demo on the real robot, perform the following steps:

1. Start the control server on the control PC (pick frequency 100, 250, or 500)

        quarc_run -r Desktop\servers\qube\quarc_py_bridge_qube_100.rt-win64

2. Launch the client on your computer (use matching freq e.g. 'QubeRR-100-v0')

        python3 quanser_robots/qube/examples/swing-up_rr.py

3. When you are done with experiments, shut down the control server

       quarc_run -q Desktop\servers\qube\quarc_py_bridge_qube_100.rt-win64


### Control loop
The canonical way of using the real robot environment:
    
    import gym
    from quanser_robots import GentlyTerminating
    env = GentlyTerminating(gym.make('QubeRR-v0'))
    ctrl = ...  # some function f: s -> a
    obs = env.reset()
    done = False
    while not done:
        act = ctrl(obs)
        obs, rwd, done, info = env.step(act)

Pay attention to the following important points:

- Reset the environment `env.reset()` right before running `env.step(act)`
  in a loop. If you forget to reset the environment and then send an action
  after some time of inactivity, you will get an outdated observation.

- Wrap the environment in the `GentlyTerminating` wrapper to ensure that
  a zero command is sent to the robot after an episode is finished.
  Qube always keeps executing the last command it received, which may damage
  the motor if constant voltage is applied for too long.
