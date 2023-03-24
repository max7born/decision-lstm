Quanser Robots
==============

Simulated environments and real robot interfaces for a set of Quanser platforms.


Getting Started
---------------
This package is compatible with Python 3.6.5. To install the it, execute

    pip3 install -e .

To confirm that the setup was successful, launch a Python3 console and run
    
    import gym
    import quanser_robots
    env = gym.make('Qube-v0')
    env.reset()
    env.render()

If installation worked well, proceed to the robot-specific documentation

- [Qube](quanser_robots/qube/Readme.md)
- [Ball Balancer](quanser_robots/ball_balancer/Readme.md)
- [Levitation](quanser_robots/levitation/Readme.md)
- [Cartpole](quanser_robots/cartpole/Readme.md)
- [Double Pendulum](quanser_robots/double_pendulum/Readme.md)

If you are getting errors during installation that you do not know how to fix,
check below whether the requirements are satisfied, and if necessary follow
the [detailed installation instructions](Install.md).


Requirements
------------
The main requirement comes from the graphics library `vpython`
which is used for rendering 3D environments.
It requires Python >= 3.5.3 (preferably Pyton 3.6.5).
Note that the default version of Python on Ubuntu 16.04 is Python 3.5.2,
so visualization will not work with it.
You can still use the environments though, just don't call `env.render()`.


Developers and Maintainers
--------------------------
Awesome stuff developed by
- Boris Belousov
- Fabio Muratore
- Hany Abdulsamad
- Jonas Eschmann
- Robin Menzenbach
- Christian Eilers
- Samuele Tosatto
- Michael Lutter

Add your name if you contributed, so that we know whom to blame.
