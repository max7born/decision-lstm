import numpy as np
from quanser_robots.common import VelocityFilter, PhysicSystem, Simulation, Timing, NoFilter
from quanser_robots.cartpole.base import CartpoleBase, X_LIM, CartpoleDynamics


class Cartpole(Simulation, CartpoleBase):

    def __init__(self, fs, fs_ctrl, long_pole=False, **kwargs):

        wcf = 62.8318
        zetaf = 0.9

        timing = Timing(fs, fs_ctrl)

        if kwargs.get('stabilization', False):
            theta_init = lambda: self._np_random.choice([self._np_random.uniform(-np.pi, -np.pi+0.1),
                                                         self._np_random.uniform(np.pi-0.1, np.pi)])

            filter = lambda init_val: NoFilter(x_init=init_val,dt=timing.dt)

        else:
            theta_init = lambda: 0.01 * self._np_random.uniform(-np.pi, np.pi)
            filter = lambda init_val: NoFilter(x_init=init_val, dt=timing.dt)
            # filter = lambda init_val: VelocityFilter(1, num=(wcf**2, 0), den=(1, 2*wcf*zetaf, wcf**2), x_init=np.array([init_val]), dt=timing.dt)

        Simulation.__init__(self, fs, fs_ctrl,
                            dynamics=CartpoleDynamics(long=long_pole),
                            entities=['x', 'theta'],
                            filters={'x': filter, 'theta': filter},
                            initial_distr={'x': lambda: 0., 'theta': theta_init})

        CartpoleBase.__init__(self, fs, fs_ctrl, **kwargs)

        # Transformations for the visualization:
        self.cart_trans = None
        self.pole_trans = None
        self.track = None
        self.axle = None

    def _sim_step(self, a):
        pos = Simulation._sim_step(self, a)

        # Normalize the angle from -pi to +pi:
        pos[1] = np.mod(pos[1] + np.pi, 2. * np.pi) - np.pi
        return pos

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = X_LIM
        scale = screen_width/world_width

        cart_y = 100  # Top of the cart
        pole_width = scale * 0.01
        pole_len = scale * self._dynamics.pl
        cart_width = scale * 0.1
        cart_height = scale * 0.05

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cart_width/2, cart_width/2, cart_height/2, -cart_height/2
            axle_offset = cart_height/4.0

            # Plot cart:
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # Plot pole:
            l, r, t, b = -pole_width/2,pole_width/2,pole_len-pole_width/2,-pole_width/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8, .6, .4)
            self.pole_trans = rendering.Transform(translation=(0, axle_offset))

            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # Plot axle:
            self.axle = rendering.make_circle(pole_width/2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            # Plot track:
            self.track = rendering.Line((0,cart_y), (screen_width,cart_y))
            self.track.set_color(0., 0., 0.)
            self.viewer.add_geom(self.track)

        # Update the visualization:
        if self._sim_state is None: return None

        x = self._sim_state
        cart_x = x[0] * scale + screen_width/2.0
        self.cart_trans.set_translation(cart_x, cart_y)
        self.pole_trans.set_rotation(x[1] - np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
