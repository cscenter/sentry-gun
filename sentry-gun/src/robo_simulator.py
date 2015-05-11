__author__ = 'vasdommes'

import numpy as np
from numpy import sin, cos, pi
from ballistics import final_coords


class RoboSimulator:
    def __init__(self, robot_coords, v, g, alpha_0, phi_0, gun_length=0.0,
                 randomize=None,
                 randomize_v=None, randomize_alpha=None, randomize_phi=None):
        self.x0, self.y0, self.z0 = robot_coords
        self.v = v
        self.g = g

        self.alpha = self.alpha_0 = alpha_0
        self.phi = self.phi_0 = phi_0

        self.gun_length = gun_length

        self.randomize = randomize
        self.randomize_v = randomize_v
        self.randomize_alpha = randomize_alpha
        self.randomize_phi = randomize_phi

        if self.randomize is None:
            self.randomize = lambda x: np.random.uniform(x * 0.99, x * 1.01)

        if self.randomize_v is None:
            self.randomize_v = self.randomize
        if self.randomize_alpha is None:
            self.randomize_alpha = self.randomize
        if self.randomize_phi is None:
            self.randomize_phi = self.randomize


    def rotate_and_shoot(self, d_alpha=0.0, d_phi=0.0):
        self.alpha += d_alpha
        self.phi += d_phi

        # Randomize initial parameters
        alpha = self.randomize_alpha(self.alpha)
        phi = self.randomize_phi(self.phi)
        v = self.randomize_v(self.v)
        g = self.g

        # Take into account length of gun
        x = self.x0 + self.gun_length * cos(alpha) * cos(phi)
        y = self.x0 + self.gun_length * cos(alpha) * sin(phi)
        z = self.z0 + self.gun_length * sin(alpha)

        self.shot_x, self.shot_y = final_coords((x, y, z),
                                                (alpha, phi), v, g)
        return self.shot_x, self.shot_y

    def last_shot_coords(self):
        if self.shot_x is None or self.shot_y is None:
            ret = False
            # raise ValueError('Please shoot before asking coords')
        ret = True
        return ret, (self.shot_x, self.shot_y)