__author__ = 'vasdommes'

import numpy as np

from numpy import sin, cos, pi, sqrt
from scipy.optimize import curve_fit, leastsq
from ballistics import final_coords


class GunLayer:
    def __init__(self, rotate_and_shoot, get_coords, target=(0, 0),
                 gun_params=None):
        self.rotate_and_shoot = rotate_and_shoot
        self.get_coords = get_coords
        self.target = target # TODO

        self.history = []

        self.alpha = 0.0
        self.phi = 0.0

        if gun_params is None:
            self.gun_params = dict.fromkeys(
                ['x', 'y', 'z', 'v', 'g', 'alpha_0', 'phi_0', 'gun_length'])

    def shoot(self, d_alpha=0.0, d_phi=0.0):
        self.alpha += d_alpha
        self.phi += d_phi
        self.rotate_and_shoot(d_alpha, d_phi)
        ret, (x, y) = self.get_coords()
        if ret:
            self.history.append((self.alpha, self.phi, x, y))

    def _final_coords(self, alpha, phi, params_dict=None):
        if params_dict is None:
            params_dict = self.gun_params
        p = {}
        for key in self.gun_params:
            p[key] = params_dict.get(key, self.gun_params[key])

        coords = (p['x'], p['y'], p['z'])
        angles = (alpha + p['alpha_0'], phi + p['phi_0'])
        v = p['v']
        g = p['g']
        gun_length = p['gun_length']
        return final_coords(coords, angles, v, g, gun_length)

    # TODO constraints: z >= 0, g >0, v >0 etc.
    def estimate_gun_params(self, param_keys=None):
        if param_keys is None:
            param_keys = self.gun_params.keys()

        def func(params):
            params_dict = {key: value for key, value in
                           zip(param_keys, params)}

            xs = [x for alpha, phi, x, y in self.history]
            ys = [y for alpha, phi, x, y in self.history]

            fit_coords = [
                self._final_coords(alpha, phi, params_dict)
                for alpha, phi, x, y
                in self.history]

            xs_fit = [x for x, _ in fit_coords]
            ys_fit = [y for _, y in fit_coords]

            return np.asarray(
                [a - b for a, b in zip(xs + ys, xs_fit + ys_fit)])

        gun_params = [self.gun_params[key] for key in param_keys]
        x0 = [x if x is not None else 1.0
              for x in gun_params]
        res = leastsq(func, x0)
        for key, value in zip(param_keys, res[0]):
            self.gun_params[key] = value
        return self.gun_params


    def calibrate(self):
        self.history = []

        self.shoot()
        self.shoot(0.0, pi / 9)
        self.shoot(0.0, -2 * pi / 9)

        self.shoot(pi / 9, 0.0)
        self.shoot(0.0, pi / 9)
        self.shoot(0.0, pi / 9)

        self.shoot(pi / 9, 0.0)
        self.shoot(0.0, -pi / 9)
        self.shoot(0.0, - pi / 9)

        self.estimate_gun_params()


    if __name__ == '__main__':
        pass