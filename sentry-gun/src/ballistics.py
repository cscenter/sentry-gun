__author__ = 'vasdommes'

from numpy import sin, cos, sqrt


def final_coords((x0, y0, z0), (alpha, phi), v, g, gun_length=0.0):
    # Take into account length of gun
    x0 += gun_length * cos(alpha) * cos(phi)
    y0 += gun_length * cos(alpha) * sin(phi)
    z0 += gun_length * sin(alpha)

    # TODO check that z0 is not too low
    dist = v * cos(alpha) / g * (v * sin(alpha)
                                 + sqrt((v * sin(alpha)) ** 2 + 2 * g * z0)
                                 )
    x = x0 + dist * cos(phi)
    y = y0 + dist * sin(phi)
    return x, y
