__author__ = 'vasdommes'

import numpy as np
from numpy import sin, cos, sqrt, arctan


class BallisticError(Exception):
    pass


def dist(alpha, v, g, z0):
    return v * cos(alpha) / g * (v * sin(alpha)
                                 + sqrt((v * sin(alpha)) ** 2 + 2 * g * z0)
                                 )


def dist_prime(alpha, v, g, z0):
    """
    Derivative of shot distance with respect to vertical angle

    :param alpha:
    :param v:
    :param g:
    :param z0:
    :return: :raise ValueError:
    """
    discr = 2 * g * z0 + (v * sin(alpha)) ** 2
    if discr < 0:
        raise BallisticError('z0 is too low')

    return v / g * (
        v * cos(alpha) ** 2 * (
            1 + v * sin(alpha) / sqrt(discr)
        )
        - sin(alpha) * (
            v * sin(alpha) + sqrt(discr)
        )
    )


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


# def max_length
# get alpha from target coords
def shot_angle(dist, z0, v, g, gun_length=0.0):
    """

    :param dist:
    :param z0:
    :param v:
    :param g:
    :param gun_length:
    :return: :raise BallisticError:
    """
    discr = v ** 4 - g * (g * dist ** 2 - 2 * z0 * v ** 2)
    if discr < 0:
        raise BallisticError('too long distance')
    return tuple(
        arctan((v ** 2 + s) / (g * dist)) for s in [-sqrt(discr), sqrt(discr)])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle