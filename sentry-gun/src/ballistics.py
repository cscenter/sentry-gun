__author__ = 'vasdommes'

from numpy import sin, cos, sqrt


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
        raise ValueError('z0 is too low')

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
    # TODO get alpha from target coords