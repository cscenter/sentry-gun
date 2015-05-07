import logging
import math
import numpy as np
import image_processing

__author__ = 'vasdommes'

logger = logging.getLogger('trajectory')


def get_acceleration(trajectory):
    """

    :type trajectory: collections.Iterable[(float,float)]
    :rtype :
    """
    # calculate acceleration
    if trajectory:
        return np.concatenate(
            ([[0.0, 0.0]], np.diff(trajectory, n=2, axis=0), [[0.0, 0.0]]))


def get_landing_point(trajectory):
    if not trajectory:
        return None
    v = np.concatenate((np.diff(trajectory, axis=0), [[0.0, 0.0]]))
    for i in xrange(1, len(v)):
        _, vy = v[i]
        _, vy_prev = v[i - 1]
        if vy <= 0 < vy_prev:
            return trajectory[i]


# TODO interpolate point using 4 neighbour points
def get_landing_point_by_acceleration(trajectory, threshold=1.0):
    if not trajectory:
        return None
    a = [ax ** 2 + ay ** 2 for ax, ay in get_acceleration(trajectory)]
    for i in xrange(2, len(a) - 2):
        neighbours = (a[i - 2], a[i - 1], a[i + 1], a[i + 2])
        if all(a[i] > x * threshold for x in neighbours):
            return trajectory[i]
            # TODO take into account change of direction?


# TODO consider only counterclockwise rotation
def get_landing_point_by_angle(trajectory, threshold_phi=0):
    """Look for first max change of direction"""
    if not trajectory:
        return None

    def prod((x1, y1), (x2, y2)):
        return x1 * x2 + y1 * y2

    def norm((x, y)):
        return math.sqrt(x ** 2 + y ** 2)

    v = np.diff(trajectory, axis=0)
    d_phi = []
    for i in xrange(1, len(v)):
        if norm(v[i]) * norm(v[i - 1]) == 0:
            d_phi.append(1.0)
        else:
            d_phi.append(prod(v[i], v[i - 1]) / norm(v[i]) / norm(v[i - 1]))

    for i in xrange(1, len(d_phi) - 1):
        neighbours = (d_phi[i - 1], d_phi[i + 1])
        if all(d_phi[i] < x - threshold_phi for x in neighbours):
            return trajectory[i + 1]


def get_ball_trajectory(mask_frames):
    return [image_processing.ball_center(frame) for frame in mask_frames]