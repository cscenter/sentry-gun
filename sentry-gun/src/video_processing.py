__author__ = 'vasdommes'

import cv2
import numpy as np
import math
import logging

logger = logging.getLogger('video_processing')


def extract_ball_from_capture(cap, max_frames_count=30 * 3):
    """

    Read frames from capture until we detect motion of the ball

    Return tuple of (original frames, ball mask frames)

    :rtype : (list(np.ndarray), list(np.ndarray))
    """
    frames = []
    mask_frames = []
    mog = cv2.BackgroundSubtractorMOG()
    motion_started = False
    while cap.isOpened():
        if len(frames) > max_frames_count:
            logger.debug('max frames count reached')
            break

        ret, frame = cap.read()
        if ret:
            mask = mog.apply(frame)
            is_ball, mask = detect_ball(mask)
            if is_ball:
                if not motion_started:
                    logger.debug('ball appeared')
                    motion_started = True
                frames.append(frame)
                mask_frames.append(mask)
            else:
                if motion_started:
                    logger.debug(
                        'ball disappeared. Frames count: {}'.format(
                            len(frames)))
                    break
                else:
                    continue
        else:
            logger.debug('Cannot read more frames')
            break
    return frames, mask_frames


def get_ball_trajectory(mask_frames):
    return [ball_center(frame) for frame in mask_frames]


# TODO move to separate module
def get_acceleration(trajectory):
    """

    :type trajectory: collections.Iterable[(float,float)]
    :rtype :
    """
    # calculate acceleration
    return np.concatenate(
        ([[0.0, 0.0]], np.diff(trajectory, n=2, axis=0), [[0.0, 0.0]]))


# TODO move to separate module
# TODO interpolate point using 4 neighbour points
def get_landing_point(trajectory, threshold=1.0):
    a = [ax ** 2 + ay ** 2 for ax, ay in get_acceleration(trajectory)]
    for i in xrange(2, len(a) - 2):
        neighbours = (a[i - 2], a[i - 1], a[i + 1], a[i + 2])
        if all(a[i] > x * threshold for x in neighbours):
            return trajectory[i]
            # TODO take into account change of direction? Current version fails on 7.avi


# TODO 1.avi and 8.avi fail
# TODO consider only counterclockwise rotation
# TODO threshold
def get_landing_point_by_angle(trajectory):
    """Look for first max change of direction"""

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
        if all(d_phi[i] < x for x in neighbours):
            return trajectory[i + 1]


def subtract_background_MOG(frames, *args, **kwargs):
    mog = cv2.BackgroundSubtractorMOG(*args, **kwargs)
    dst = []
    for frame in frames:
        dst.append(mog.apply(frame))
    return dst


def subtract_background_MOG2(frames, *args, **kwargs):
    mog2 = cv2.BackgroundSubtractorMOG2(*args, **kwargs)
    dst = []
    for frame in frames:
        dst.append(mog2.apply(frame))
    return dst


def subtract_background_gray(frames, background=None, dst=None):
    """

    returns gray image
    """

    if dst is None:
        dst = np.empty(frames.shape[:3], frames.dtype)
    if background is None:
        background = np.average(frames, axis=0).astype(np.uint8)

    background = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    # for i in range(len(frames)):
    for frame, dst_frame in zip(frames, dst):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        diff = cv2.cvtColor(cv2.absdiff(lab, background), cv2.COLOR_LAB2BGR)
        cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY, dst_frame)
    return dst


# TODO use ball hue to remove noise?
# TODO check whether the result really looks like ball?
def detect_ball(mask, ball_size=3):
    """

    :rtype: (bool, np.ndarray)
    """
    if ball_size % 2 == 0:
        ball_size -= 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ball_size, ball_size))
    opened = cv2.morphologyEx(mask, op=cv2.MORPH_OPEN, kernel=ker)
    # closed = cv2.morphologyEx(opened, op=cv2.MORPH_CLOSE, kernel=ker)

    contours, hier = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]

    ball = np.zeros(mask.shape, mask.dtype)
    if areas:
        area, contour = max(zip(areas, contours), key=(lambda x: x[0]))
        cv2.drawContours(ball, [contour], -1, color=255, thickness=-1)
        ret = True
    else:
        ret = False
    return ret, ball


def ball_center(ball_mask):
    m = cv2.moments(ball_mask, binaryImage=True)
    if m['m00'] > 0:
        mass = m['m00']
        x = m['m10'] / mass
        y = m['m01'] / mass
        return x, y
    else:
        return None


def ball_radius(ball_mask):
    m = cv2.moments(ball_mask, binaryImage=True)
    if m['m00'] > 0:
        mass = m['m00']
        dx = math.sqrt(m['mu20'] / mass)
        dy = math.sqrt(m['mu02'] / mass)
        return math.sqrt(dx ** 2 + dy ** 2)
    else:
        return None


def ball_fitEllipse(ball_mask):
    contours, hier = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ValueError("Input image must have one contour")

    c = contours[0]
    # ellipse = ((x,y),(MajAxis,minAxis),angle
    return cv2.fitEllipse(c)


def unshake(src, sample):
    """

    Compensate camera shaking.
    :rtype : np.ndarray
    """
    f0 = cv2.cvtColor(np.float32(sample), cv2.COLOR_BGR2GRAY)
    f = np.float32(cv2.cvtColor(np.float32(src), cv2.COLOR_BGR2GRAY))
    (dx, dy) = cv2.phaseCorrelate(f, f0)
    (h, w, _) = src.shape
    return cv2.warpAffine(src, M=np.asarray([[1, 0, dx], [0, 1, dy]]),
                          dsize=(w, h), borderMode=cv2.BORDER_TRANSPARENT)
