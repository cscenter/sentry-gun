__author__ = 'vasdommes'

import cv2
import numpy as np
import math


def subtract_background_MOG(frames, *args, **kwargs):
    mog = cv2.BackgroundSubtractorMOG(*args, **kwargs)
    dst = np.empty(frames.shape[:3], frames.dtype)
    for i in range(len(frames)):
        dst[i] = mog.apply(frames[i])
    return dst


def subtract_background_MOG2(frames, *args, **kwargs):
    mog2 = cv2.BackgroundSubtractorMOG2(*args, **kwargs)
    dst = np.empty(frames.shape[:3], frames.dtype)
    for i in range(len(frames)):
        dst[i] = mog2.apply(frames[i])
    return dst


def subtract_background_gray(frames, background=None, dst=None):
    """

    returns gray image

    :type frames: np.ndarray

    :type dst: np.ndarray

    :param frames:
    :param background:
    :param dst:
    :rtype: np.ndarray
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

    :rtype: np.ndarray
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
        area, contour = max(zip(areas, contours))
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
        x = m['m10'] / mass
        y = m['m01'] / mass
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
    f0 = cv2.cvtColor(np.float32(sample), cv2.COLOR_BGR2GRAY)
    f = np.float32(cv2.cvtColor(np.float32(src), cv2.COLOR_BGR2GRAY))
    (dx, dy) = cv2.phaseCorrelate(f, f0)
    (h, w, _) = src.shape
    return cv2.warpAffine(src, M=np.asarray([[1, 0, dx], [0, 1, dy]]),
                          dsize=(w, h), borderMode=cv2.BORDER_TRANSPARENT)
