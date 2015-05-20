from collections import deque
import logging
from multiprocessing.pool import ThreadPool
import copy

import cv2
import numpy as np

import img_util

__author__ = 'vasdommes'

logger = logging.getLogger('video_processing')


def preprocess(img, dst=None, carpet_mask=None):
    h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    # Only orange colors
    cv2.inRange(h, 18, 35, h)
    cv2.inRange(s, 50, 255, s)

    # mask = image_processing.mask_hue(img, min_hue=20, max_hue=40)
    mask = cv2.bitwise_and(h, s, mask=carpet_mask)
    return cv2.bitwise_and(img, img, dst, mask)


def get_ball_mask(img, dst=None, lowerb=(20, 50, 0), upperb=(40, 255, 255),
                  carpet_mask=None):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, lowerb, upperb)
    return cv2.bitwise_and(mask, mask, dst=dst, mask=carpet_mask)


def extract_ball_from_capture(cap, max_frames_count=-1, skip_count=0,
                              carpet_mask=None, get_mask=None,
                              carpet_lowerb=None,
                              carpet_upperb=None,
                              ball_lowerb=None,
                              ball_upperb=None,
                              ball_size=3):
    """

    Read frames from capture until we detect motion of the ball

    Return tuple of (original frames, ball mask frames)

    :param carpet_lowerb:
    :rtype : (list(np.ndarray), list(np.ndarray))
    """
    frames = []
    mask_frames = []
    # mog = cv2.BackgroundSubtractorMOG(history=5, nmixtures=4,backgroundRatio=0.7)
    mog = cv2.BackgroundSubtractorMOG()
    motion_started = False
    for _ in xrange(skip_count):
        if cap.isOpened():
            cap.read()

    prev_mask = None
    while cap.isOpened():
        if 0 < max_frames_count <= len(frames):
            logger.debug('max frames count reached')
            break

        ret, frame = cap.read()
        if ret:
            if get_mask is not None:
                if carpet_mask is None:
                    carpet_mask = img_util.green_carpet_mask(frame,
                                                             carpet_lowerb,
                                                             carpet_upperb)
                mask = get_mask(frame, lowerb=ball_lowerb, upperb=ball_upperb,
                                carpet_mask=carpet_mask)
                if prev_mask is None:
                    move_mask = np.zeros_like(mask)
                else:
                    move_mask = cv2.bitwise_not(prev_mask, mask=mask)
                prev_mask = mask
            else:
                logger.debug('get_mask = None')
                move_mask = mog.apply(frame)

            is_ball, move_mask = img_util.detect_ball(move_mask, ball_size)
            if is_ball:
                if not motion_started:
                    logger.debug('ball appeared')
                    motion_started = True
                frames.append(frame)
                mask_frames.append(move_mask)
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


def extract_ball_from_capture_threaded(cap, max_frames_count=-1, skip_count=0,
                                       carpet_mask=None, get_mask=None,
                                       carpet_lowerb=None,
                                       carpet_upperb=None,
                                       ball_lowerb=None,
                                       ball_upperb=None):
    """

    Read frames from capture until we detect motion of the ball

    Return tuple of (original frames, ball mask frames)

    :param carpet_lowerb:
    :rtype : (list(np.ndarray), list(np.ndarray))
    """
    frames = []
    mask_frames = []
    # mog = cv2.BackgroundSubtractorMOG(history=5, nmixtures=4,backgroundRatio=0.7)
    mog = cv2.BackgroundSubtractorMOG()
    motion_started = False
    for _ in xrange(skip_count):
        if cap.isOpened():
            cap.read()

    prev_mask = None

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    def process_frame(frame, prev_mask, carpet_mask):
        if carpet_mask is None:
            carpet_mask = img_util.green_carpet_mask(frame,
                                                     carpet_lowerb,
                                                     carpet_upperb)
        mask = get_mask(frame, lowerb=ball_lowerb, upperb=ball_upperb,
                        carpet_mask=carpet_mask)
        if prev_mask is None:
            move_mask = np.zeros_like(mask)
        else:
            move_mask = cv2.bitwise_not(prev_mask, mask=mask)

        prev_mask = mask
        is_ball, move_mask = img_util.detect_ball(move_mask)
        return is_ball, move_mask, prev_mask, frame

    while cap.isOpened():
        while len(pending) > 0 and pending[0].ready():
            is_ball, move_mask, prev_mask, frame = pending.popleft().get()
            if is_ball:
                if not motion_started:
                    logger.debug('ball appeared')
                    motion_started = True
                frames.append(frame)
                mask_frames.append(move_mask)
            else:
                if motion_started:
                    logger.debug(
                        'ball disappeared. Frames count: {}'.format(
                            len(frames)))
                    break
                else:
                    continue

            if 0 < max_frames_count < len(frames):
                logger.debug('max frames count reached')
                break

        if True:
            ret, frame = cap.read()
            task = pool.apply_async(process_frame,
                                    (frame, prev_mask, carpet_mask))
            pending.append(task)
        else:
            logger.debug('Cannot read more frames')
            break
    return frames, mask_frames


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
        dst = []
    if background is None:
        background = np.average(np.asarray(frames, dtype=np.uint8),
                                axis=0).astype(np.uint8)

    background = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    # for i in range(len(frames)):
    for frame in frames:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        diff = cv2.cvtColor(cv2.absdiff(lab, background), cv2.COLOR_LAB2BGR)
        dst_frame = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        dst.append(dst_frame)
    return dst


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
