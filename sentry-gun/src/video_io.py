__author__ = 'vasdommes'

import cv2
import os


def write_video(path, frames, fps, frame_size=None, isColor=True):
    """
    Write video frames to file, override if exists

    :type frame_size: (int,int)
    :type fps: float
    :type path: str | unicode
    :rtype : None
    """
    if not frames:
        raise ValueError('no frames ')
    if fps <= 0:
        raise ValueError('fps must be positive')
    if frame_size is None:
        height, width = frames[0].shape[0:2]
        frame_size = width, height
    out = get_video_writer(path, cv2.cv.CV_FOURCC(*'XVID'), fps,
                           frame_size, isColor)
    for frame in frames:
        out.write(frame)
    out.release()


def get_video_writer(path, fourcc, fps, frame_size,
                     isColor):
    """
    Remove output file (if exists) and returns VideoWriter object

    :type path: str | unicode
    :type frame_size: (int,int)
    :type fps: float
    :rtype: cv2.VideoWriter
    """
    if os.path.exists(path):
        os.remove(path)
    return cv2.VideoWriter(path, fourcc, fps, frame_size, isColor)
