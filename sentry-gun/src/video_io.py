import cv2
import os

__author__ = 'vasdommes'


def write_video(path, frames, fps=30.0, frame_size=None, isColor=True):
    """
    Write video frames to file, override if exists

    :type frame_size: (int,int)
    :type fps: float
    :type path: str | unicode
    :rtype : None
    """
    if frames is None or len(frames) == 0:
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


def get_frames(input_path, max_count=None):
    """
    Get frames from video file

    :param input_path:
    :return:
    """
    cap = cv2.VideoCapture(input_path)
    frames = []
    if max_count:
        for _ in xrange(max_count):
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
    cap.release()
    return frames