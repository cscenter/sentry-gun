import sys

__author__ = 'vasdommes'

import cv2
import numpy as np
import os
import time
import logging
import io

import video_processing
import video_io


def test1(input_path, out_prefix=None):
    logger.info('test1(\'{}\') started'.format(input_path))
    cap = cv2.VideoCapture(input_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    if out_prefix is None:
        out_prefix = os.path.join('output', os.path.basename(input_path))

    def write_video(path, frames):
        video_io.write_video(path, frames, fps,
                             frame_size=(int(width), int(height)))
        logger.info("Video written to file {}".format(path))

    frames, mask_frames = video_processing.extract_ball_from_capture(cap)
    cap.release()

    trajectory = video_processing.get_ball_trajectory(mask_frames)
    if trajectory:
        logger.info('trajectory found')
    else:
        logger.info('trajectory not found')

    for x, y in trajectory:
        center = (int(round(x)), int(round(y)))
        for frame in frames:
            cv2.circle(frame, center, 3, (0, 0, 255), thickness=-1)

    landing_point = video_processing.get_landing_point(trajectory)
    if landing_point:
        logger.info('landing point: {}'.format(landing_point))
        for frame in frames:
            x, y = landing_point
            center = (int(round(x)), int(round(y)))
            cv2.circle(frame, center, radius=4, color=(255, 0, 0),
                       thickness=-1)
    else:
        logger.info('landing point not found')

    landing_point = video_processing.get_landing_point_by_angle(trajectory)
    if landing_point:
        logger.info('landing point by angle: {}'.format(landing_point))
        for frame in frames:
            x, y = landing_point
            center = (int(round(x)), int(round(y)))
            cv2.circle(frame, center, radius=4, color=(0, 255, 0),
                       thickness=-1)
    else:
        logger.info('landing point by angle not found')

    write_video(out_prefix + '_frames.avi', frames)
    # write_video(out_prefix + '_mask_frames.avi', mask_frames)

    acc = video_processing.get_acceleration(trajectory)

    with io.open(out_prefix + '_traj.dat', mode='w') as handle:
        for i, (x, y) in enumerate(trajectory):
            ax, ay = acc[i]
            handle.write(u'{} {} {} {} {}\n'.format(i, x, y, ax, ay))


if __name__ == '__main__':
    hldr = logging.StreamHandler(sys.stdout)
    hldr.setLevel(logging.INFO)
    hldr.setFormatter(logging.Formatter(
        '%(asctime)s; %(name)s; %(levelname)s; %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))

    logger = logging.getLogger('video_test')
    logger.addHandler(hldr)
    logger.setLevel(logging.INFO)
    logger.info('started at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    for i in xrange(0, 10):
        test1(os.path.join('input', '{}.avi'.format(i)))

    logger.info('finished')