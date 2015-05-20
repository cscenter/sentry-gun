from numpy.core.umath import pi
from gun_laying import GunLayer

__author__ = 'vasdommes'

import sys
import time
import cv2
import numpy as np
import logging
import os

import gui
import video_io
import img_util
import video_util
import trajectory as tr
import robot


def get_perspective_transform(cap):
    # Skip first frames
    for _ in xrange(10):
        if cap.isOpened():
            cap.read()
        else:
            raise IOError('Cannot connect to webcam')

    ret, img = cap.read()
    if not ret:
        raise IOError('Cannot connect to webcam')

    while True:
        carpet_corners = np.float32(gui.get_coords(img,
                                                   winname='Select 4 carpet corners from bottom-left, clockwise'))
        if len(carpet_corners) == 4:
            break

    logger.info('carpet corners: {}'.format(carpet_corners))
    corners_on_plane = np.float32(
        [(0.0, 0.0), (0, 133), (195, 133), (195, 0)])
    return cv2.getPerspectiveTransform(carpet_corners, corners_on_plane)


def perspective_transform((x, y), m):
    res = cv2.perspectiveTransform(src=np.float32([[(x, y)]]), m=m)
    return res[0, 0, 0], res[0, 0, 1]


def perspective_transform_contour(contour, m):
    return np.apply_along_axis(lambda p: perspective_transform(p, m), axis=2,
                               arr=contour)


def program(out_dir):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.cv.CV_CAP_PROP_FPS, 30.0)
    # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 864)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    perspective_matrix = get_perspective_transform(cap)
    #
    logger.info('matrix of perspective transform: {}'.format(
        perspective_matrix))

    img = read_from_capture(cap)
    cv2.imwrite(os.path.join(out_dir, 'img.jpg'), img)

    # Find carpet
    lowerb, upperb = gui.get_hls_range(img,
                                       winname='Choose HLS range for carpet')

    logger.info('Carpet HLS range: {} - {}'.format(lowerb, upperb))

    carpet_mask = img_util.green_carpet_mask(img, lowerb, upperb)
    cv2.imwrite(os.path.join(out_dir, 'carpet_mask.jpg'), carpet_mask)

    # Find target
    img = read_from_capture(cap)
    lowerb, upperb = gui.get_hls_range(img,
                                       winname='Choose HLS range for target')
    logger.info('Target HLS range: {} - {}'.format(lowerb, upperb))

    ret, target_contour = img_util.target_contour(img, lowerb, upperb,
                                                  carpet_mask)
    logging.debug('target contour: {}'.format(target_contour))
    if not ret:
        raise ValueError('Cannot find target')

    target_contour_plane = perspective_transform_contour(target_contour,
                                                         perspective_matrix)
    m = cv2.moments(target_contour_plane)
    mass = m['m00']
    # x,y coordinates on plane
    if mass > 0:
        target_coords = m['m10'] / mass, m['m01'] / mass
        logger.info(
            'target coordinates on plane: (x,y) = {}'.format(target_coords))
    else:
        raise ValueError('Cannot find target center')

    cv2.drawContours(img, contours=[target_contour], contourIdx=-1,
                     color=(255, 255, 255),
                     thickness=2)
    img_carpet = cv2.bitwise_and(img, img, mask=carpet_mask)
    cv2.addWeighted(img, 0.25, img_carpet, 0.75, 0, dst=img_carpet)

    cv2.imwrite(os.path.join(out_dir, 'carpet_target.jpg'), img_carpet)

    # Find ball
    img = read_from_capture(cap)
    ball_lowerb, ball_upperb = gui.get_hls_range(img,
                                                 winname='Choose HLS range for ball')

    logger.info('Ball HLS range: {} - {}'.format(ball_lowerb, ball_upperb))

    conn = robot.RobotConnector(winscp_path='C:/TRIKStudio/winscp/WinSCP.com')
    trik = robot.Robot(conn, angle_to_encoder=180 / np.pi)

    traj = None

    def get_coords():
        if traj is None:
            return False, (0, 0)
        else:
            idx, landing_point = tr.get_landing_point(traj)
            if landing_point is not None:
                return True, perspective_transform(landing_point,
                                                   perspective_matrix)
        return False, (0, 0)

    gun_params = {'x': 300, 'y': 50, 'z': 97, 'v': 300, 'g': 981,
                  'alpha_0': 15 * pi / 180, 'phi_0': 0.0, 'gun_length': 10}
    gun_layer = GunLayer(rotate_and_shoot=trik.rotate_and_shoot,
                         get_coords=get_coords, target=target_coords,
                         gun_params=gun_params)

    for i in xrange(10):
        out_prefix = os.path.join(out_dir, str(i))

        commands = tuple()
        stdout, stderr = trik.open_and_trikRun(*commands)
        logger.debug('trik: sent command = {}'.format(commands))
        logger.debug('trik: stdout = {}'.format(stdout))
        logger.debug('trik: stderr = {}'.format(stderr))

        logger.info('ready to shoot...')
        gun_layer.shoot_at_target()
        logger.info('ready to capture...')
        frames, mask_ball = video_util.extract_ball_from_capture(cap,
                                                                 max_frames_count=30,
                                                                 skip_count=0,
                                                                 carpet_mask=carpet_mask,
                                                                 get_mask=video_util.get_ball_mask,
                                                                 ball_lowerb=ball_lowerb,
                                                                 ball_upperb=ball_upperb,
                                                                 ball_size=3
                                                                 )

        logger.debug("Frames captured: {}".format(len(frames)))

        traj = tr.get_ball_trajectory(mask_ball)
        if not traj:
            logger.info('trajectory not found')
        else:
            logger.info('trajectory found')
            for x, y in traj:
                center = (int(round(x)), int(round(y)))
                for frame in frames:
                    cv2.circle(frame, center, 3, (0, 0, 255), thickness=-1)
            idx, landing_point = tr.get_landing_point(traj)
            if landing_point is not None:
                logger.info('landing point on plane: {}'.format(
                    perspective_transform(landing_point, perspective_matrix)))

                for frame in frames:
                    x, y = landing_point
                    center = (int(round(x)), int(round(y)))
                    cv2.circle(frame, center, radius=10, color=(255, 255, 255),
                               thickness=1)
        video_io.write_video(out_prefix + '_frames_ball.avi', frames)
        video_io.write_video(out_prefix + '_ball_mask.avi', mask_ball)

    cap.release()


def read_from_capture(cap):
    ret, img = cap.read()
    if not ret:
        logger.error('Cannot read from camera')
        raise IOError('Cannot read from camera')
    return img


def main():
    global hldr, logger

    out_dir = time.strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(out_dir)

    hldr = logging.FileHandler(os.path.join(out_dir, 'log.txt'))
    hldr.setLevel(logging.DEBUG)
    hldr.setFormatter(logging.Formatter(
        '%(asctime)s; %(name)s; %(levelname)s; %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger = logging.getLogger('sentry_gun')
    logger.addHandler(hldr)

    hldr_con = logging.StreamHandler(sys.stdout)
    hldr_con.setLevel(logging.DEBUG)
    hldr_con.setFormatter(logging.Formatter(
        '%(asctime)s; %(name)s; %(levelname)s; %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(hldr_con)

    logger.setLevel(logging.DEBUG)
    video_util.logger.setLevel(logging.DEBUG)
    video_util.logger.addHandler(hldr)
    video_util.logger.addHandler(hldr_con)

    logger.info('started at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    program(out_dir)

    logger.info('finished')


if __name__ == '__main__':
    main()
