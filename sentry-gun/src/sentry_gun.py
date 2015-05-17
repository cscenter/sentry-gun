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
    # img = cv2.imread('../test/input/photo/Picture 4.jpg')
    if not ret:
        raise IOError('Cannot connect to webcam')

    while True:
        carpet_angles = np.float32(gui.get_coords(img,
                                                  winname='Select 4 carpet angles from bottom-left, clockwise'))
        if len(carpet_angles) == 4:
            break

    angles_square = np.float32(
        [(0, 133), (0.0, 0.0), (195, 0), (195, 133)])
    return cv2.getPerspectiveTransform(carpet_angles, angles_square)


def perspective_transform((x, y), m):
    res = cv2.perspectiveTransform(src=np.float32([[(x, y)]]), m=m)
    return res[0, 0, 0], res[0, 0, 1]


def perspective_transform_contour(contour, m):
    return np.apply_along_axis(lambda p: perspective_transform(p, m), axis=2,
                               arr=contour)


def program(out_dir):
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    cap.set(cv2.cv.CV_CAP_PROP_FPS, 30.0)
    # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)

    perspective_matrix = get_perspective_transform(cap)
    #
    logger.info('matrix of perspective transform: {}'.format(
        perspective_matrix))

    ret, img = cap.read()
    if not ret:
        logger.error('Cannot read from camera')
        raise IOError('Cannot read from camera')
    cv2.imwrite(os.path.join(out_dir, 'img.jpg'), img)

    # Find carpet
    lowerb, upperb = gui.get_hls_range(img,
                                       winname='Choose HLS range for carpet')
    logger.info('Carpet HLS range: {} - {}'.format(lowerb, upperb))

    carpet_mask = img_util.green_carpet_mask(img, lowerb, upperb)
    cv2.imwrite(os.path.join(out_dir, 'carpet_mask.jpg'), carpet_mask)

    # Find target
    ret, img = cap.read()
    if not ret:
        logger.error('Cannot read from camera')
        raise IOError('Cannot read from camera')
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
    ret, img = cap.read()
    if not ret:
        logger.error('Cannot read from camera')
        raise IOError('Cannot read from camera')
    ball_lowerb, ball_upperb = gui.get_hls_range(img,
                                                 winname='Choose HLS range for ball')
    logger.info('Ball HLS range: {} - {}'.format(lowerb, upperb))

    trik = robot.RobotConnector(winscp_path='C:/TRIKStudio/winscp/WinSCP.com')

    for i in xrange(10):
        out_prefix = os.path.join(out_dir, str(i))

        commands = tuple()
        stdout, stderr = trik.open_and_trikRun(commands)
        logger.debug('trik: sent command = {}'.format(commands))
        logger.debug('trik: stdout = {}'.format(stdout))
        logger.debug('trik: stderr = {}'.format(stderr))

        logger.info('ready to capture...')
        frames, mask_ball = video_util.extract_ball_from_capture(cap,
                                                                 skip_count=0,
                                                                 carpet_mask=carpet_mask,
                                                                 get_mask=video_util.get_ball_mask,
                                                                 ball_lowerb=ball_lowerb,
                                                                 ball_upperb=ball_upperb
                                                                 )

        logger.debug("Frames captured: {}".format(len(frames)))

        # write_video(out_prefix + '_frames.avi', frames)
        # video_io.write_video(out_prefix + '_ball.avi', mask_ball)

        traj = tr.get_ball_trajectory(mask_ball)
        if not traj:
            logger.info('trajectory not found')
        else:
            logger.info('trajectory found')
            for x, y in traj:
                center = (int(round(x)), int(round(y)))
                for frame in frames:
                    cv2.circle(frame, center, 3, (0, 0, 255), thickness=-1)
            landing_point = tr.get_landing_point(traj)
            if landing_point:
                logger.info('landing point on plane: {}'.format(
                    perspective_transform(landing_point, perspective_matrix)))

                for frame in frames:
                    x, y = landing_point
                    center = (int(round(x)), int(round(y)))
                    cv2.circle(frame, center, radius=10, color=(255, 255, 255),
                               thickness=1)
        video_io.write_video(out_prefix + '_frames_ball.avi', frames)

        # dst = cv2.warpPerspective(img, perspective_matrix, dsize=(600, 600))
        #
        # #
        # # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # # cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY,
        # # dst=gray)
        # #
        # # contours, hier = cv2.findContours(gray, mode=cv2.RETR_EXTERNAL,
        # # method=cv2.CHAIN_APPROX_SIMPLE)
        #
        # cv2.imshow('dst', dst)
        #
        # cv2.waitKey()
        #
        # cv2.destroyAllWindows()
        cap.release()


def main():
    global hldr, logger

    out_dir = time.strftime('%Y-%m-%d_%H-%M-%S')
    # os.mkdir('aaa')
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
    logger.info('started at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    program(out_dir)

    logger.info('finished')


if __name__ == '__main__':
    main()