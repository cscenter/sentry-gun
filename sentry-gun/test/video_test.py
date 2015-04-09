__author__ = 'vasdommes'

import cv2
import numpy as np
import subprocess
import os
import video_processing
import math
import logging

logging.basicConfig(level=logging.INFO)

input_path = os.path.join("input", "input_HD.avi")
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


def write_video(path, frames, frameSize=None, fps=fps):
    if not frameSize:
        height, width = frames.shape[1:3]
        frameSize = width, height
    path = os.path.join('output', path)
    out = get_video_writer(path, frameSize=frameSize, fps=fps)
    for frame in frames:
        out.write(frame)
    out.release()
    logging.info("Video written to file {}".format(path))


def get_video_writer(path, fourcc=cv2.cv.CV_FOURCC(*'XVID'), fps=fps,
                     frameSize=(width, height), isColor=True):
    """

    :rtype: cv2.VideoWriter
    """
    if os.path.exists(path):
        os.remove(path)
    return cv2.VideoWriter(path, fourcc, fps, frameSize, isColor)


logging.info("Start reading file {}".format(input_path))
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
    else:
        break
cap.release()
frames = np.asarray(frames[:-1])
logging.info("{1} frames were read from {0}".format(input_path, len(frames)))
logging.info("End reading file")

background = np.float32(np.average(frames, axis=0))
cv2.imwrite(os.path.join("output", "background.jpg"), background)
logging.info("Background calculated")

mog = video_processing.subtract_background_MOG(frames)
write_video('mog.avi', mog)

mog_contours = np.empty(shape=frames.shape, dtype=frames.dtype)
for i in range(len(mog)):
    f = mog[i]
    mog_contours[i] = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    res = mog_contours[i]

    contours, hier = cv2.findContours(f, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    ellipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
    # ellipse = ((x,y),(MajAxis,minAxis),angle
    for e in ellipses:
        cv2.ellipse(res, e, color=(0, 0, 255), thickness=2)
logging.info('contours found')
write_video('mog_contours.avi', mog_contours)

balls = np.empty(mog.shape, mog.dtype)
for i in range(len(mog)):
    ret, balls[i] = video_processing.detect_ball(mog[i], ball_size=3)
    logging.debug(ret)
    # if ret:
    # x, y = video_processing.ball_center(balls[i])
    # r = video_processing.ball_radius(balls[i])
write_video('ball_mask.avi', balls)

balls_orig = np.empty(frames.shape, mog.dtype)
for i in range(len(mog)):
    balls_orig[i] = cv2.bitwise_and(frames[i],
                                    cv2.cvtColor(balls[i], cv2.COLOR_GRAY2BGR))
write_video('ball_orig.avi', balls_orig)

cv2.destroyAllWindows()