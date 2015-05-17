import cv2
import numpy as np
import img_util

__author__ = 'vasdommes'


def get_hls_range(img, winname=None):
    if winname is None:
        winname = 'Choose HLS bounds'

    h, w, _ = img.shape
    f = 720.0 / h
    img_720p = cv2.resize(img, dsize=None, fx=f, fy=f)

    def on_trackbar_changed(x):
        lowerb = tuple(
            cv2.getTrackbarPos(ch + '_min', winname) for ch in 'HLS')
        upperb = tuple(
            cv2.getTrackbarPos(ch + '_max', winname) for ch in 'HLS')
        out = img_util.apply_mask_hls(img_720p, lowerb, upperb)
        cv2.imshow(winname, out)

    cv2.imshow(winname, img_720p)
    for ch in 'HLS':
        cv2.createTrackbar(ch + '_min', winname, 0, 255, on_trackbar_changed)
        cv2.createTrackbar(ch + '_max', winname, 255, 255, on_trackbar_changed)

    while True:
        cv2.waitKey()
        lowerb = tuple(
            cv2.getTrackbarPos(ch + '_min', winname) for ch in 'HLS')
        upperb = tuple(
            cv2.getTrackbarPos(ch + '_max', winname) for ch in 'HLS')
        cv2.destroyWindow(winname)
        return lowerb, upperb


def show_hls(img, winname=None):
    if winname is None:
        winname = 'Choose HLS bounds'

    h, w, _ = img.shape
    f = 720.0 / h
    img_720p = cv2.resize(img, dsize=None, fx=f, fy=f)

    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = img[y, x]
            hls = cv2.cvtColor(np.asarray([[bgr]], dtype=img.dtype),
                               cv2.COLOR_BGR2HLS)[0, 0]
            out = img_720p.copy()
            cv2.putText(out, text='BGR={}, HLS={}'.format(bgr, hls),
                        org=(0, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255))

            cv2.imshow(winname, out)

    cv2.imshow(winname, img_720p)
    cv2.setMouseCallback(winname, onmouse)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow(winname)
            return


def get_coords(img, winname=None):
    out = img.copy()
    coords = []

    if winname is None:
        winname = 'choose coords'

    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.circle(out, (x, y), radius=10, color=(0, 0, 255),
                       thickness=-1)
            cv2.imshow(winname, out)

    cv2.imshow(winname, out)
    cv2.setMouseCallback(winname, onmouse)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            out = img.copy()
            coords = []
        if k == ord(' '):
            cv2.destroyWindow(winname)
            return coords