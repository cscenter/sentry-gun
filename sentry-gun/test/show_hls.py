__author__ = 'vasdommes'

import cv2
import numpy as np
import os


def show_hls(img):
    h, w, _ = img.shape
    f = 720.0 / h
    img_720p = cv2.resize(img, dsize=None, fx=f, fy=f)

    def on_trackbar_changed(x):
        h_min = cv2.getTrackbarPos('H_min', 'image')
        l_min = cv2.getTrackbarPos('L_min', 'image')
        s_min = cv2.getTrackbarPos('S_min', 'image')
        h_max = cv2.getTrackbarPos('H_max', 'image')
        l_max = cv2.getTrackbarPos('L_max', 'image')
        s_max = cv2.getTrackbarPos('S_max', 'image')

        out = apply_mask_hls(img_720p, (h_min, h_max), (l_min, l_max),
                             (s_min, s_max))
        cv2.imshow('image', out)


    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = img[y, x]
            hls = cv2.cvtColor(np.asarray([[bgr]], dtype=img.dtype),
                               cv2.COLOR_BGR2HLS)[0, 0]
            out = img_720p.copy()
            cv2.putText(out, text='BGR={}, HLS={}'.format(bgr, hls),
                        org=(0, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255))

            cv2.imshow('image', out)

    cv2.imshow('image', img_720p)
    cv2.setMouseCallback('image', onmouse)
    cv2.createTrackbar('H_min', 'image', 0, 255, on_trackbar_changed)
    cv2.createTrackbar('L_min', 'image', 0, 255, on_trackbar_changed)
    cv2.createTrackbar('S_min', 'image', 0, 255, on_trackbar_changed)
    cv2.createTrackbar('H_max', 'image', 255, 255, on_trackbar_changed)
    cv2.createTrackbar('L_max', 'image', 255, 255, on_trackbar_changed)
    cv2.createTrackbar('S_max', 'image', 255, 255, on_trackbar_changed)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow('image')
            return


def apply_mask_hls(img, h_range=(0, 255), l_range=(0, 255), s_range=(0, 255),
                   dst=None):
    h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    h_mask = cv2.inRange(h, *h_range)
    l_mask = cv2.inRange(l, *l_range)
    s_mask = cv2.inRange(s, *s_range)

    mask = np.zeros_like(h_mask)
    cv2.bitwise_and(h_mask, l_mask, mask, s_mask)

    return cv2.bitwise_and(img, img, dst, mask)


def main():
    img = cv2.imread(os.path.join('input', 'photo', 'Picture 7.jpg'))
    show_hls(img)


if __name__ == '__main__':
    main()