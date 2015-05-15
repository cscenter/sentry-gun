__author__ = 'vasdommes'

import cv2
import os


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


def main():
    img = cv2.imread(os.path.join('../test/input', 'photo', 'Picture 7.jpg'))
    print(get_coords(img))


if __name__ == '__main__':
    main()