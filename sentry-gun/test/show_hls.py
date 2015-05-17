__author__ = 'vasdommes'

import os

import cv2

import gui


def main():
    img = cv2.imread(os.path.join('input', 'photo', 'img.jpg'))
    gui.show_hls(img)


if __name__ == '__main__':
    main()