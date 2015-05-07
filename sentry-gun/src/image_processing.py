import math
import cv2
import numpy as np
import logging
import scipy.ndimage as ndimage

__author__ = 'vasdommes'

logger = logging.getLogger('image_processing')


def equalize_bgr(img, dst=None):
    if img is None:
        raise ValueError('img is None')

    dst = cv2.cvtColor(img, cv2.COLOR_BGR2HLS, dst)
    h, l, s = cv2.split(dst)
    cv2.equalizeHist(l, dst=l)
    cv2.merge((h, l, s), dst=dst)
    return cv2.cvtColor(dst, cv2.COLOR_HLS2BGR, dst=dst)


def largest_contour_blob(img):
    if len(img.shape) != 2:
        raise ValueError('img is not grayscale')

    # bin_count = np.bincount(img.flat)
    # if np.count_nonzero(bin_count) != 2:
    # raise ValueError('img is not binary, bincount = {}'.format(bin_count))

    mask = img.copy()

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    blob = np.zeros(mask.shape, mask.dtype)
    if areas:
        area, contour = max(zip(areas, contours), key=(lambda x: x[0]))
        cv2.drawContours(blob, [contour], -1, color=255, thickness=-1)
        ret = True
    else:
        ret = False
    return ret, blob


def largest_blob(img):
    """
    Find only the largest external contour and fill it.

    :param img: np.ndarray
    :return:
    """
    labeled_array, n_features = ndimage.label(img)
    if n_features == 0:
        return False, np.zeros_like(img)
    bins = np.bincount(labeled_array.flat)
    label = int(np.argmax(bins[1:]) + 1)
    res = np.where((labeled_array == label), 255, 0).astype(img.dtype)
    # res= cv2.inRange(labeled_array, label, label)
    return True, res


# TODO use ball hue to remove noise?
# TODO check whether the result really looks like ball?
def detect_ball(mask, ball_size=11):
    """

    :rtype: (bool, np.ndarray)
    """
    if ball_size % 2 == 0:
        ball_size -= 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ball_size, ball_size))
    opened = cv2.morphologyEx(mask, op=cv2.MORPH_OPEN, kernel=ker)
    # closed = cv2.morphologyEx(opened, op=cv2.MORPH_CLOSE, kernel=ker)
    return largest_contour_blob(opened)


def ball_fitEllipse(ball_mask):
    contours, hier = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ValueError("Input image must have one contour")

    c = contours[0]
    # ellipse = ((x,y),(MajAxis,minAxis),angle
    return cv2.fitEllipse(c)


def ball_radius(ball_mask):
    m = cv2.moments(ball_mask, binaryImage=True)
    if m['m00'] > 0:
        mass = m['m00']
        dx = math.sqrt(m['mu20'] / mass)
        dy = math.sqrt(m['mu02'] / mass)
        return math.sqrt(dx ** 2 + dy ** 2)
    else:
        return None


def ball_center(ball_mask):
    m = cv2.moments(ball_mask, binaryImage=True)
    if m['m00'] > 0:
        mass = m['m00']
        x = m['m10'] / mass
        y = m['m01'] / mass
        return x, y
    else:
        return None


def mask_hue(img, min_hue, max_hue):
    h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    return cv2.inRange(h, min_hue, max_hue)


def green_carpet_mask(img, min_hue=55, max_hue=70, ker_erode=None,
                      ker_close=None, ker_erode2=None):
    """
    Find green carpet on image

    :param img:
    """
    mask = mask_hue(img, min_hue, max_hue)
    if not ker_erode:
        ker_erode = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(11, 11))
    cv2.morphologyEx(mask, op=cv2.MORPH_ERODE, kernel=ker_erode, dst=mask)
    ret, mask = largest_contour_blob(mask)
    if ret:
        cv2.morphologyEx(mask, op=cv2.MORPH_DILATE, kernel=ker_erode, dst=mask)
        if not ker_close:
            ker_close = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))
        cv2.morphologyEx(mask, cv2.MORPH_DILATE, ker_close, dst=mask)
        if not ker_erode2:
            ker_erode2 = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                   ksize=(31, 31))
        return cv2.morphologyEx(mask, op=cv2.MORPH_ERODE, kernel=ker_erode2,
                                dst=mask)
    else:
        return np.zeros_like(mask)


def test_carpet(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    cv2.threshold(img, thresh=0, maxval=255,
                  type=cv2.THRESH_OTSU | cv2.THRESH_BINARY,
                  dst=img)
    # cv2.imshow('img', img)
    _, b1 = largest_blob(img)
    _, b2 = largest_contour_blob(img)
    cv2.imshow('b1', b1)
    cv2.imshow('b2', b2)
    cv2.imshow('db', cv2.bitwise_xor(b1, b2))
    cv2.waitKey()
    cv2.destroyAllWindows()
    return

    # ret, blob = largest_blob(img)
    ret, blob = True, img
    if ret:
        # cv2.imshow('blob', blob)

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(11, 11))
        eroded = cv2.morphologyEx(blob, op=cv2.MORPH_ERODE, kernel=ker)
        eroded = cv2.copyMakeBorder(eroded, 10, 10, 10, 10,
                                    cv2.BORDER_CONSTANT,
                                    value=0)

        # cv2.imshow('erode', eroded)

        ret, eroded = largest_contour_blob(eroded)
        # cv2.imshow('blob_erode', eroded)

        dilated = cv2.morphologyEx(eroded, op=cv2.MORPH_DILATE, kernel=ker)
        # cv2.imshow('blob_dilate', dilated)

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(101, 101))
        closed = cv2.morphologyEx(dilated, op=cv2.MORPH_CLOSE, kernel=ker)
        cv2.imshow(input_path + '_blob_closed', closed)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    import os.path

    for i in xrange(131, 140):
        input_path = os.path.join(os.path.pardir, 'test', 'output', 'green',
                                  '{}.avi_GREEN.jpg'.format(i))

        test_carpet(input_path)


if __name__ == '__main__':
    # main()
    pass
