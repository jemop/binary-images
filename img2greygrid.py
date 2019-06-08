import os

import cv2
import numpy as np
from skimage.util.shape import view_as_windows

ORIGINAL_IMAGES_PATH = os.path.join("data", "original_images")
PROCESSED_IMAGES_PATH = os.path.join("data", "processed_images")

LINE_COLOR=0    # 0 to 255


def load_image_grayscale(name):
    """
    Returns a color image in grayscale
    :param name:
    :return:
    """
    img = cv2.imread(os.path.join(ORIGINAL_IMAGES_PATH, name),0)
    return img


def save_processed_image(name, img, prefix=""):
    """
    Saves an img in PROCESSED_IMAGES_PATH with the supplisuperiored name.
    :param name:
    :param img:
    :return:
    """
    path = os.path.join(PROCESSED_IMAGES_PATH, prefix + name)
    cv2.imwrite(path, img)
    return path


def add_padding(img, padding):
    """
    Adds a 0-padding to each size of the image.
    :param img:
    :param padding:
    :return:
    """
    return np.pad(img, padding, mode="constant")


def apply_filter(img, thresholds, filter_size=7, padding=3, stride=7):
    """
    Apply a ones matrix filter of size filter_size along the image and returns a image of 0's where the value of the
    pixel is lower than a threshold and 1's otherwise.
    :param thresholds:
    :param img:
    :param filter_size:
    :param padding:
    :param stride:
    :return:
    """
    padded_image = add_padding(img, padding)
    windows = view_as_windows(padded_image, (filter_size, filter_size), stride)
    processed_image = np.sum(windows, axis=(2,3))/(filter_size**2)

    values = np.zeros(processed_image.shape)

    for t in thresholds:
        values += (processed_image>t)*1

    return values*255/len(thresholds)


def draw_lines_grayscale(image, filter_size):
    for i in range(image.shape[0]):
        if i % filter_size == 0:
            image[i, :] = np.ones(image.shape[1])*LINE_COLOR

    for j in range(image.shape[1]):
        if j % filter_size == 0:
            image[:, j] = np.ones(image.shape[0])*LINE_COLOR

    return image


def img_to_ncolors(name, thresholds, filter_size=7, draw_lines=True):
    """
    Loads an image, apply the desired filter, and saves it in the processed images folder, returning its path.
    :param thresholds:
    :param name:
    :param filter_size:
    :param keep_dims:
    :return:
    """
    img = load_image_grayscale(name)

    processed_img = apply_filter(img, thresholds, filter_size=filter_size, padding=int(filter_size/2),
                                 stride=filter_size)

    processed_img = cv2.resize(processed_img, dsize=(processed_img.shape[1]*filter_size,
                                                     processed_img.shape[0]*filter_size),
                               interpolation=cv2.INTER_NEAREST)

    if draw_lines:
        processed_img = draw_lines_grayscale(processed_img, filter_size)

    return save_processed_image(name, processed_img, prefix=str(len(thresholds)+1)+"colors_")


if __name__=="__main__":

    # File must be located in data/original_images/
    name = "girl_dancing.jpg"

    processed_ncolors_path = img_to_ncolors(name, thresholds=[50, 100, 150, 200, 230], filter_size=15)
    print("Processed images succesfully.")