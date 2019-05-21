import cv2
import os
import numpy as np
from skimage.util.shape import view_as_windows

from config import ORIGINAL_IMAGES_PATH, PROCESSED_IMAGES_PATH

def load_image_grayscale(name):
    """
    Returns a color image in grayscale
    :param name:
    :return:
    """
    img = cv2.imread(get_original_image_path(name),0)
    return img


def save_processed_image(name, img, prefix=""):
    """
    Saves an img in PROCESSED_IMAGES_PATH with the supplisuperiored name.
    :param name:
    :param img:
    :return:
    """
    path = get_processed_image_path(name, prefix)
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


def apply_filter(img, filter_size=7, padding=3, stride=7, threshold=100):
    """
    Apply a ones matrix filter of size filter_size along the image and returns a image of 0's where the value of the
    pixel is lower than a threshold and 1's otherwise.
    :param img:
    :param filter_size:
    :param padding:
    :param stride:
    :param threshold:
    :return:
    """
    padded_image = add_padding(img, padding)
    windows = view_as_windows(padded_image, (filter_size, filter_size), stride)
    processed_image = np.sum(windows, axis=(2,3))/(filter_size**2)
    processed_image = (processed_image>threshold)*255
    return processed_image


def img_to_binary(name, filter_size=7, threshold=100, keep_dims=True):
    """
    Loads an image, apply the desired filter, and saves it in the processed images folder, returning its path.
    :param name:
    :param filter_size:
    :param threshold:
    :param keep_dims:
    :return:
    """
    img = load_image_grayscale(name)
    processed_img = apply_filter(img, filter_size=filter_size, padding=int(filter_size/2), stride=filter_size,
                               threshold=threshold)
    if keep_dims:
        processed_img = np.array(processed_img, dtype='uint8')
        processed_img = cv2.resize(processed_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return save_processed_image(name, processed_img, prefix="2colors_")


def img_to_3colors(name, filter_size=7, threshold1=100, threshold2=200, keep_dims=True):
    """
    Loads an image, apply the desired filter, and saves it in the processed images folder, returning its path.
    :param name:
    :param filter_size:
    :param threshold:
    :param keep_dims:
    :return:
    """
    img = load_image_grayscale(name)
    grey_spots = apply_filter(img, filter_size=filter_size, padding=int(filter_size/2), stride=filter_size,
                               threshold=threshold1)
    dark_spots = apply_filter(img, filter_size=filter_size, padding=int(filter_size/2), stride=filter_size,
                               threshold=threshold2)
    if keep_dims:
        processed_img = np.array(grey_spots, dtype='uint8')/2+np.array(dark_spots, dtype='uint8')/2
        processed_img = cv2.resize(processed_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return save_processed_image(name, processed_img, prefix="3colors_")


def get_original_image_path(name):
    return os.path.join(ORIGINAL_IMAGES_PATH, name)


def get_processed_image_path(name, prefix=""):
    return os.path.join(PROCESSED_IMAGES_PATH, prefix+name)
