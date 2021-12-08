import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image


def add_weighted(img1, img2):
    img2 = cv2.resize(img2, img1.shape[1::-1])
    dst1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    dst2 = cv2.addWeighted(img1, 0.5, img2, 0.2, 128)
    imgs = cv2.hconcat([dst1, dst2])

    img_path = calc_image_path()
    cv2.imwrite(img_path, imgs)
    cv2.imwrite("images/image1.jpeg", img1)
    cv2.imwrite("images/image2.jpeg", img2)
    return img_path


def bitwise_and(img1, mask):
    img2 = cv2.resize(mask, img1.shape[1::-1])
    dst = cv2.bitwise_and(img1, img2)
    img_path = calc_image_path()
    cv2.imwrite(img_path, dst)
    cv2.imwrite("images/mask1.jpeg", mask)
    return img_path


def add_weighted_for_np():
    img1 = np.array(Image.open('images/image1.jpeg'))
    img2 = np.array(Image.open(
        'images/image2.jpeg').resize(img1.shape[1::-1], Image.BILINEAR))
    dst = img1 * 0.5 + img2 * 0.5
    img_path = calc_image_path()
    Image.fromarray(dst.astype(np.uint8)).save(img_path)
    return img_path


def add_weighted_for_np_and_clip():
    img1 = np.array(Image.open('images/image1.jpeg'))
    img2 = np.array(Image.open(
        'images/image2.jpeg').resize(img1.shape[1::-1], Image.BILINEAR))
    dst = img1 * 0.5 + img2 * 0.2 + (96, 128, 160)
    dst = dst.clip(0, 255)
    img_path = calc_image_path()
    Image.fromarray(dst.astype(np.uint8)).save(img_path)
    return img_path


def mask_for_np():
    img = np.array(Image.open('images/image1.jpeg'))
    mask = np.array(Image.open(
        'images/image2.jpeg').resize(img.shape[1::-1], Image.BILINEAR))
    mask = mask / 255
    dst = img * mask
    img_path = calc_image_path()
    Image.fromarray(dst.astype(np.uint8)).save(img_path)
    return img_path


def complicate_mask_alpha_blend():
    array = get_gradient_3d(512, 256, (0, 0, 0),
                            (255, 255, 255), (True, True, True))
    Image.fromarray(np.uint8(array)).save(
        'images/gray_gradient_h.jpg', quality=95)
    img1 = np.array(Image.open('images/image1.jpeg'))
    img2 = np.array(Image.open(
        'images/image2.jpeg').resize(img1.shape[1::-1], Image.BILINEAR))
    mask = np.array(Image.open(
        'images/gray_gradient_h.jpg').resize(img1.shape[1::-1], Image.BILINEAR))
    mask = mask / 255
    dst = img1 * mask + img2 * (1 - mask)
    img_path = calc_image_path()
    Image.fromarray(dst.astype(np.uint8)).save(img_path)
    return img_path


def complicate_mask_alpha_blend_more_file():
    img1 = np.array(Image.open('images/image1.jpeg'))
    img2 = np.array(Image.open(
        'images/image2.jpeg').resize(img1.shape[1::-1], Image.BILINEAR))
    mask1 = np.array(Image.open(
        'images/gray_gradient_h.jpg').resize(img1.shape[1::-1], Image.BILINEAR))
    mask2 = np.array(Image.open(
        'images/mask1.jpeg').resize(img1.shape[1::-1], Image.BILINEAR))
    mask1 = mask1 / 255
    mask2 = mask2 / 255
    dst = (img1 * mask1 + img2 * (1 - mask1)) * mask2
    img_path = calc_image_path()
    Image.fromarray(dst.astype(np.uint8)).save(img_path)
    return img_path


def draw_mask(img):
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (50, 50), (100, 200), (255, 255, 255), thickness=-1)
    cv2.circle(mask, (200, 100), 50, (255, 255, 255), thickness=-1)
    cv2.fillConvexPoly(mask, np.array(
        [[330, 50], [300, 200], [360, 150]]), (255, 255, 255))
    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
    dst = img * (mask_blur / 255)
    imgs = cv2.hconcat([mask, mask_blur])
    img_path = calc_image_path()
    img_dir = "images/"
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path_2 = os.path.join(img_dir, 'draw_mask_' + dt_now+'.jpeg')
    cv2.imwrite(img_path, imgs)
    cv2.imwrite(img_path_2, dst)
    return img_path, img_path_2


def calc_image_path():
    img_dir = "images/"
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    return os.path.join(img_dir, dt_now+'.jpeg')


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(
            start, stop, width, height, is_horizontal)

    return result


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
