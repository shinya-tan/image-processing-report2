import os
import datetime
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib
matplotlib.use('Agg')

img_dir = "images/"


def gamma_graph():
    plt.clf()
    ary_gamma = np.array([0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25])

    for var_gamma in ary_gamma:
        var_px = np.empty(256, np.uint8)
        for i in range(256):
            var_px[i] = np.clip(pow(i / 255.0, var_gamma) * 255.0, 0, 255)
        plt.plot(var_px, label=str(var_gamma))

    plt.legend()
    plt.xlabel("INPUT")
    plt.ylabel("OUTPUT")
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    plt.savefig(img_path)
    return img_path


def gamma_calc(img):
    gamma = 0.8
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    imgA = cv2.LUT(img, lookUpTable)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    imgs = cv2.hconcat([img, imgA])
    cv2.imwrite(img_path, imgs)
    return img_path


def gamma_calc_extension(img, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    imgA = cv2.LUT(img, lookUpTable)
    imgA_RGB = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgP = Image.fromarray(imgA_RGB)
    obj_draw = ImageDraw.Draw(imgP)
    courier_font = ImageFont.truetype("/System/Library/Fonts/Courier.ttc", 40)
    obj_draw.text((10, 10), f"{str(gamma)}",fill='white', font=courier_font)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    imgP.save(img_path)
    return img_path
