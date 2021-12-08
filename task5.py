import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt
import os

def get_color(img):
    img_dir = "images/"
    # BGR
    bgr_lower = np.array([102, 255, 255])
    bgr_upper = np.array([102, 255, 255])
    bgr_result = bgr_extraction(img, bgr_lower, bgr_upper)

    # HSV
    hsv_lower = np.array([30, 153, 255])
    hsv_upper = np.array([30, 153, 255])
    hsv_result = hsv_extraction(img, hsv_lower, hsv_upper)
    imgs = cv2.hconcat([bgr_result, hsv_result])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, imgs)
    return img_path


def bgr_extraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper)  
    result = cv2.bitwise_and(image, image, mask=img_mask) 
    return result

def hsv_extraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper) 
    result = cv2.bitwise_and(image, image, mask=hsv_mask)
    return result
