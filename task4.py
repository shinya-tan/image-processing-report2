import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt
import os

img_dir = "images/"

def get_corner(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,2,3,0.04)

  dst = cv2.dilate(dst,None)

  img[dst>0.01*dst.max()] = [0,0,255]

  dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
  img_path = os.path.join(img_dir, dt_now+'.jpeg')
  cv2.imwrite(img_path, img)
  return img_path

def good_features_to_track(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
  corners = np.int0(corners)

  for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
  dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
  img_path = os.path.join(img_dir, dt_now+'.jpeg')
  plt.imshow(img)
  plt.savefig(img_path)
  return img_path