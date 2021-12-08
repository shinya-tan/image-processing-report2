import os
import matplotlib.pyplot as plt
import cv2
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')


img_dir = "images/"


def show_image(img):
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path


def image_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def binarization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 80
    _, img_threshold = cv2.threshold(img, threshold, 255, 0)
    # cv2.imshow('image', img_threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img_threshold)
    return img_path


def resize(img):
    img_resize = cv2.resize(img, (150, 100))
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img_resize)
    return img_path


def trimming(img):
    img1 = img[0: 100, 0: 150]
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img1)
    return img_path


def draw_shape():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(img, (100, 100), 50, (255, 255, 0), thickness=-1)
    cv2.line(img, (0, 0), (512, 512), (0, 200, 0),
             thickness=3, lineType=cv2.LINE_4)
    cv2.rectangle(img, (255, 255), (125, 130), (100, 100, 50), thickness=-1)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path


def save_image(img):
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path


def flip_image(img):
    img = cv2.flip(img, 1)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path


def blur_filter(img):
    img_blur = cv2.blur(img, (7, 7))
    imgs = cv2.hconcat([img, img_blur])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, imgs)
    return img_path


def gaussian_filter(img):
    img_blur = cv2.GaussianBlur(img, (7, 7), sigmaX=10, sigmaY=10)
    imgs = cv2.hconcat([img, img_blur])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, imgs)
    return img_path


def sobel_filter(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    plt.subplot(1, 2, 1), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')

    plt.savefig(img_path)
    return img_path


def canny_filter(img):
    plt.clf()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 40, 150)
    plt.subplot(1, 2, 1), plt.imshow(edges, cmap='gray')
    plt.title('Edge'), plt.xticks([]), plt.yticks([])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    plt.savefig(img_path)
    return img_path

def perspective_transform(img):
    height, width = img.shape[:2]
    dst_pts = np.array([[185, 67], [380, 106], 
                [16, 146], [221, 218]], dtype=np.float32)
    src_pts = np.array([[150, 0], [width-1-50, 30], 
            [20, height-1], [width-1, height-50]], dtype=np.float32)
    get_prespective_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(img, get_prespective_transform, (width, height), flags=cv2.INTER_CUBIC)
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, img)
    return img_path

def histogram(img):
    plt.clf()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalize_hist = cv2.equalizeHist(img)
    after_img = np.hstack((img, equalize_hist))
    plt.subplot(2,1,1), plt.hist(img.ravel(),256,[0,256])
    plt.subplot(2,1,2), plt.hist(after_img.ravel(),256,[0,256])
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = os.path.join(img_dir, dt_now+'.jpeg')
    cv2.imwrite(img_path, after_img)
    img_path_graph = os.path.join(img_dir, dt_now+'graph.jpeg')
    plt.savefig(img_path_graph)
    return img_path, img_path_graph

