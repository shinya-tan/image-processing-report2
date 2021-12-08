from flask import Flask
import os
import io
import numpy as np
import task1
import task2
import task3
import task4
import task5
import cv2
from flask import Flask, render_template, request

UPLOAD_FOLDER = './images'

app = Flask(__name__, static_folder=UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return ''' <p>Hello World</p> '''


@app.route('/send', methods=['GET', 'POST'])
def send():
    show_image_url = None
    gray_img_url = None
    binarization_url = None
    resize_url = None
    draw_shape_url = None
    save_image_url = None
    flip_image_url = None
    blur_filter_url = None
    gaussian_filter_url = None
    sobel_filter_url = None
    canny_filter_url = None
    perspective_transform_url = None
    histogram_url = None
    histogram_graph_url = None
    trimming_url = None
    if request.method == 'POST':
        img_file = request.files['img']

        if not(img_file and allowed_file(img_file.filename)):
            return ''' <p>許可されていない拡張子です</p> '''
        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        show_image_url = task1.show_image(img)
        gray_img_url = task1.image_to_gray(img)
        binarization_url = task1.binarization(img)
        resize_url = task1.resize(img)
        trimming_url = task1.trimming(img)
        draw_shape_url = task1.draw_shape()
        save_image_url = task1.save_image(img)
        flip_image_url = task1.flip_image(img)
        blur_filter_url = task1.blur_filter(img)
        gaussian_filter_url = task1.gaussian_filter(img)
        sobel_filter_url = task1.sobel_filter(img)
        canny_filter_url = task1.canny_filter(img)
        perspective_transform_url = task1.perspective_transform(img)
        histogram_url, histogram_graph_url = task1.histogram(img)
    return render_template('task1.html', show_image_url=show_image_url,
                           gray_img_url=gray_img_url, binarization_url=binarization_url,
                           resize_url=resize_url, trimming_url=trimming_url, draw_shape_url=draw_shape_url,
                           save_image_url=save_image_url, flip_image_url=flip_image_url,
                           blur_filter_url=blur_filter_url, gaussian_filter_url=gaussian_filter_url,
                           sobel_filter_url=sobel_filter_url, canny_filter_url=canny_filter_url,
                           perspective_transform_url=perspective_transform_url,
                           histogram_url=histogram_url, histogram_graph_url=histogram_graph_url)


@app.route('/send2', methods=['GET', 'POST'])
def send2():
    gamma_graph_url = None
    gamma_calc_url = None
    gamma_calc_extension_url = None
    if request.method == 'POST':
        img_file = request.files['img']
        value = request.form['value']
        if not(img_file and allowed_file(img_file.filename)):
            return ''' <p>許可されていない拡張子です</p> '''
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gamma_graph_url = task2.gamma_graph()
        gamma_calc_url = task2.gamma_calc(img)
        gamma_calc_extension_url = task2.gamma_calc_extension(
            img, float(value))
    return render_template('task2.html', gamma_graph_url=gamma_graph_url, gamma_calc_url=gamma_calc_url, gamma_calc_extension_url=gamma_calc_extension_url)


@app.route('/send3', methods=['GET', 'POST'])
def send3():
    add_weighted_url = None
    bitwise_and_url = None
    add_weighted_for_np_url = None
    add_weighted_for_np_and_clip_url = None
    mask_for_np_url = None
    complicate_mask_alpha_blend_url = None
    complicate_mask_alpha_blend_more_file_url = None
    draw_mask_url = None
    draw_mask_image_url = None
    if request.method == 'POST':
        img_file1 = request.files['img1']
        img_file2 = request.files['img2']
        mask_file = request.files['mask']
        f1 = img_file1.stream.read()
        bin_data = io.BytesIO(f1)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        f2 = img_file2.stream.read()
        bin_data = io.BytesIO(f2)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        f3 = mask_file.stream.read()
        bin_data = io.BytesIO(f3)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        mask = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        add_weighted_url = task3.add_weighted(img1, img2)
        bitwise_and_url = task3.bitwise_and(img1,mask)
        add_weighted_for_np_url = task3.add_weighted_for_np()
        add_weighted_for_np_and_clip_url = task3.add_weighted_for_np_and_clip()
        mask_for_np_url = task3.mask_for_np()
        complicate_mask_alpha_blend_url = task3.complicate_mask_alpha_blend()
        complicate_mask_alpha_blend_more_file_url = task3.complicate_mask_alpha_blend_more_file()
        draw_mask_url, draw_mask_image_url = task3.draw_mask(img1)
    return render_template('task3.html', add_weighted_url=add_weighted_url, bitwise_and_url=bitwise_and_url,
                           add_weighted_for_np_url=add_weighted_for_np_url, add_weighted_for_np_and_clip_url=add_weighted_for_np_and_clip_url,
                           mask_for_np_url=mask_for_np_url, complicate_mask_alpha_blend_url=complicate_mask_alpha_blend_url,
                           complicate_mask_alpha_blend_more_file_url=complicate_mask_alpha_blend_more_file_url,
                           draw_mask_url=draw_mask_url, draw_mask_image_url=draw_mask_image_url)



@app.route('/send4', methods=['GET', 'POST'])
def send4():
    get_corner_url = None
    good_features_to_track_url = None
    if request.method == 'POST':
        img_file = request.files['img']
        if not(img_file and allowed_file(img_file.filename)):
            return ''' <p>許可されていない拡張子です</p> '''
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        get_corner_url = task4.get_corner(img)
        good_features_to_track_url = task4.good_features_to_track(img)
    return render_template('task4.html', get_corner_url=get_corner_url, good_features_to_track_url=good_features_to_track_url)

@app.route('/send5', methods=['GET', 'POST'])
def send5():
    get_color_url = None
    if request.method == 'POST':
        img_file = request.files['img']
        if not(img_file and allowed_file(img_file.filename)):
            return ''' <p>許可されていない拡張子です</p> '''
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        get_color_url = task5.get_color(img)
    return render_template('task5.html', get_color_url=get_color_url)
if __name__ == '__main__':
    app.debug = True
    app.run()
