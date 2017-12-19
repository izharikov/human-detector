import os, requests
import cv2
from object_detector.detector import detect_img
import numpy as np

from PIL import Image
from io import BytesIO

from flask import Flask, request, render_template, redirect
import pedestrian

app = Flask(__name__)

proceed_images_path = "static"


@app.route('/', methods=['GET', 'POST'])
def guess():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
            process_image(img, file.filename)
            return redirect(str.format('/?img_url={0}', file.filename))
    return render_template('index.html', sample_list=[1, 2, 3, 4])


def process_image(file, filename, func_to_process=pedestrian.process_image):
    img = func_to_process(file)
    cv2.imwrite(os.path.join(proceed_images_path, 'before', filename), file)
    cv2.imwrite(os.path.join(proceed_images_path, 'after', filename), img)
    return


@app.route('/custom', methods=['GET', 'POST'])
def guess_custom():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
            process_image(img, file.filename, func_to_process=detect_img)
            return redirect(str.format('/custom?img_url={0}', file.filename))
    return render_template('index.html', sample_list=[1, 2, 3, 4])


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=3230)
