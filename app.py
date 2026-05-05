import os
import uuid
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploadimages')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'models', 'plant_disease_recog_model_pwp.keras'))

with open(os.path.join(os.path.dirname(__file__), 'plant_disease.json'), 'r') as f:
    plant_disease = json.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    return np.array([feature])


def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    return plant_disease[prediction.argmax()]


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method != 'POST':
        return redirect('/')

    image = request.files.get('img')
    if not image or not allowed_file(image.filename):
        return render_template('home.html', error='Please upload a valid image (png, jpg, jpeg).')

    ext = image.filename.rsplit('.', 1)[1].lower()
    temp_filename = f"temp_{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    image.save(temp_path)

    try:
        prediction = model_predict(temp_path)
    finally:
        os.remove(temp_path)

    return render_template('home.html', result=True, imagepath=None, prediction=prediction)


if __name__ == '__main__':
    app.run()
