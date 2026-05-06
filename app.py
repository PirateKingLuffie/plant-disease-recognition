import os
import uuid
import json
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from flask import Flask, render_template, request, redirect, send_from_directory

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploadimages')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_interpreter = None

def get_interpreter():
    global _interpreter
    if _interpreter is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'plant_disease_model.tflite')
        _interpreter = tflite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()
    return _interpreter

with open(os.path.join(os.path.dirname(__file__), 'plant_disease.json'), 'r') as f:
    plant_disease = json.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((160, 160))
    img_array = np.array(img, dtype=np.float32)[np.newaxis, ...]

    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
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
