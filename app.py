import os
import uuid
import json
from PIL import Image
from flask import Flask, render_template, request, redirect, send_from_directory
import google.generativeai as genai

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploadimages')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

PROMPT = """You are an expert plant pathologist and botanist.
Analyze this image and respond ONLY with a valid JSON object — no markdown, no code fences, no explanation.

If the image shows a plant or any plant part (leaf, fruit, stem, root, crop, tree, flower):
{
  "display_name": "Short disease name or 'Healthy' (e.g. 'Mango Anthracnose', 'Healthy Tomato')",
  "plant": "Common plant name (e.g. 'Mango', 'Tomato', 'Rice')",
  "type": "One of: Fungal Disease | Bacterial Disease | Viral Disease | Pest Damage | Nutrient Deficiency | Healthy | Other",
  "severity": "One of: Low | Moderate | High | Critical | None",
  "affected_parts": "Comma-separated list (e.g. 'Leaves, Fruit') or 'None' if healthy",
  "symptoms": "2-3 sentences describing visible symptoms",
  "cause": "1-2 sentences on what causes this",
  "cure": "2-3 sentences on treatment steps",
  "prevention": "2-3 sentences on prevention",
  "season": "When it typically occurs (e.g. 'Monsoon Season') or 'N/A' if healthy",
  "confidence": <integer 0-100>
}

If the image does NOT show a plant at all:
{
  "display_name": "Not a Plant",
  "plant": "Unknown",
  "type": "Detection Error",
  "severity": "None",
  "affected_parts": "N/A",
  "symptoms": "The uploaded image does not appear to contain a plant or plant part.",
  "cause": "N/A",
  "cure": "Please upload a clear photo of a plant, leaf, fruit, or crop.",
  "prevention": "N/A",
  "season": "N/A",
  "confidence": 0
}"""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gemini_predict(image_path):
    # Resize large images to save quota
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if max(w, h) > 1024:
        ratio = 1024 / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        img.save(image_path, quality=85)

    img = Image.open(image_path)
    response = model.generate_content([PROMPT, img])
    raw = response.text.strip()

    # Strip markdown fences if Gemini wraps them anyway
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # Ensure all keys exist
    defaults = {
        'display_name': 'Unknown',
        'plant': 'Unknown',
        'type': 'Unknown',
        'severity': 'N/A',
        'affected_parts': 'N/A',
        'symptoms': 'N/A',
        'cause': 'N/A',
        'cure': 'N/A',
        'prevention': 'N/A',
        'season': 'N/A',
        'confidence': 0,
    }
    for k, v in defaults.items():
        result.setdefault(k, v)

    return result


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
        return render_template('home.html', error='Please upload a valid image (png, jpg, jpeg, webp).')

    ext = image.filename.rsplit('.', 1)[1].lower()
    temp_filename = f"upload_{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    image.save(temp_path)

    try:
        prediction = gemini_predict(temp_path)
        image_url = f'/uploadimages/{temp_filename}'
    except json.JSONDecodeError:
        os.remove(temp_path)
        return render_template('home.html', error='AI returned an unexpected response. Please try again.')
    except Exception as e:
        os.remove(temp_path)
        err = str(e)
        if 'api_key' in err.lower() or 'api key' in err.lower():
            return render_template('home.html', error='Gemini API key not configured. Please contact the site admin.')
        return render_template('home.html', error=f'Analysis failed: {err}')

    return render_template('home.html', result=True, imagepath=image_url, prediction=prediction)


if __name__ == '__main__':
    app.run()
