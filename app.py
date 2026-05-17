import os
import uuid
import json
import base64
from PIL import Image
from flask import Flask, render_template, request, redirect, send_from_directory
from openai import OpenAI

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploadimages')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

SYSTEM_PROMPT = """You are an expert plant pathologist and botanist with deep knowledge of plant diseases worldwide.
When given an image of a plant, leaf, fruit, stem, or any plant part, analyze it and respond ONLY with a valid JSON object.

If the image clearly shows a plant or plant part, return:
{
  "display_name": "Short disease name or 'Healthy' (e.g. 'Mango Anthracnose', 'Healthy Tomato')",
  "plant": "Common plant name (e.g. 'Mango', 'Tomato', 'Rice')",
  "type": "One of: Fungal Disease | Bacterial Disease | Viral Disease | Pest Damage | Nutrient Deficiency | Healthy | Other",
  "severity": "One of: Low | Moderate | High | Critical | None",
  "affected_parts": "Comma-separated list (e.g. 'Leaves, Fruit') or 'None' if healthy",
  "symptoms": "2-3 sentences describing visible symptoms",
  "cause": "1-2 sentences on what causes this disease/condition",
  "cure": "2-3 sentences on treatment steps",
  "prevention": "2-3 sentences on how to prevent it",
  "season": "When it typically occurs (e.g. 'Monsoon Season', 'Year-round') or 'N/A' if healthy",
  "confidence": <integer 0-100 representing your confidence>
}

If the image does NOT show a plant at all (e.g. it's a person, car, food, random object), return:
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
}

Return ONLY the JSON. No markdown, no explanation, no code blocks."""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(image_path):
    """Encode image to base64, resize if too large to save API costs."""
    img = Image.open(image_path).convert('RGB')
    # Resize to max 1024px on longest side
    max_size = 1024
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        img.save(image_path, quality=85)

    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def gpt_predict(image_path):
    """Send image to GPT-4o Vision and parse the JSON response."""
    ext = image_path.rsplit('.', 1)[-1].lower()
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'

    b64 = encode_image(image_path)

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime};base64,{b64}',
                            'detail': 'high'
                        }
                    },
                    {
                        'type': 'text',
                        'text': 'Analyze this plant image and return the JSON diagnosis.'
                    }
                ]
            }
        ],
        max_tokens=800,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if GPT wraps in them anyway
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # Ensure all expected keys exist with fallbacks
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
        prediction = gpt_predict(temp_path)
        image_url = f'/uploadimages/{temp_filename}'
    except json.JSONDecodeError:
        os.remove(temp_path)
        return render_template('home.html', error='AI returned an unexpected response. Please try again.')
    except Exception as e:
        os.remove(temp_path)
        err = str(e)
        if 'api_key' in err.lower() or 'authentication' in err.lower():
            return render_template('home.html', error='OpenAI API key not configured. Please contact the site admin.')
        return render_template('home.html', error=f'Analysis failed: {err}')

    return render_template('home.html', result=True, imagepath=image_url, prediction=prediction)


if __name__ == '__main__':
    app.run()
