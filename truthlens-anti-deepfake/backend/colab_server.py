import os, cv2, numpy as np, base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

app = Flask(__name__)
CORS(app) # Enable CORS

# NUCLEAR CORS FIX (REQUIRED!)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def home(): return "TruthLens Running"

@app.route('/analyze-frame', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS': return jsonify({}), 200 # Handle Preflight
    try:
        data = request.json
        if not data or 'image' not in data: return jsonify({"error": "No image"}), 400
        
        img_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: return jsonify({"error": "Bad image"}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        found = len(faces) > 0
        risk = 0.1
        
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100: risk += 0.4
        
        return jsonify({
            "found": found,
            "risk_score": risk,
            "verdict": "Fake" if risk > 0.6 else "Safe"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    url = ngrok.connect(5000).public_url
    print(f"NEW URL: {url}")
    app.run(port=5000)
