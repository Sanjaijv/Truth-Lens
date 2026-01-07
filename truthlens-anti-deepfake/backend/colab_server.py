# Install dependencies
# !pip install flask flask-cors pyngrok opencv-python-headless numpy

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

app = Flask(__name__)
# Enable CORS for all domains so extension can call it
# Enable CORS for all domains
CORS(app) 

# EXPLICIT CORS BYPASS (The Nuclear Option)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response 

# --- NGROK AUTH SETUP ---
# 1. Sign up at https://dashboard.ngrok.com/signup (Free)
# 2. Get your Authtoken at https://dashboard.ngrok.com/get-started/your-authtoken
# 3. Replace "YOUR_AUTHTOKEN_HERE" below with your actual token
ngrok.set_auth_token("37fClHkFQ3NhugV6U8nrSPCnxMI_2oseTrwm5HCzKaBqcJZtG") 
# ------------------------ 

# Load a simple Face Detector for demo (since we don't have the sophisticated model files here yet)
# In production, you would load your .h5 or PyTorch model here.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return "TruthLens Cloud Analysis Running!"

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data"}), 400

        # Decode Base64
        # Format: "data:image/jpeg;base64,......"
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # 1. Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        found_faces = len(faces) > 0
        
        # 2. Fake Detection Logic (Placeholder/Simple)
        # Real Deepfake detection would go here. 
        # For now, let's allow the extension to control the "Fake" status for testing? 
        # Or return a random score weighted by image properties?
        
        # Let's emit a valid result object
        risk_score = 0.1 # Safe by default
        
        # Simple heuristic: If image is too blurry/noisy = higher risk
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100: # Blurry
            risk_score += 0.3
            
        result = {
            "found": found_faces,
            "faces_count": len(faces),
            "risk_score": risk_score,
            "verdict": "Fake" if risk_score > 0.6 else "Safe",
            "details": {
                "sharpness": laplacian_var
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set your Authtoken if you haven't !ngrok authtoken ...
    # port = 5000
    public_url = ngrok.connect(5000).public_url
    print(f" * Public URL: {public_url}")
    print(f" * Copy this URL into the TruthLens Extension!")
    app.run(port=5000)
