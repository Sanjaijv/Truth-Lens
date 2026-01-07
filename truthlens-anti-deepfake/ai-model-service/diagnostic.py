import os
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = 'trained_models/background_autoencoder.h5'
VIDEO_PATH = '../backend/uploads'

def get_latest_video():
    videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    if not videos: return None
    videos.sort(key=lambda x: os.path.getmtime(os.path.join(VIDEO_PATH, x)), reverse=True)
    return os.path.join(VIDEO_PATH, videos[0])

def diagnostic():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    video_path = get_latest_video()
    if not video_path:
        print("No video found.")
        return

    print(f"Analyzing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, (224, 224))
        normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        frames.append(normalized)
    cap.release()

    frames = np.array(frames).astype(np.float32)
    reconstructions = model.predict(frames)
    errors = np.mean(np.square(frames - reconstructions), axis=(1, 2, 3))
    
    print("\nRaw MSE per frame:")
    for i, e in enumerate(errors):
        print(f"Frame {i}: {e:.8f}")
    
    print(f"\nAverage MSE: {np.mean(errors):.8f}")

if __name__ == "__main__":
    diagnostic()
