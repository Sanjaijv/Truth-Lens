import torch
import cv2
import numpy as np
from quickScan import QuickScanModel

device = "cpu"
model = QuickScanModel(pretrained=True).to(device)
model.eval()

def load_video(video_path, max_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32") / 255.0
        frame = frame.transpose(2, 0, 1)
        frames.append(frame)

    cap.release()
    return np.stack(frames)

def run_quick_scan(video_path):
    frames_np = load_video(video_path)
    frames = torch.tensor(frames_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(frames)
        prob = torch.sigmoid(logits).item()

    return {
        "ai_probability": prob,
        "verdict": "AI-Generated" if prob > 0.5 else "Real"
    }
