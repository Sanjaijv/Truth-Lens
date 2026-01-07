# preprocessing/extract_frames.py
import cv2
import os

def extract_frames(video_path, output_dir, max_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    extracted = []
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        extracted.append(frame_path)
        count += 1
        
    cap.release()
    return extracted
