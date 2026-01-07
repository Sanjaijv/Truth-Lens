import cv2
import numpy as np

class BackgroundIrregularityDetector:
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        print("BackgroundIrregularityDetector initialized.")

    def detect(self, frames):
        irregularity_results = []
        for i, frame in enumerate(frames):
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame

            fgmask = self.fgbg.apply(frame_uint8)
            total_pixels = fgmask.shape[0] * fgmask.shape[1]
            foreground_pixels = cv2.countNonZero(fgmask)
            irregularity_score = foreground_pixels / total_pixels
            is_irregular = irregularity_score > 0.05

            irregularity_results.append({
                "frame_index": i,
                "score": round(irregularity_score, 4),
                "is_irregular": is_irregular,
                "description": f"Foreground detected: {irregularity_score*100:.2f}%"
            })
        
        print(f"Detected irregularities for {len(frames)} frames.")
        return irregularity_results

if __name__ == '__main__':
    print("Running a dummy test for BackgroundIrregularityDetector.")
    detector = BackgroundIrregularityDetector()
    dummy_frames = []
    dummy_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    frame_with_object = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(frame_with_object, (50, 50), (100, 100), (255, 255, 255), -1)
    dummy_frames.append(frame_with_object)
    results = detector.detect(dummy_frames)
    for res in results:
        print(res)
