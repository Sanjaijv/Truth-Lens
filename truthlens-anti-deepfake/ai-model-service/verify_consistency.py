import requests
import sys
import os

def test_consistency(video_path):
    url = "http://localhost:5000/analyze-video"
    results = []
    
    print(f"Testing consistency for: {video_path}")
    
    for i in range(3):
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'scanType': 'quick'}
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                likelihood = response.json().get('aiLikelihood')
                results.append(likelihood)
                print(f"Run {i+1}: AI Likelihood = {likelihood}")
            else:
                print(f"Run {i+1}: Failed with status {response.status_code}")
                return

    if len(set(results)) == 1:
        print("\nSUCCESS: All results are identical! The model is now deterministic.")
    else:
        print(f"\nFAILURE: Results inconsistent! {results}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_consistency(sys.argv[1])
    else:
        uploads_dir = "../backend/uploads"
        videos = [f for f in os.listdir(uploads_dir) if f.endswith('.mp4')]
        if videos:
            test_consistency(os.path.join(uploads_dir, videos[0]))
        else:
            print("No video file found to test.")
