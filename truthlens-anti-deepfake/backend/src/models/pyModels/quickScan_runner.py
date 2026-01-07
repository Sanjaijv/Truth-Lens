import sys
import json
from inference import run_quick_scan

if __name__ == "__main__":
    video_path = sys.argv[1]
    result = run_quick_scan(video_path)
    print(json.dumps(result))