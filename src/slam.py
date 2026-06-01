import cv2
import torch
import numpy as np
from helpers import parse_args, resolve_video_path
from process_frame import process_frame
from display2d import Display2D
from extractor import FeatureExtractor
from matcher import FeatureMatcher
from map import Map

W = 1920//2
H = 1080//2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device used: {DEVICE}")

K = np.array([[11.287663,  0.0,       480.0],
  [ 0.0,       11.287663, 270.0],
  [ 0.0,       0.0,         1.0]])

# F = 270
# K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

def slam():
  args = parse_args()
  video_path = resolve_video_path(args.video)
  print(f"[INFO] Video path: {video_path}")

  if not video_path.exists():
    print("[ERROR] Video file does not exist.")
    print("        Pass an absolute path or run with the correct relative path.")
    return

  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    print("[ERROR] OpenCV failed to open the video.")
    return

  ret, frame = cap.read()
  if not ret or frame is None:
    print("[ERROR] Video opened but no readable frames were found.")
    cap.release()
    return

  display = Display2D(W, H)
  extractor = FeatureExtractor(DEVICE)
  matcher = FeatureMatcher(DEVICE)
  map = Map()

  while True:
    process_frame(frame, extractor, matcher, display, K, W, H, map)

    if cv2.waitKey(1) == ord('q'):
      break
    
    ret, frame = cap.read()
    
    if not ret or frame is None:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  slam()