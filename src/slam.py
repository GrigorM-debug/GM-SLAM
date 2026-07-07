import cv2
import torch
import numpy as np
from helpers import parse_args, resolve_video_path, resolve_output_path
from process_frame import process_frame
from display2d import Display2D
from extractor import FeatureExtractor
from matcher import FeatureMatcher
from map import Map
from optimize import BAWorker
from semantic_slam import FrameSegmenter, SemanticDisplay2D

W = 1920//2
H = 1080//2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device used: {DEVICE}")

F = 270
K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

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

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    print("[ERROR] Video opened but no readable frames were found.")
    cap.release()
    return

  display = Display2D(W, H)
  extractor = FeatureExtractor(DEVICE)
  matcher = FeatureMatcher(DEVICE)
  map = Map()
  ba_worker = BAWorker()
  semantic_display = None
  segmenter = None

  if args.semantic:
    print("[INFO] Semantic segmentation enabled.")
    segmenter = FrameSegmenter(DEVICE)
    semantic_display = SemanticDisplay2D(W, H)

  if args.reverse:
    print("[INFO] Reversing the video.")
    frame_index = total_frames - 1
    step = -1
  else:
    frame_index = 0
    step = 1

  def read_frame_at(index, seek=True):
    if seek:
      cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, img = cap.read()
    if not ret or img is None:
      return False, None
    actual_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if abs(actual_index - index) > 1:
      print(
        f"[WARN] Seek requested frame {index}, decoder landed near frame {actual_index}."
      )
    return True, img

  ret, frame = read_frame_at(frame_index, seek=True)
  if not ret:
    print("[ERROR] Failed to read the first frame.")
    cap.release()
    return

  map.create_viewer(W, H, float(K[0, 0]))

  while True:
    process_frame(
      frame,
      extractor,
      matcher,
      display,
      K,
      W,
      H,
      map,
      segmenter=segmenter,
      semantic_display=semantic_display,
      ba_worker=ba_worker,
      reverse=args.reverse,
    )

    if cv2.waitKey(1) == ord('q'):
      break
    
    frame_index += step

    if frame_index < 0 or frame_index >= total_frames:
      break

    last_processed = frame.copy()
    ret, frame = read_frame_at(frame_index, seek=args.reverse)
    if not ret:
      break

    frame_diff = np.mean(np.abs(frame.astype(np.float32) - last_processed.astype(np.float32)))
    if frame_diff < 1.0:
      if args.reverse:
        print(
          f"[WARN] Duplicate frame at index {frame_index}; "
          "reverse seeking may be unreliable for this video."
        )
        break
      else:
        print(
          f"[WARN] Duplicate frame at index {frame_index}; "
          "video may contain a static/duplicate frame here, skipping."
        )
        continue

  ba_worker.shutdown()
  ba_worker.apply_results(map)

  output_path = resolve_output_path(args.output, video_path)
  map.save(output_path, K)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  slam()