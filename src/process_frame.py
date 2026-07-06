import cv2
from pose_estimation import estimate_camera_pose, estimate_3d_point_position
from optimize import BA_CHECK_INTERVAL, MIN_FRAMES

def process_frame(
  img,
  extractor,
  matcher,
  display,
  K,
  W,
  H,
  map,
  segmenter=None,
  semantic_display=None,
  ba_worker=None,
  reverse=False,
):
  img = cv2.resize(img, (W, H))
  feats = extractor.extract(img)

  if ba_worker is not None:
    ba_worker.apply_results(map)

  frame = img.copy()
  semantic_frame = None
  segmentation_map = None
  if segmenter is not None:
    semantic_frame, segmentation_map = segmenter.segment_frame(frame)

  vis, n_good, n_kpts, pts_prev, pts_curr = matcher.match_and_draw(img, feats)

  label = f"Features: {n_kpts}  Matches: {n_good}"

  font = cv2.FONT_HERSHEY_COMPLEX
  font_scale = 1
  thickness = 2

  (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
  x = (vis.shape[1] - text_w) // 2
  y = 20 + text_h

  cv2.putText(
    vis,
    label,
    (x, y),
    font,
    font_scale,
    (255, 255, 255),
    thickness,
    cv2.LINE_AA,
  )

  inlier_mask, R, t, pose_mask = estimate_camera_pose(pts_prev, pts_curr, K, reverse=reverse)

  if R is None or t is None:
    display.paint(vis)
    if semantic_display is not None and semantic_frame is not None:
      semantic_display.paint(semantic_frame)
    map.display()
    return

  estimate_3d_point_position(
    map,
    frame,
    K,
    pts_prev,
    pts_curr,
    R,
    t,
    inlier_mask,
    pose_mask,
    segmenter=segmenter,
    segmentation_map=segmentation_map,
    reverse=reverse,
  )

  # First BA once MIN_FRAMES is reached, then every BA_CHECK_INTERVAL new frames.
  frames_since_min = len(map.frames) - MIN_FRAMES
  if (
    ba_worker is not None
    and not reverse
    and frames_since_min >= 0
    and frames_since_min % BA_CHECK_INTERVAL == 0
  ):
    ba_worker.try_start(map, K)

  display.paint(vis)
  if semantic_display is not None and semantic_frame is not None:
    semantic_display.paint(semantic_frame)
  map.display()
