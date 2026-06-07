import cv2
from pose_estimation import estimate_camera_pose, estimate_3d_point_position
from optimize import optimize_map, BA_CHECK_INTERVAL, MIN_FRAMES

def process_frame(img, extractor, matcher, display, K, W, H, map):
  img = cv2.resize(img, (W, H))
  feats = extractor.extract(img)
  frame = img.copy()

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

  inlier_mask, R, t, pose_mask = estimate_camera_pose(pts_prev, pts_curr, K)

  if R is None or t is None:
    display.paint(vis)
    map.display()
    return

  estimate_3d_point_position(map, frame, K, pts_prev, pts_curr, R, t, inlier_mask, pose_mask)

  if len(map.frames) >= MIN_FRAMES and len(map.frames) % BA_CHECK_INTERVAL == 0:
    optimize_map(map, K)

  display.paint(vis)
  map.display()
