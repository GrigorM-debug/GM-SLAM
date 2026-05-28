import cv2
import numpy as np
from triangulate import triangulate
from point import Point

class Frame:
  def __init__(self, map, img, K, pose):
    self.id = len(map.frames)
    self.img = img
    self.K = K
    self.pose = pose

def process_frame(img, extractor, matcher, display, K, W, H, map):
  img = cv2.resize(img, (W, H))
  feats = extractor.extract(img)

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

  if pts_prev is not None and pts_curr is not None:
    # Compute the Essensial matrix 
    E, inlier_mask = cv2.findEssentialMat(
                          pts_prev, 
                          pts_curr, 
                          cameraMatrix=K,
                          method=cv2.USAC_MAGSAC,
                          prob=0.999,   
                          threshold=1.0)

    if E is not None and inlier_mask is not None:
      print(f"Essential matrix: {E}")
        
      # Recover the relative camera pose
      valid_points, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, cameraMatrix=K, mask=inlier_mask)

      print(f"Rotation matrix: {R}")
      print(f"translation vector: {t}")

      # Triangulation will be here
      pose1 = np.eye(4)
      pose2 = np.eye(4)
      pose2[:3, :3] = R
      pose2[:3, 3] = t.flatten()

      f1 = Frame(map, img, K, pose=pose1)
      f2 = Frame(map, img, K, pose=pose2)

      map.frames.append(f1)
      map.frames.append(f2)

      mask = (inlier_mask.ravel() > 0) & (pose_mask.ravel() > 0)
      pts1_inliers = pts_prev[mask]
      pts2_inliers = pts_curr[mask]
      
      if len(pts1_inliers) > 0:     
        pts4d = triangulate(pose1=f1.pose, pose2=f2.pose, pts1=pts1_inliers, pts2=pts2_inliers)
        pts4d /= pts4d[:, 3:]
    
        good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

        for i, p in enumerate(pts4d):
          if not good_pts4d[i]:
            continue

          point = Point(map, p)
          point.add_observation(f1, i) # frame 1
          point.add_observation(f2, i) # frame 2

          map.points.append(point)

  display.paint(vis)
  map.display()
