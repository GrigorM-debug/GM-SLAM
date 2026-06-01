import cv2
import numpy as np
from frame import Frame
from point import Point
from triangulate import triangulate

def estimate_camera_pose(pts_prev, pts_curr, K):
  if pts_prev is None or pts_curr is None:
    return None, None, None, None

  if len(pts_prev) < 5 or len(pts_curr) < 5:
    return None, None, None, None
  
  E, inlier_mask = cv2.findEssentialMat(
              pts_prev, 
              pts_curr, 
              cameraMatrix=K,
              method=cv2.USAC_MAGSAC,
              prob=0.999,   
              threshold=1.0)
  
  if E is None or inlier_mask is None:
    return None, None, None, None
  
  print(f"Essential matrix: {E}")

  valid_points, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, cameraMatrix=K, mask=inlier_mask)

  print(f"Rotation matrix: {R}")
  print(f"translation vector: {t}")

  return inlier_mask, R, t, pose_mask
        

def estimate_3d_point_position(map, img, K, pts_prev, pts_curr, R, t, inlier_mask, pose_mask):
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


