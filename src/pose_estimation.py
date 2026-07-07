import cv2
import numpy as np
from frame import Frame
from point import Point
from triangulate import triangulate

def estimate_camera_pose(pts_prev, pts_curr, K, reverse=False):
  if pts_prev is None or pts_curr is None:
    return None, None, None, None

  if len(pts_prev) < 5 or len(pts_curr) < 5:
    return None, None, None, None

  if reverse:
    pts1, pts2 = pts_curr, pts_prev
  else:
    pts1, pts2 = pts_prev, pts_curr

  E, inlier_mask = cv2.findEssentialMat(
    pts1,
    pts2,
    cameraMatrix=K,
    method=cv2.USAC_MAGSAC,
    prob=0.999,
    threshold=1.0,
  )

  if E is None or inlier_mask is None:
    return None, None, None, None

  _, R, t, pose_mask = cv2.recoverPose(
    E, pts1, pts2, cameraMatrix=K, mask=inlier_mask
  )

  return inlier_mask, R, t, pose_mask


def triangulate_to_world(K, pose1, pose2, pts1, pts2):
  pts4d = triangulate(K, pose1, pose2, pts1, pts2)
  w = pts4d[:, 3]
  valid = np.isfinite(pts4d).all(axis=1) & (np.abs(w) > 1e-8)
  pts3d = np.zeros((len(pts4d), 3))
  pts3d[valid] = pts4d[valid, :3] / w[valid, None]
  return pts3d, valid


def median_scene_depth(K, pose1, pose2, pts1, pts2):
  pts3d, valid = triangulate_to_world(K, pose1, pose2, pts1, pts2)
  depths = []
  for p in pts3d[valid]:
    z = (pose1 @ np.append(p, 1.0))[2]
    if z > 0:
      depths.append(z)
  if not depths:
    return None
  return float(np.median(depths))

def camera_center(pose):
  return np.linalg.inv(pose)[:3, 3]


def sample_pixel_color(frame, uv):
  x, y = int(round(uv[0])), int(round(uv[1]))
  x = np.clip(x, 0, frame.shape[1] - 1)
  y = np.clip(y, 0, frame.shape[0] - 1)
  b, g, r = frame[y, x]
  return np.array([r, g, b], dtype=np.uint8)

def estimate_3d_point_position(
  map,
  frame,
  K,
  pts_prev,
  pts_curr,
  R,
  t,
  inlier_mask,
  pose_mask,
  segmenter=None,
  segmentation_map=None,
  reverse=False,
):
  t_vec = t.flatten()

  mask = (inlier_mask.ravel() > 0) & (pose_mask.ravel() > 0)
  pts1 = pts_prev[mask]
  pts2 = pts_curr[mask]
  if len(pts1) == 0:
    return

  relative = np.eye(4)
  relative[:3, :3] = R

  if len(map.frames) == 0:
    map.frames.append(Frame(map, frame, K, pose=np.eye(4)))

  prev_pose = map.frames[-1].pose

  if not map.scale_initialized:
    rel_unit = relative.copy()
    rel_unit[:3, 3] = t_vec
    if reverse:
      provisional = np.linalg.inv(rel_unit) @ prev_pose
    else:
      provisional = rel_unit @ prev_pose
    depth = median_scene_depth(K, prev_pose, provisional, pts1, pts2)
    if depth is not None and depth > 1e-3:
      map.depth_scale = depth
      map.scale_initialized = True

  relative[:3, 3] = t_vec * map.depth_scale
  if reverse:
    new_pose = np.linalg.inv(relative) @ prev_pose
  else:
    new_pose = relative @ prev_pose
  map.frames.append(Frame(map, frame, K, pose=new_pose))

  f1 = map.frames[-2]
  f2 = map.frames[-1]
  
  if reverse:
    baseline = np.linalg.norm(camera_center(f2.pose) - camera_center(f1.pose))
    if baseline < 1e-3:
      return

  pts3d, valid = triangulate_to_world(K, f1.pose, f2.pose, pts1, pts2)

  for i, p in enumerate(pts3d):
    if not valid[i]:
      continue
    X = np.append(p, 1.0)

    if reverse:
      z1 = (f1.pose @ X)[2]
      z2 = (f2.pose @ X)[2]
      if z1 <= 0.1 or z2 <= 0.1:
        continue
      if z1 > 300.0 or z2 > 300.0:
        continue

    color = sample_pixel_color(frame, pts2[i])
    semantic_info = None
    if segmenter is not None and segmentation_map is not None:
      semantic_info = segmenter.sample_target_class(segmentation_map, pts2[i])

    point = Point(
      map,
      X,
      color,
      semantic_label=None if semantic_info is None else semantic_info["label"],
      semantic_name=None if semantic_info is None else semantic_info["name"],
      semantic_color=None if semantic_info is None else semantic_info["color"],
    )
    point.add_observation(f1, pts1[i])
    point.add_observation(f2, pts2[i])
    map.points.append(point)
