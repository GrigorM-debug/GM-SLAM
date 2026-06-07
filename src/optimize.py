"""
This module contains the functions for bundle adjustment back-end optimization.
Only local bundle adjustment is implemented here. Meaning that only the recent frames and points are optimized.
The optization runs every BA_CHECK_INTERVAL frames and only if the reprojection error is greater than REPROJ_ERROR_THRESHOLD.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# With this params the systems stated crashing in my case
# BA_CEHCK_INTERVAL = 10
# BA_WINDOW = 10


BA_CHECK_INTERVAL = 10  # how often to measure reprojection error
BA_WINDOW = 5
REPROJ_ERROR_THRESHOLD = 0.5  # px; skip BA when error is at or below this

MIN_FRAMES = 5
MIN_POINTS = 4
MIN_OBSERVATIONS = 8

def rodrigues_to_R(rvec):
  theta = np.linalg.norm(rvec)
  if theta < 1e-10:
    return np.eye(3)
  k = rvec / theta
  K = np.array([[    0, -k[2],  k[1]],
                [ k[2],     0, -k[0]],
                [-k[1],  k[0],     0]])
  
  return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def R_to_rodrigues(R):
  theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
  if theta < 1e-10:
    return np.zeros(3)

  return (theta / (2 * np.sin(theta))) * np.array(
    [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def project(cam, pt, K):
  R  = rodrigues_to_R(cam[:3])
  t  = cam[3:6]
  Pc = R @ pt + t

  if Pc[2] <= 0:
    return np.array([np.nan, np.nan])

  u = K[0, 0] * Pc[0] / Pc[2] + K[0, 2]
  v = K[1, 1] * Pc[1] / Pc[2] + K[1, 2]

  return np.array([u, v])

def residuals(
  params, 
  n_cams, 
  n_pts, 
  camera_indices, 
  point_indices, 
  observations, 
  K
  ):
  cams = params[:n_cams * 6].reshape(n_cams, 6)
  pts  = params[n_cams * 6:].reshape(n_pts,  3)
  res  = []
  for ci, pi, ob in zip(camera_indices, point_indices, observations):
      uv = project(cams[ci], pts[pi], K)
      res.append(uv - ob)
  return np.concatenate(res) 

def build_sparsity(n_cams, n_pts, camera_indices, point_indices):
  n_obs    = len(camera_indices)
  n_params = n_cams * 6 + n_pts * 3
  A = lil_matrix((2 * n_obs, n_params), dtype=int)

  for i, (ci, pi) in enumerate(zip(camera_indices, point_indices)):
    row = 2 * i
    A[row:row+2, ci*6 : ci*6+6] = 1
    col = n_cams * 6 + pi * 3
    A[row:row+2, col : col+3] = 1

  return A


def run_bundle_adjustment(
  camera_params, 
  points_3d, 
  camera_indices, 
  point_indices, 
  observations, 
  K, 
  loss = "huber", 
  verbose=False
  ):
  
  n_cams = camera_params.shape[0]
  n_pts = points_3d.shape[0]

  x0 = np.concatenate([camera_params.ravel(), points_3d.ravel()])
  A  = build_sparsity(n_cams, n_pts, camera_indices, point_indices)

  result = least_squares(
    residuals,
    x0,
    jac_sparsity = A,
    method       = 'trf',           
    loss         = loss,
    verbose      = 2 if verbose else 0,
    args         = (n_cams, n_pts, camera_indices, point_indices,
                    observations, K),
  )

  cameras_opt = result.x[:n_cams * 6].reshape(n_cams, 6)
  points_opt  = result.x[n_cams * 6:].reshape(n_pts,  3)

  return cameras_opt, points_opt, result

def reprojection_error(
  camera_params, 
  points_3d, 
  camera_indices, 
  point_indices, 
  observations, 
  K
  ):
  res = residuals(np.concatenate([camera_params.ravel(), points_3d.ravel()]),
        camera_params.shape[0], points_3d.shape[0],
        camera_indices, point_indices, observations, K,)

  return float(np.mean(np.linalg.norm(res.reshape(-1, 2), axis=1)))


def pose_to_params(pose):
  rvec = R_to_rodrigues(pose[:3, :3])
  return np.concatenate([rvec, pose[:3, 3]])


def params_to_pose(params):
  pose = np.eye(4)
  pose[:3, :3] = rodrigues_to_R(params[:3])
  pose[:3, 3] = params[3:6]
  return pose

def collect_ba_data(map, window):
  frames = map.frames[-window:]
  if len(frames) < 2:
    return None

  frame_set = set(frames)
  points = [
    p for p in map.points
    if any(f in frame_set for f in p.frames)
  ]
  if len(points) < MIN_POINTS:
    return None

  frame_to_idx = {f: i for i, f in enumerate(frames)}
  camera_params = np.array([pose_to_params(f.pose) for f in frames])
  points_3d = np.array([p.pt[:3] / p.pt[3] for p in points])

  camera_indices = []
  point_indices = []
  observations = []
  for pi, p in enumerate(points):
    for f, uv in zip(p.frames, p.observations):
      if f not in frame_to_idx:
        continue
      camera_indices.append(frame_to_idx[f])
      point_indices.append(pi)
      observations.append(uv)

  if len(observations) < MIN_OBSERVATIONS:
    return None

  return {
    "frames": frames,
    "points": points,
    "camera_params": camera_params,
    "points_3d": points_3d,
    "camera_indices": np.array(camera_indices, dtype=int),
    "point_indices": np.array(point_indices, dtype=int),
    "observations": np.array(observations),
  }


def optimize_map(
  map, 
  K, 
  window=BA_WINDOW, 
  threshold=REPROJ_ERROR_THRESHOLD, 
  verbose=False
  ):
  data = collect_ba_data(map, window)
  if data is None:
    return

  frames = data["frames"]
  points = data["points"]
  camera_params = data["camera_params"]
  points_3d = data["points_3d"]
  camera_indices = data["camera_indices"]
  point_indices = data["point_indices"]
  observations = data["observations"]

  err_before = reprojection_error(
    camera_params, points_3d, camera_indices, point_indices, observations, K,
  )

  if err_before <= threshold:
    if verbose:
      print(f"[BA] skipped — reproj error {err_before:.2f} px <= {threshold:.2f} px threshold")
    return

  first_params = camera_params[0].copy()

  cameras_opt, points_opt, _ = run_bundle_adjustment(
    camera_params,
    points_3d,
    camera_indices,
    point_indices,
    observations,
    K,
    verbose=verbose,
  )

  cameras_opt[0] = first_params

  err_after = reprojection_error(
    cameras_opt, points_opt, camera_indices, point_indices, observations, K,
  )

  print(f"[BA] reproj error {err_before:.2f} -> {err_after:.2f} px  "
        f"({len(frames)} frames, {len(points)} points, {len(observations)} obs)")

  for i, f in enumerate(frames):
    f.pose = params_to_pose(cameras_opt[i])
  for i, p in enumerate(points):
    p.pt = np.append(points_opt[i], 1.0)