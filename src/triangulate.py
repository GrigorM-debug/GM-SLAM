import cv2
import numpy as np

def triangulate(K, pose1, pose2, pts1, pts2):
  if len(pts1) == 0:
    return np.zeros((0, 4))

  P1 = K @ pose1[:3, :]
  P2 = K @ pose2[:3, :]

  pts1 = np.asarray(pts1, dtype=np.float64)
  pts2 = np.asarray(pts2, dtype=np.float64)

  pts4d = cv2.triangulatePoints(
    P1,
    P2,
    pts1.T.reshape(2, -1),
    pts2.T.reshape(2, -1),
  )

  return pts4d.T
