import cv2
from lightglue import LightGlue
from lightglue.utils import rbd

class FeatureMatcher:
  def __init__(self, device):
    self.device = device
    # Default params
    self.matcher = LightGlue(features="aliked").eval().to(device)
    # Accuracy params
    # self.matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
    # Speed params
    # self.matcher = LightGlue(features="aliked", depth_confidence=0.9, width_confidence=0.95).eval().to(self.device)
    self.prev_feats = None
    self.score_threshold = 0.2
    self.keypoint_radius = 5
    self.keypoint_thickness = 3
    self.match_line_thickness = 4
    self.pts_prev = None
    self.pts_curr = None

  def match_and_draw(self, img, feats):
    kpts = feats["keypoints"][0]

    if self.prev_feats is None:
      kp_np = kpts.cpu().numpy()
      for p in kp_np:
        cv2.circle(
          img,
          (int(p[0]), int(p[1])),
          self.keypoint_radius,
          (0, 255, 0),
          self.keypoint_thickness,
        )
      self.prev_feats = feats
      return img, 0, int(kpts.shape[0]), None, None

    matches01 = self.matcher({"image0": self.prev_feats, "image1": feats})
    matches01 = rbd(matches01)

    matches = matches01["matches"]
    scores = matches01["scores"]

    if scores.numel() > 0:
      keep = scores > self.score_threshold
      matches = matches[keep]

    prev_kpts = self.prev_feats["keypoints"][0].cpu().numpy()
    curr_kpts = kpts.cpu().numpy()

    matches_np = matches.cpu().numpy() if matches.numel() > 0 else []
    n_good = 0

    for m in matches_np:
      p_prev = prev_kpts[int(m[0])]
      p_curr = curr_kpts[int(m[1])]
      x1, y1 = int(p_prev[0]), int(p_prev[1])
      x2, y2 = int(p_curr[0]), int(p_curr[1])

      cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        self.match_line_thickness,
      )
      cv2.circle(
        img,
        (x2, y2),
        self.keypoint_radius,
        (0, 255, 0),
        self.keypoint_thickness,
      )
      n_good += 1

    if len(matches_np) > 0:
      self.pts_prev = prev_kpts[matches_np[:,0]]
      self.pts_curr = curr_kpts[matches_np[:,1]]
    else:
      self.pts_prev = None
      self.pts_curr = None

    self.prev_feats = feats

    return img, int(matches.shape[0]), int(kpts.shape[0]), self.pts_prev, self.pts_curr