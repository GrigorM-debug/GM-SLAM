import torch
from lightglue import ALIKED

class FeatureExtractor:
  def __init__(self, device):
    self.device = device
    # Default params
    self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
    # Accuracy params
    # self.extractor = ALIKED(max_num_keypoints=4000).eval().to(self.device)
    # Speed params
    # self.extractor = ALIKED(max_num_keypoints=1024).eval().to(self.device)
          
  def extract(self, img):
    img_t = torch.from_numpy(img).float().to(self.device) / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    feats = self.extractor.extract(img_t)
    feats = self.extractor.extract(img_t)
    return feats