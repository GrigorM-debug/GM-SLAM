class Frame:
  def __init__(self, map, img, K, pose):
    self.id = len(map.frames)
    self.img = img
    self.K = K
    self.pose = pose