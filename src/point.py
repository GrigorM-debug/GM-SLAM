
# A 3D point in the map
class Point:
  def __init__(self, map, loc):
    self.frames = []
    self.pt = loc
    self.idxs = []
    self.id = len(map.points)

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxs.append(idx)

