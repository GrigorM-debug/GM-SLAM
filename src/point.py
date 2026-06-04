import numpy as np

# A 3D point in the map
class Point:
  def __init__(self, map, loc):
    self.frames = []
    self.pt = loc
    self.observations = []
    self.id = len(map.points)

  def add_observation(self, frame, uv):
    self.frames.append(frame)
    self.observations.append(np.asarray(uv, dtype=np.float64))

