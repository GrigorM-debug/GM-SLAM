import numpy as np

# A 3D point in the map
class Point:
  def __init__(
    self,
    map,
    loc,
    color,
    semantic_label=None,
    semantic_name=None,
    semantic_color=None,
  ):
    self.frames = []
    self.pt = loc
    self.observations = []
    self.id = len(map.points)
    self.color = color
    self.semantic_label = semantic_label
    self.semantic_name = semantic_name
    self.semantic_color = semantic_color

  def add_observation(self, frame, uv):
    self.frames.append(frame)
    self.observations.append(np.asarray(uv, dtype=np.float64))


