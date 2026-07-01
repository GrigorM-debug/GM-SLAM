import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

MAP_SAVE_VERSION = 2

class _PoseFrame:
  __slots__ = ("pose",)

  def __init__(self, pose):
    self.pose = pose

class _MapPoint:
  __slots__ = ("pt", "color", "semantic_label", "semantic_name", "semantic_color")

  def __init__(
    self,
    pt,
    color,
    semantic_label=None,
    semantic_name=None,
    semantic_color=None,
  ):
    self.pt = pt
    self.color = color
    self.semantic_label = semantic_label
    self.semantic_name = semantic_name
    self.semantic_color = semantic_color

class Map:
  POINT_RADIUS = 5
  TRAJECTORY_RADIUS = 2

  def __init__(self):
    self.frames = []
    self.points = []
    self.initialized = False
    self.view_width = 1280
    self.view_height = 720
    self.focal_length = 420.0
    self.depth_scale = 1.0
    self.scale_initialized = False

  def create_viewer(self, width, height, focal_length):
    self.view_width = width
    self.view_height = height
    self.focal_length = focal_length
    rr.init("GM_SLAM Map")
    blueprint = rrb.Blueprint(rrb.Spatial3DView(name="Map", origin="/world"), collapse_panels=True)
    rr.send_blueprint(blueprint)
    self.initialized = True
    rr.spawn()

  def display(self):
    if not self.initialized:
      return

    poses = [f.pose for f in self.frames]

    if poses:
      camera_centers = np.array([np.linalg.inv(p)[:3, 3] for p in poses])

      for i, pose in enumerate(poses):
        Twc = np.linalg.inv(pose)
        rr.log(
          f"world/cameras/cam_{i}",
          rr.Transform3D(translation=Twc[:3, 3], mat3x3=Twc[:3, :3], from_parent=False),
        )
        rr.log(
          f"world/cameras/cam_{i}/pinhole",
          rr.Pinhole(
            focal_length=self.focal_length,
            width=self.view_width,
            height=self.view_height,
            camera_xyz=rr.ViewCoordinates.RDF,
          ),
        )

      if len(camera_centers) >= 2:
        rr.log(
          "world/trajectory",
          rr.LineStrips3D(
            strips=[camera_centers],
            colors=[0, 255, 0],
            radii=self.TRAJECTORY_RADIUS,
          ),
        )

      pts = [p.pt for p in self.points]
      if pts:
        pts_array = np.array(pts)
        if pts_array.ndim == 2 and pts_array.shape[1] == 4:
          pts_array = pts_array[:, :3] / pts_array[:, 3:4]

        colors = np.array([p.color for p in self.points], dtype=np.uint8)

        rr.log(
          "world/map_points",
          rr.Points3D(
            pts_array,
            colors=colors,
            radii=self.POINT_RADIUS,
          ),
        )

        semantic_points = [
          p for p in self.points
          if getattr(p, "semantic_color", None) is not None
        ]
        if semantic_points:
          semantic_pts = np.array([p.pt for p in semantic_points])
          if semantic_pts.ndim == 2 and semantic_pts.shape[1] == 4:
            semantic_pts = semantic_pts[:, :3] / semantic_pts[:, 3:4]

          semantic_colors = np.array(
            [p.semantic_color for p in semantic_points],
            dtype=np.uint8,
          )

          rr.log(
            "world/semantic_points",
            rr.Points3D(
              semantic_pts,
              colors=semantic_colors,
              radii=self.POINT_RADIUS * 1.5,
            ),
          )

  def save(self, path, K):
    path = Path(path)
    if path.suffix != ".npz":
      path = path.with_suffix(".npz")

    if not self.frames:
      print("[WARN] Map is empty — nothing to save.")
      return False

    path.parent.mkdir(parents=True, exist_ok=True)

    poses = np.array([f.pose for f in self.frames], dtype=np.float64)
    camera_centers = np.array(
      [np.linalg.inv(p)[:3, 3] for p in poses], dtype=np.float64
    )

    if self.points:
      points = np.array([p.pt for p in self.points], dtype=np.float64)
      colors = np.array([p.color for p in self.points], dtype=np.uint8)
    else:
      points = np.zeros((0, 4), dtype=np.float64)
      colors = np.zeros((0, 3), dtype=np.uint8)

    frame_to_idx = {f: i for i, f in enumerate(self.frames)}
    obs_frame_idx = []
    obs_point_idx = []
    obs_uv = []
    for pi, point in enumerate(self.points):
      for frame, uv in zip(point.frames, point.observations):
        if frame not in frame_to_idx:
          continue
        obs_frame_idx.append(frame_to_idx[frame])
        obs_point_idx.append(pi)
        obs_uv.append(uv)

    np.savez_compressed(
      path,
      version=np.array([MAP_SAVE_VERSION], dtype=np.int32),
      K=np.asarray(K, dtype=np.float64),
      depth_scale=np.array([self.depth_scale], dtype=np.float64),
      scale_initialized=np.array([int(self.scale_initialized)], dtype=np.int8),
      view_width=np.array([self.view_width], dtype=np.int32),
      view_height=np.array([self.view_height], dtype=np.int32),
      focal_length=np.array([self.focal_length], dtype=np.float64),
      poses=poses,
      camera_centers=camera_centers,
      points=points,
      colors=colors,
      obs_frame_idx=np.array(obs_frame_idx, dtype=np.int32),
      obs_point_idx=np.array(obs_point_idx, dtype=np.int32),
      obs_uv=np.array(obs_uv, dtype=np.float64)
      if obs_uv
      else np.zeros((0, 2), dtype=np.float64),
    )

    print(f"[INFO] Map saved to {path}")
    print(
      f"       {len(self.frames)} frames, {len(self.points)} points, "
      f"{len(obs_frame_idx)} observations"
    )
    return True

  @classmethod
  def load(cls, path):
    path = Path(path)
    if path.suffix != ".npz":
      path = path.with_suffix(".npz")

    if not path.exists():
      raise FileNotFoundError(f"Map file not found: {path}")

    data = np.load(path)
    version = int(data["version"][0])
    if version != MAP_SAVE_VERSION:
      print(
        f"[WARN] Map file version {version} differs from current "
        f"version {MAP_SAVE_VERSION}; load may be incomplete."
      )

    map_obj = cls()
    map_obj.depth_scale = float(data["depth_scale"][0])
    map_obj.scale_initialized = bool(data["scale_initialized"][0])
    map_obj.view_width = int(data["view_width"][0])
    map_obj.view_height = int(data["view_height"][0])
    map_obj.focal_length = float(data["focal_length"][0])
    map_obj.frames = [_PoseFrame(pose) for pose in data["poses"]]

    default_color = np.array([255, 0, 0], dtype=np.uint8)
    if "colors" in data:
      point_colors = data["colors"]
    else:
      point_colors = np.tile(default_color, (len(data["points"]), 1))

    map_obj.points = [
      _MapPoint(pt, color)
      for pt, color in zip(data["points"], point_colors)
    ]

    print(f"[INFO] Map loaded from {path}")
    print(
      f"       {len(map_obj.frames)} frames, {len(map_obj.points)} points"
    )
    return map_obj