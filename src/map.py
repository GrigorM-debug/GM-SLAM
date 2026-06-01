import numpy as np
import rerun as rr
import rerun.blueprint as rrb

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

        rr.log(
          "world/map_points",
          rr.Points3D(
            pts_array,
            colors=[[255, 0, 0]],
            radii=self.POINT_RADIUS,
          ),
        )