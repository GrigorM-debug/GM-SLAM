import numpy as np
from multiprocessing import Process, Queue

class Map:
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = None

  def create_viewver(self):
    self.q = Queue()
    p = Process(target=self.viewer_thread, args=(self.q,))
    p.daemon = True
    p.start()
  
  def viewer_thread(self, q):
    self.viewer_init(1280, 720)
    while True:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    pass

  def viewer_refresh(self, q):
    pass

  def display(self):
    pass
    # if self.q is None:
    #   return 
      
    # poses, pts = [], []

    # for f in self.frames:
    #   poses.append(f.pose)

    # for p in self.points:
    #   pts.append(p.pt)
      
    # self.q.put((np.array(poses), np.array(pts)))
