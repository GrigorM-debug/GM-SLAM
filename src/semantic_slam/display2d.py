import cv2
class SemanticDisplay2D:
  def __init__(self, width, height, title="GM-SLAM Semantic View"):
    self.width = width
    self.height = height
    self.title = title
    cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.title, self.width, self.height)

  def paint(self, img):
    frame = cv2.resize(img, (self.width, self.height))
    cv2.imshow(self.title, frame)
