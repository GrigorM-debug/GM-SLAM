import pygame
import cv2

class Display2D:
  def __init__(self, W, H):
    self.W = W
    self.H = H
    pygame.init()
    self.window = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GM-SLAM")
    self.clock = pygame.time.Clock()
    self.font = pygame.font.SysFont("Arial", 30, bold=True)

  def paint(self, img):
    self.clock.tick(60)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        exit(0)
    img = cv2.resize(img, (self.W, self.H))

    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pygame.surfarray.blit_array(self.window, img.swapaxes(0, 1))
    fps = self.clock.get_fps()
    text_surface = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
    self.window.blit(text_surface, (10, 10))
    pygame.display.flip()