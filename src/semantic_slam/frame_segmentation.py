import cv2
import numpy as np
import torch
import torchvision.models.segmentation as seg_models

class FrameSegmenter:
  TARGET_CLASS_NAMES = {
    "person", 
    "bicycle", 
    "car", 
    "motorbike", 
    "bus"
  }
  HIGHLIGHT_RGB_COLORS = {
    "person": (255, 105, 180),
    "bicycle": (180, 0, 255),
    "car": (255, 230, 0),
    "motorbike": (0, 255, 255),
    "bus": (255, 140, 0),
  }

  def __init__(self, device, alpha=0.45):
    self.device = device
    self.alpha = alpha
    self.weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT
    self.categories = self.weights.meta["categories"]
    self.preprocess = self.weights.transforms()
    self.model = seg_models.deeplabv3_resnet50(weights=self.weights)
    self.model.eval().to(self.device)
    self.palette = self.build_high_contrast_palette()
    self.palette_bgr = self.palette[:, ::-1].copy()
    self.target_labels = {
      label_id
      for label_id, class_name in enumerate(self.categories)
      if class_name in self.TARGET_CLASS_NAMES
    }

  def segment_frame(self, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = self.preprocess(frame_tensor).unsqueeze(0).to(self.device)

    with torch.inference_mode():
      logits = self.model(frame_tensor)["out"][0]

    segmentation_map = logits.argmax(0).detach().cpu().numpy().astype(np.uint8)
    segmentation_map = self.resize_segmentation_map(frame, segmentation_map)
    return self.render_segmentation(frame, segmentation_map), segmentation_map

  def resize_segmentation_map(self, frame, segmentation_map):
    frame_h, frame_w = frame.shape[:2]
    if segmentation_map.shape != (frame_h, frame_w):
      segmentation_map = cv2.resize(
        segmentation_map,
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
      )
    return segmentation_map

  def render_segmentation(self, frame, segmentation_map):
    color_mask = self.palette_bgr[segmentation_map]
    blended = cv2.addWeighted(frame, 1.0 - self.alpha, color_mask, self.alpha, 0.0)

    detected_labels = np.unique(segmentation_map)
    detected_labels = [label for label in detected_labels if label != 0]
    self.draw_legend(blended, detected_labels)
    return blended

  def sample_target_class(self, segmentation_map, uv):
    x, y = int(round(uv[0])), int(round(uv[1]))
    x = int(np.clip(x, 0, segmentation_map.shape[1] - 1))
    y = int(np.clip(y, 0, segmentation_map.shape[0] - 1))
    label_id = int(segmentation_map[y, x])

    if label_id not in self.target_labels:
      return None

    return {
      "label": label_id,
      "name": self.categories[label_id],
      "color": self.palette[label_id].copy(),
    }

  def draw_legend(self, frame, label_ids):
    y = 20
    for label_id in label_ids[:8]:
      color = tuple(int(c) for c in self.palette_bgr[label_id])
      class_name = self.categories[label_id]
      cv2.rectangle(frame, (15, y), (35, y + 20), color, -1)
      cv2.putText(
        frame,
        class_name,
        (45, y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
      )
      y += 28

  @staticmethod
  def build_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for label_id in range(num_classes):
      palette[label_id] = (
        int((37 * label_id) % 255),
        int((17 * label_id + 80) % 255),
        int((29 * label_id + 160) % 255),
      )
    palette[0] = (0, 0, 0)
    return palette

  def build_high_contrast_palette(self):
    palette = self.build_palette(len(self.categories))
    for label_id, class_name in enumerate(self.categories):
      if class_name in self.HIGHLIGHT_RGB_COLORS:
        palette[label_id] = self.HIGHLIGHT_RGB_COLORS[class_name]
    return palette
