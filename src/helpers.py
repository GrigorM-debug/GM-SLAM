import argparse
from datetime import datetime
from pathlib import Path

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "video",
    type=str,
    nargs="?",
    default=None,
    help="Path to the video (optional)",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Path to save the map (.npz). Defaults to maps/<video_name>_<timestamp>.npz",
  )
  parser.add_argument(
    "--semantic",
    action="store_true",
    help="Enable semantic segmentation. Default is disabled (False).",
  )
  return parser.parse_args()


def resolve_video_path(video_arg):
  script_dir = Path(__file__).resolve().parent
  default_video = script_dir.parent / "videos" / "video4.mp4"

  if video_arg is None:
    return default_video

  p = Path(video_arg)
  if p.is_absolute():
    return p
  return (Path.cwd() / p).resolve()


def resolve_output_path(output_arg, video_path):
  if output_arg is not None:
    p = Path(output_arg)
    if not p.is_absolute():
      p = (Path.cwd() / p).resolve()
    return p

  project_root = Path(__file__).resolve().parent.parent
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  return project_root / "maps" / f"{video_path.stem}_{timestamp}.npz"