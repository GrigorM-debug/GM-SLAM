import argparse
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
  return parser.parse_args()


def resolve_video_path(video_arg):
  script_dir = Path(__file__).resolve().parent
  default_video = script_dir.parent / "videos" / "video.mp4"

  if video_arg is None:
    return default_video

  p = Path(video_arg)
  if p.is_absolute():
    return p
  return (Path.cwd() / p).resolve()