import argparse
import time
from pathlib import Path
from map import Map

def parse_args():
  parser = argparse.ArgumentParser(
    description="Load a saved map and view it in Rerun.",
  )
  parser.add_argument(
    "map_path",
    type=str,
    help="Path to a saved map (.npz)",
  )
  return parser.parse_args()


def resolve_map_path(map_arg):
  p = Path(map_arg)
  if p.is_absolute():
    return p
  return (Path.cwd() / p).resolve()

def main():
  args = parse_args()
  map_path = resolve_map_path(args.map_path)

  try:
    map_obj = Map.load(map_path)
  except FileNotFoundError as exc:
    print(f"[ERROR] {exc}")
    return

  map_obj.create_viewer(
    map_obj.view_width,
    map_obj.view_height,
    map_obj.focal_length,
  )
  map_obj.display()

  print("[INFO] Map displayed in Rerun. Press Ctrl+C to exit.")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  main()
