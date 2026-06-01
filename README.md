# GM-SLAM

Monocular feature-based Visual SLAM in Python using Deep Leaning algorithms. A single camera stream is processed frame by frame: learned features are detected and matched, camera motion is estimated from two-view geometry, and 3D map points are triangulated and shown in a live 3D viewer.

## Overview

The pipeline runs on video input (file or default path under `videos/`):

1. **Feature extraction** — [ALIKED](https://github.com/cvg/LightGlue) keypoints via LightGlue.
2. **Feature matching** — consecutive frames matched with [LightGlue](https://github.com/cvg/LightGlue).
3. **Outliers filtering** - I used MAGSAC++ instead of the classic RANSAC 
4. **Pose estimation** — essential matrix and pose recovery with OpenCV (`findEssentialMat`, `recoverPose`).
5. **Mapping** — triangulation of inlier matches; scale from median scene depth on the second frame.
6. **Visualization** — 2D match view (Pygame) and 3D map / camera trajectory ([Rerun](https://rerun.io/)).


## Libraries

| Library | Role in this project |
| --- | --- |
| NumPy | Poses, homogeneous coordinates, triangulation math |
| OpenCV | Video I/O, essential matrix, pose recovery, triangulation, calibration |
| PyTorch | GPU/CPU backend for learned features and matching |
| LightGlue | ALIKED extractor and LightGlue matcher between frames |
| Rerun | Live 3D viewer for map points, camera poses, and trajectory |
| Pygame | 2D display window with match visualization and FPS |

## Requirements

- Python 3.10+ recommended
- CUDA-capable GPU optional (CPU fallback supported)
- A video file for input (see [Usage](#usage))

## Installation

```bash
git clone https://github.com/<your-username>/GM-SLAM.git
cd GM-SLAM

pip install -r requirements.txt
```

Now we have to install the deep learning algorithms. First go to the source directory.
```bash
cd src
```

Then run this commands:
```bash
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .

```

Deep learning algorithms uses Pytorch under the hood. After runing the command python -m pip install -e . it install pytorch but in my case it was CPU only version. I had uninstall the installed version using the command:
```bash
pip uninstall torch torchvision torchaudio -y  
```

Then you have to check what version of CUDA your GPU suports using the command:
```bash
nvidia-smi 
```

After you know what version of CUDA your GPU suppors you can install the Pytorch from the official site: https://pytorch.org/get-started/locally/

## Usage
In my case i had to start the Rerun Viewer using the command:
```bash
python -m rerun 
```

Then you start the slam using the command:
```bash
python slam.py "../videos/video4.mp4"
```

## Project structure

```
GM-SLAM/
├── src/
│   ├── slam.py              # Main loop
│   ├── process_frame.py     # Per-frame pipeline
│   ├── extractor.py         # ALIKED feature extraction
│   ├── matcher.py           # LightGlue matching + 2D overlay
│   ├── pose_estimation.py   # Essential matrix, pose, triangulation into map
│   ├── triangulate.py       # Two-view triangulation
│   ├── map.py               # Frames, points, Rerun 3D view
│   ├── display2d.py         # Pygame 2D viewer
│   ├── frame.py / point.py  # Map entities
│   ├── helpers.py           # CLI and video path resolution
│   └── camera/
│       └── calibrate-camera.py
├── videos/                  # Input videos (gitignored)
└── requirements.txt
```

## TODOS
- I don't why but in the Rerun viewer the map is a little bit vertical so i will try to fix it 
- Bundle adjustment back-end optimization
- Loop Closure detection
- Pose Graph Optimization after loop detection

## License

See [LICENSE](LICENSE).
