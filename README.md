# ARIS Python Reader

Python implementation of an ARIS/DDF sonar reader and fish tracking workflow.
The project is designed as a MATLAB-style pipeline for:

- opening `.aris` / `.ddf` files,
- converting beams to fan/cartesian images,
- running fish detection + tracking,
- exporting analysis CSVs for offline review.

This codebase uses **classical computer vision and tracking** (thresholding, morphology, connected components, Kalman filtering, Hungarian assignment), not deep learning models.

## 1) What This Project Does

Given ARIS or DDF sonar files, the pipeline:

1. reads frame and header data,
2. maps raw sonar beams to fan/cartesian image space,
3. optionally subtracts a static background pattern,
4. detects fish-like blobs,
5. tracks fish identities across frames,
6. computes per-frame and per-fish metrics,
7. exports MATLAB-style CSV reports under `exports/statistics/`.

## 2) Repository Entry Points

- `aris_reader.py`
  - Main interactive menu (recommended for day-to-day use).
  - Supports playback, fish tracking, exports, static pattern generation, visualization.
- `aris_python_api.py`
  - Scriptable CLI/API interface (`play`, `export-avi`, `export-images`, `batch`).
- `create_mangrove_mask.py`
  - Tool to build habitat/mangrove masks (interactive polygon mode or pattern-threshold mode).
- `simple_visualizer.py`
  - Post-processing visualizer for exported tracking CSV files.

## 3) Installation

### Requirements

- Python 3.7+
- See `requirements.txt`:
  - `numpy`
  - `opencv-python`
  - `scipy`
  - `matplotlib`

### Install

```bash
pip install -r requirements.txt
```

Optional (only needed for some time-index helper paths):

```bash
pip install pandas
```

## 4) Quick Start (Interactive Workflow)

Run:

```bash
python aris_reader.py
```

If a file is not auto-detected, pass one directly:

```bash
python aris_reader.py "/full/path/to/file.aris"
```

Recommended workflow:

1. Press `P` once to compute and save a static pattern (best detection stability).
2. Select `1` to play with fish tracking enabled.
3. Press `SPACE` to play/pause.
4. Press `ESC` to stop; CSV statistics auto-export on completion.
5. Use `V` to open the visualization workflow for exported tracking data.

## 5) Playback Controls

During the OpenCV viewer:

- `SPACE` - play/pause
- `ESC` - exit and auto-export CSVs
- `R` - reset to first frame
- `D` - toggle fish detection on/off
- `C` - clear detector state
- `E` - export statistics immediately
- `T` - toggle saved tracks overlay

## 6) How the Fish Tracking Pipeline Works

### Frame Processing

1. Raw sonar frame is read from ARIS/DDF.
2. Frame is mapped to fan/cartesian image space.
3. Optional static pattern subtraction is applied.
4. Binary thresholding creates a foreground mask.
5. Morphological opening/closing cleans noise.
6. Connected components produce candidate blobs.
7. Area/aspect-ratio filters keep plausible fish detections.

### Multi-Object Tracking

For each frame:

1. Existing tracks predict next position (Kalman or linear fallback).
2. Detections are assigned to tracks via Hungarian matching.
3. Kalman correction updates assigned tracks.
4. Unmatched tracks age/invisibility counters increase.
5. Old/low-quality tracks are removed.
6. Unmatched detections spawn new tracks.
7. Reliability gating filters out short/noisy/stationary artifacts.

### Export

When playback completes (or manual export is triggered), the system writes:

- `*_global_info.csv`
- `*_tracking_info.csv`
- `*_validation_info_ranges.csv`
- `*_validation_info_max_fishes_per_range.csv`
- `*_summary.csv`

to `exports/statistics/`.

## 7) Configuration System

Per-video configuration is loaded automatically from:

`config/config_<video_basename>.json`

If missing, the project can generate a template config automatically.

Common tunable groups in `fish_tracking_config.py`:

- intensity thresholds
- fish size / shape filters
- morphology kernel sizes
- track confirmation/invisibility limits
- Kalman process/measurement noise
- ROI bounds
- static pattern parameters
- mangrove/validation parameters

## 8) Static Pattern Workflow

Static pattern subtraction suppresses stationary background clutter.

### Generate pattern from a video

- In `aris_reader.py`, choose option `P`.
- Pattern files are saved under `exports/patterns/` (`.npy` and optionally `.mat`).

### Enable in config

Set these keys in your video config JSON:

```json
{
  "USE_STATIC_PATTERN": true,
  "STATIC_PATTERN_FILE": "exports/patterns/<video>_pattern.npy",
  "BW_THRESHOLD": 0.12,
  "BW_THRESHOLD_MODE": "relative"
}
```

## 9) Mangrove / Habitat Masks

Create mask interactively:

```bash
python create_mangrove_mask.py "/path/to/file.aris" --frame 50 --output mangrove_mask.png
```

Or create from an existing static pattern:

```bash
python create_mangrove_mask.py --pattern exports/patterns/<video>_pattern.npy --threshold 0.6 --output mangrove_mask.png
```

Then set in config:

```json
{
  "MANGROVE_MASK_FILE": "mangrove_mask.png"
}
```

## 10) Scriptable CLI Examples

Play a frame range:

```bash
python aris_python_api.py play file.aris --frames 1 500
```

Export AVI:

```bash
python aris_python_api.py export-avi file.aris --output file_export.avi --type cartesian --fps 30
```

Export images:

```bash
python aris_python_api.py export-images file.aris --output out_images --frames 1 100
```

Batch process directory:

```bash
python aris_python_api.py batch ./input_dir --output ./results --limit 50
```

## 11) Expected Outputs

- `exports/statistics/` - analysis CSV outputs per processed file
- `exports/patterns/` - static pattern files
- `<video>_export.avi` - optional video export
- `<video>_first_frame/` - first-frame image exports
- custom output folders from batch/image export commands

## 12) Troubleshooting

- **No files found**: provide explicit path to `.aris`/`.ddf` when launching `aris_reader.py`.
- **Low detections**: compute static pattern (`P`) and reduce threshold values in config.
- **Too many false positives**: increase `MIN_FISH_AREA`, tighten aspect ratio, or increase morphology kernels.
- **Unexpected track drops**: tune `MAX_INVISIBLE_COUNT`, `MIN_TRACK_HITS`, and Kalman noise params.
- **Color map warning (`bluebar.mat`)**: code falls back if unavailable, but keeping `bluebar.mat` in repo root preserves expected visualization style.

## 13) Contact

Email: brlozano@fiu.edu