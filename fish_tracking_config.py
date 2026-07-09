#!/usr/bin/env python3
"""
Unified Fish Tracking Configuration
Single configuration file for all fish detection and tracking parameters
Per-video configuration system for optimal results
"""

import cv2
import os
import json
import re

CONFIG_ROOT_DIR = "config"


class FishTrackingConfig:
    """
    Unified configuration for fish detection and tracking
    All parameters in one place, tunable per video file
    """
    
    def __init__(self, config_file_path=None):
        """
        Initialize configuration
        
        Args:
            config_file_path: Path to JSON config file for this specific video
        """
        # === DEFAULT PARAMETERS (Base Settings) ===
        self.load_defaults()
        
        # Load video-specific overrides if provided
        if config_file_path and os.path.exists(config_file_path):
            self.load_from_file(config_file_path)
            print(f"Loaded config from: {config_file_path}")
        else:
            print("Using default configuration")
    
    def load_defaults(self):
        """Load default parameters - tuned to match working _926 configuration"""

        # === INTENSITY DETECTION ===
        self.INTENSITY_THRESHOLD = 40       # Threshold for detection (low works with static pattern ON)
        self.MIN_BBOX_WIDTH = 5             # Minimum bounding box width
        self.MIN_BBOX_HEIGHT = 5            # Minimum bounding box height

        # === DETECTION CRITERIA ===
        self.MIN_FISH_AREA = 6              # Minimum fish area (pixels)
        self.MAX_FISH_AREA = 2500           # Maximum fish area (pixels)
        self.MIN_ASPECT_RATIO = 0.2         # Min width/height ratio (fish are elongated)
        self.MAX_ASPECT_RATIO = 5.0         # Max width/height ratio

        # === MORPHOLOGICAL FILTERING ===
        self.MORPH_OPENING_SIZE = 1         # Remove noise (erosion + dilation)
        self.MORPH_CLOSING_SIZE = 2         # Fill holes (dilation + erosion)

        # === TRACKING PARAMETERS (MATLAB-equivalent) ===
        self.MIN_TRACK_HITS = 2             # Frames needed to confirm track (MATLAB: minVisibleCount)
        self.MAX_INVISIBLE_COUNT = 6        # Max frames without detection (MATLAB: invisibleForTooLong)
        self.AGE_THRESHOLD = 1              # Min age for visibility ratio check (MATLAB: ageThreshold)
        self.MIN_VISIBILITY_RATIO = 0.0     # Min visibility ratio for young tracks (0.0 = off)
        self.MAX_ASSIGNMENT_COST = 65.0     # Max distance for track assignment
        self.MAX_TOTAL_TRACKS = 200         # Max concurrent tracks
        self.MIN_TRACK_DISPLACEMENT = 8.0   # Min total displacement (px) to be a real fish
        self.MIN_DISPLACEMENT_AGE = 10      # Frames before displacement check applies

        # === KALMAN FILTER (MATLAB-equivalent) ===
        self.USE_KALMAN = True              # Enable Kalman filter tracking (vs simple linear)
        self.KF_INITIAL_ERROR = 10.0        # Initial estimate error (MATLAB: KFInitialEstimateError)
        self.KF_PROCESS_NOISE = 5.0         # Motion noise (MATLAB: KFMotionNoise)
        self.KF_MEASUREMENT_NOISE = 5.0     # Measurement noise (MATLAB: KFMeasurementNoise)
        self.COST_OF_NON_ASSIGNMENT = 10.0  # Cost threshold for assignment (MATLAB: costOfNonAssignment)
        self.KF_GATING_THRESHOLD_D2 = 9.21  # Chi-square 2 DOF @ 99% for Mahalanobis gating

        # === ROI (REGION OF INTEREST) ===
        self.ENABLE_ROI = True              # Use detection region
        self.ROI_X_START = 0.05             # Left boundary (0.0-1.0)
        self.ROI_X_END = 0.95               # Right boundary
        self.ROI_Y_START = 0.15             # Top boundary (skip noisy top area)
        self.ROI_Y_END = 0.95               # Bottom boundary

        # === STATIC PATTERN (MATLAB parity) ===
        self.USE_STATIC_PATTERN = False     # Enable static pattern removal (set true after Option P)
        self.STATIC_PATTERN_FILE = ""       # Path to pattern file (.npy or .mat)
        self.STATIC_PATTERN_FRAMES = 200    # Frames to use for pattern computation
        self.BW_THRESHOLD = 0.15            # Binary threshold for pattern difference
        self.BW_THRESHOLD_MODE = "relative" # "relative" (0-1) or "absolute" (0-255)
        self.PATTERN_DETECTION_MODE = "subtract_positive"  # MATLAB parity: threshold max(frame-pattern, 0); alternative: "abs_diff"
        self.FAN_IMAGE_X_SIZE = 400         # Fan image width in pixels (MATLAB tutorial commonly uses 500)
        self.FAN_HALF_ANGLE_DEG = 14.0      # Half field-of-view used by fan mapping (MATLAB legacy mapscan uses 14.4)
        self.FAN_BEAM_SMOOTH = 4            # Beam interpolation factor (MATLAB smooth: 1, 4, or 8)

        # === MANGROVE MASK (OPTIONAL - MATLAB parity) ===
        self.MANGROVE_MASK_FILE = ""        # Path to binary PNG mask (optional)
        self.VALIDATION_MASK_FILE = ""      # Path to validation area mask (optional)
        self.PIX_SCALE = 0.001              # Meters per pixel (approximate)
        self.PREY_PREDATOR_THRESHOLD_CM = 5.0   # Fish length < threshold => prey
        self.SCHOOL_MIN_FISHES = 10             # Min fish count to consider schooling
        self.VALIDATION_MAX_FISH_PERIOD_S = 300 # Validation aggregation window (seconds)

        # === DISPLAY SETTINGS ===
        self.BBOX_COLOR = (0, 255, 255)     # Fish box color (BGR: Yellow)
        self.BBOX_THICKNESS = 1             # Box line thickness
        self.SHOW_FISH_IDS = True           # Show fish ID numbers
        self.LABEL_COLOR = (255, 255, 0)    # Label text color (BGR: Cyan)
        self.LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.LABEL_FONT_SCALE = 0.4         # Label text size
        self.LABEL_THICKNESS = 1            # Label text thickness

        # === DEBUG OPTIONS ===
        self.SHOW_DEBUG_WINDOWS = False     # Show processing steps
        self.SHOW_DETECTION_STATS = True    # Print frame statistics
        self.STATS_UPDATE_INTERVAL = 30     # Stats every N frames
    
    def load_from_file(self, config_file_path):
        """Load configuration overrides from JSON file"""
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)

            for key, value in config_data.items():
                if key.startswith("_"):
                    continue
                if hasattr(self, key):
                    setattr(self, key, value)
                    print(f"   Override: {key} = {value}")
                else:
                    print(f"   Warning: Unknown parameter '{key}' ignored")

        except Exception as e:
            print(f"   Error loading config file: {e}")
            print("   Using default parameters")
    
    def save_to_file(self, config_file_path):
        """Save current configuration to JSON file"""
        config_data = {}
        
        # Get all configuration attributes
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                # Skip non-serializable types
                if isinstance(attr_value, (int, float, str, bool, list, tuple)):
                    config_data[attr_name] = attr_value
        
        try:
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Configuration saved to: {config_file_path}")
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def create_video_config_template(self, video_filename):
        """Create a template config file for a specific video"""
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        config_filename = f"config_{base_name}.json"
        config_path = _get_template_target_path(video_filename, config_filename)

        template = _build_config_template(base_name)

        try:
            template_dir = os.path.dirname(config_path)
            if template_dir:
                os.makedirs(template_dir, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(template, f, indent=4)
            print(f"Created config template: {config_path}")
            print(f"   Edit this file to tune parameters for your video")
            return config_path
        except Exception as e:
            print(f" Failed to create template: {e}")
            return None
    
    def get_config_dict(self):
        """Get configuration as dictionary for compatibility"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        return config_dict

    def print_summary(self):
        """Print current configuration summary"""
        print("\nCurrent Fish Tracking Configuration:")
        print("=" * 50)
        print(f"Intensity Threshold: {self.INTENSITY_THRESHOLD}")
        print(f"Fish Size Range: {self.MIN_FISH_AREA} - {self.MAX_FISH_AREA} pixels")
        print(f"Track Confirmation: {self.MIN_TRACK_HITS} frames")
        print(f"ROI Region: Y {self.ROI_Y_START:.1%} - {self.ROI_Y_END:.1%}")
        print(f"Max Concurrent Tracks: {self.MAX_TOTAL_TRACKS}")
        print(
            f"Fan Geometry: width={self.FAN_IMAGE_X_SIZE}, "
            f"half_angle={self.FAN_HALF_ANGLE_DEG:.2f} deg, smooth={self.FAN_BEAM_SMOOTH}"
        )
        print("=" * 50)


def _build_config_template(base_name):
    """
    Build a complete config template dict for a given video base name.
    Single source of truth — used by both create_video_config_template()
    and get_config_for_video() so the two never diverge again.
    """
    pattern_path = f"exports/patterns/{base_name}_pattern.npy"
    return {
        "_comment": f"Fish tracking configuration for {base_name}",
        "_description": "Adjust these parameters for optimal fish detection",

        "INTENSITY_THRESHOLD": 40,
        "_INTENSITY_THRESHOLD_help": "Pixel brightness threshold (MATLAB BwThres equivalent, 0-255). Low value works because static pattern removes background first.",

        "MIN_FISH_AREA": 6,
        "_MIN_FISH_AREA_help": "Minimum fish blob size in pixels (try 4-15)",
        "MAX_FISH_AREA": 2500,
        "_MAX_FISH_AREA_help": "Maximum fish blob size in pixels (try 1000-3000)",
        "MIN_ASPECT_RATIO": 0.2,
        "_MIN_ASPECT_RATIO_help": "Minimum bounding box width/height ratio (rejects thin line artifacts)",
        "MAX_ASPECT_RATIO": 5.0,
        "_MAX_ASPECT_RATIO_help": "Maximum bounding box width/height ratio (rejects very elongated shapes)",

        "MORPH_OPENING_SIZE": 1,
        "_MORPH_OPENING_SIZE_help": "Erosion+dilation kernel to remove speckle noise (0=off, 1-3 typical)",
        "MORPH_CLOSING_SIZE": 2,
        "_MORPH_CLOSING_SIZE_help": "Dilation+erosion kernel to fill holes in blobs (0=off, 2-4 typical)",

        "MIN_TRACK_HITS": 2,
        "_MIN_TRACK_HITS_help": "Frames needed to confirm a track as real (1=fast, 3=stable)",
        "MAX_INVISIBLE_COUNT": 6,
        "_MAX_INVISIBLE_COUNT_help": "Frames a track survives without a detection before deletion (3-10)",
        "AGE_THRESHOLD": 1,
        "_AGE_THRESHOLD_help": "Minimum track age before visibility ratio check applies (1=lenient, 5=strict)",
        "MIN_VISIBILITY_RATIO": 0.0,
        "_MIN_VISIBILITY_RATIO_help": "Minimum fraction of frames a young track must be visible (0.0=off, 0.5-0.7=strict)",
        "MAX_TOTAL_TRACKS": 200,
        "_MAX_TOTAL_TRACKS_help": "Hard cap on simultaneous active tracks",
        "MAX_ASSIGNMENT_COST": 65.0,
        "_MAX_ASSIGNMENT_COST_help": "Max pixel distance for track-to-detection matching (30-80)",
        "MIN_TRACK_DISPLACEMENT": 8.0,
        "_MIN_TRACK_DISPLACEMENT_help": "Min total pixel movement for a track to count as real fish (0=disabled, 5-10 typical)",
        "MIN_DISPLACEMENT_AGE": 10,
        "_MIN_DISPLACEMENT_AGE_help": "Frames before displacement filter activates (5-15 typical)",

        "USE_KALMAN": True,
        "_USE_KALMAN_help": "Enable Kalman filter prediction and Mahalanobis assignment (recommended true)",
        "KF_PROCESS_NOISE": 5.0,
        "_KF_PROCESS_NOISE_help": "How unpredictable fish motion is between frames (higher = more erratic allowed)",
        "KF_MEASUREMENT_NOISE": 5.0,
        "_KF_MEASUREMENT_NOISE_help": "Expected noise in centroid detections (higher = trust prediction more)",
        "KF_INITIAL_ERROR": 10.0,
        "_KF_INITIAL_ERROR_help": "Initial state uncertainty when a new track starts",
        "COST_OF_NON_ASSIGNMENT": 10.0,
        "_COST_OF_NON_ASSIGNMENT_help": "Max Mahalanobis cost to accept a track-detection match",
        "KF_GATING_THRESHOLD_D2": 9.21,
        "_KF_GATING_THRESHOLD_D2_help": "Chi-squared gate at 99% confidence for 2 DOF (default 9.21)",

        "ENABLE_ROI": True,
        "_ENABLE_ROI_help": "Restrict detection to a sub-region of the fan image",
        "ROI_X_START": 0.05,
        "_ROI_X_START_help": "Left boundary of detection zone (0.0-1.0)",
        "ROI_X_END": 0.95,
        "_ROI_X_END_help": "Right boundary of detection zone (0.0-1.0)",
        "ROI_Y_START": 0.15,
        "_ROI_Y_START_help": "Top boundary of detection zone — skip noisy near-range area (0.1-0.3)",
        "ROI_Y_END": 0.95,
        "_ROI_Y_END_help": "Bottom boundary of detection zone (0.8-0.95)",

        "USE_STATIC_PATTERN": False,
        "_USE_STATIC_PATTERN_help": "Subtract static background pattern before thresholding. Set true after running Option P.",
        "STATIC_PATTERN_FILE": pattern_path,
        "_STATIC_PATTERN_FILE_help": "Path to precomputed .npy pattern file — generated with Option P",
        "BW_THRESHOLD": 0.15,
        "_BW_THRESHOLD_help": "Minimum difference from pattern to count as foreground (0.0-1.0 in relative mode)",
        "BW_THRESHOLD_MODE": "relative",
        "_BW_THRESHOLD_MODE_help": "relative = fraction of 255 | absolute = direct 0-255 value",
        "PATTERN_DETECTION_MODE": "subtract_positive",
        "_PATTERN_DETECTION_MODE_help": "subtract_positive = MATLAB parity max(frame-pattern,0); abs_diff = legacy |frame-pattern| mask",
        "FAN_IMAGE_X_SIZE": 400,
        "_FAN_IMAGE_X_SIZE_help": "Fan image width in pixels. Keep pattern/playback consistent. MATLAB tutorial often uses 500.",
        "FAN_HALF_ANGLE_DEG": 14.0,
        "_FAN_HALF_ANGLE_DEG_help": "Half field-of-view (degrees) used in fan mapping. MATLAB legacy mapscan uses 14.4.",
        "FAN_BEAM_SMOOTH": 4,
        "_FAN_BEAM_SMOOTH_help": "Beam interpolation factor (valid: 1, 4, 8). Must match pattern/playback mapping.",

        "PREY_PREDATOR_THRESHOLD_CM": 5.0,
        "_PREY_PREDATOR_THRESHOLD_CM_help": "Fish length below this is counted as prey; otherwise predator",
        "SCHOOL_MIN_FISHES": 10,
        "_SCHOOL_MIN_FISHES_help": "Minimum fish count in a frame before school detection logic is evaluated",
        "VALIDATION_MAX_FISH_PERIOD_S": 300,
        "_VALIDATION_MAX_FISH_PERIOD_S_help": "Validation time-window length in seconds (MATLAB default 300)",
    }


def _tokenize_path(path_text):
    """Normalize a path-like string into lowercase alphanumeric tokens."""
    normalized = os.path.normpath(str(path_text)).replace("\\", "/").lower()
    return {token for token in re.split(r"[^a-z0-9]+", normalized) if token}


def _candidate_directory_meta(candidate_path):
    """
    Return (directory_tokens, is_config_root, in_config_tree, directory_depth)
    for a candidate config file path.
    """
    config_root_norm = os.path.normpath(CONFIG_ROOT_DIR)
    candidate_norm = os.path.normpath(candidate_path)
    in_config_tree = (
        candidate_norm == config_root_norm or
        candidate_norm.startswith(config_root_norm + os.sep)
    )

    if in_config_tree:
        rel_path = os.path.relpath(candidate_norm, config_root_norm)
        rel_dir = os.path.dirname(rel_path)
        if rel_dir == ".":
            rel_dir = ""
        directory_tokens = _tokenize_path(rel_dir)
        directory_depth = 0 if not rel_dir else rel_dir.count(os.sep) + 1
        is_config_root = directory_depth == 0
        return directory_tokens, is_config_root, True, directory_depth

    return set(), False, False, 0


def _select_best_path_for_video(video_filename, candidate_paths):
    """
    Choose the best config path for a video based on directory-name token overlap.
    If no candidate matches video path tokens, prefer config/ root for compatibility.
    """
    if not candidate_paths:
        return None

    video_tokens = _tokenize_path(video_filename)

    def _rank(candidate_path):
        directory_tokens, is_config_root, in_config_tree, directory_depth = _candidate_directory_meta(candidate_path)
        overlap = len(video_tokens.intersection(directory_tokens))
        return (
            overlap,
            1 if is_config_root else 0,
            1 if in_config_tree else 0,
            -directory_depth,
        )

    return max(candidate_paths, key=_rank)


def _find_existing_config_path(video_filename, config_filename):
    """Find an existing config file by recursively scanning config/."""
    candidate_paths = []

    if os.path.isdir(CONFIG_ROOT_DIR):
        for current_root, _, filenames in os.walk(CONFIG_ROOT_DIR):
            if config_filename in filenames:
                candidate_paths.append(os.path.join(current_root, config_filename))

    # Legacy fallback: allow top-level config files outside config/.
    if os.path.exists(config_filename):
        candidate_paths.append(config_filename)

    return _select_best_path_for_video(video_filename, candidate_paths)


def _get_template_target_path(video_filename, config_filename):
    """
    Choose where to create a new template if no config exists.
    Prefer matching subfolders under config/; otherwise use config/ root.
    """
    if not os.path.isdir(CONFIG_ROOT_DIR):
        return os.path.join(CONFIG_ROOT_DIR, config_filename)

    candidate_dirs = {CONFIG_ROOT_DIR}
    for current_root, _, filenames in os.walk(CONFIG_ROOT_DIR):
        if any(name.startswith("config_") and name.endswith(".json") for name in filenames):
            candidate_dirs.add(current_root)

    synthesized_paths = [os.path.join(directory, config_filename) for directory in candidate_dirs]
    selected = _select_best_path_for_video(video_filename, synthesized_paths)
    if not selected:
        return os.path.join(CONFIG_ROOT_DIR, config_filename)

    selected_tokens, _, _, _ = _candidate_directory_meta(selected)
    if selected_tokens:
        return selected

    # No folder-name match -> use default config/ root.
    return os.path.join(CONFIG_ROOT_DIR, config_filename)


def get_config_for_video(video_filename):
    """
    Get configuration for a specific video file
    
    Args:
        video_filename: Path to ARIS video file
    
    Returns:
        FishTrackingConfig: Configuration object
    """
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    config_filename = f"config_{base_name}.json"
    existing_config_path = _find_existing_config_path(video_filename, config_filename)
    if existing_config_path:
        return FishTrackingConfig(existing_config_path)

    # Not found - create template in best-matching config folder (or config/ root).
    template_path = _get_template_target_path(video_filename, config_filename)
    template_dir = os.path.dirname(template_path)
    if template_dir:
        os.makedirs(template_dir, exist_ok=True)

    config = FishTrackingConfig()
    print("\nNo config found for this video. Creating template...")

    template = _build_config_template(base_name)
    try:
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=4)
        print(f"Created template: {template_path}")
        print("   Edit this file to tune parameters for your video")
    except Exception as e:
        print(f"Failed to create template: {e}")

    return config


if __name__ == "__main__":
    # Demo the unified config system
    print("Unified Fish Tracking Configuration System")
    print("=" * 60)
    
    # Show default config
    config = FishTrackingConfig()
    config.print_summary()
    
    # Demo video-specific config
    print(f"\nVideo-specific configuration example:")
    video_config = get_config_for_video("example_video.aris")
    
    print(f"\nConfiguration management:")
    print(f"   • Default parameters work for most videos")
    print(f"   • Create config_[videoname].json for custom tuning")
    print(f"   • Edit JSON file to adjust detection parameters")
    print(f"   • Configuration auto-loads when video is opened")
