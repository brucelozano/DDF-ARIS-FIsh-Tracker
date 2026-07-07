#!/usr/bin/env python3
"""
Simplified Fish Detector
Uses unified configuration system - no presets, just direct parameter control
"""

import numpy as np
import cv2
import time
import os
from collections import deque
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fish_tracking_config import FishTrackingConfig
from mangrove_distance_mapper import MangroveDistanceMapper, ValidationMaskFilter


class KalmanFilterCV2D:
    """
    2D Constant Velocity Kalman Filter - MATLAB equivalent
    State: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    """
    
    def __init__(self, initial_xy, process_noise, measurement_noise, initial_error):
        """
        Initialize Kalman filter with constant velocity model
        
        Args:
            initial_xy: Initial position (x, y)
            process_noise: Process noise variance (scalar)
            measurement_noise: Measurement noise variance (scalar)  
            initial_error: Initial state covariance (scalar)
        """
        # State vector: [x, y, vx, vy]^T
        self.x = np.array([[float(initial_xy[0])], 
                          [float(initial_xy[1])], 
                          [0.0], 
                          [0.0]], dtype=np.float64)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float64)
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float64)
        
        # State covariance matrix
        self.P = np.eye(4, dtype=np.float64) * float(initial_error)
        
        # Process noise covariance matrix
        self.Q = np.eye(4, dtype=np.float64) * float(process_noise)
        
        # Measurement noise covariance matrix
        self.R = np.eye(2, dtype=np.float64) * float(measurement_noise)
        
        # Identity matrix
        self.I = np.eye(4, dtype=np.float64)
    
    def predict(self):
        """
        Predict step - returns predicted measurement and innovation covariance
        
        Returns:
            z_pred: Predicted measurement [x, y]
            S: Innovation covariance matrix (2x2)
        """
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Predicted measurement
        z_pred = (self.H @ self.x).reshape(-1)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        return z_pred, S
    
    def correct(self, z_measured):
        """
        Correction step - update state with measurement
        
        Args:
            z_measured: Measurement [x, y]
            
        Returns:
            z_corrected: Corrected state estimate [x, y]
        """
        z = np.asarray(z_measured, dtype=np.float64).reshape(2, 1)
        
        # Innovation
        y = z - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (self.I - K @ self.H) @ self.P
        
        # Return corrected position estimate
        z_corrected = (self.H @ self.x).reshape(-1)
        return z_corrected
    
    def distance(self, z_measured):
        """
        Compute Mahalanobis distance to measurement (MATLAB equivalent)
        
        Args:
            z_measured: Measurement [x, y]
            
        Returns:
            distance: Mahalanobis distance
        """
        z = np.asarray(z_measured, dtype=np.float64)
        z_pred = (self.H @ self.x).reshape(-1)
        
        # Innovation
        dz = z - z_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance squared
        d2 = float(dz.T @ np.linalg.inv(S) @ dz)
        
        return np.sqrt(max(d2, 0.0))


class FishTrack:
    """Fish track with history suitable for offline analysis exports"""
    
    def __init__(self, track_id, bbox, centroid, config,
                 major_axis=None, minor_axis=None, orientation=0.0, eccentricity=0.0,
                 frame_index=None):
        self.id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.config = config
        
        # Track lifecycle
        self.age = 1
        self.hits = 1
        self.total_visible_count = 1
        self.invisible_count = 0
        
        # Initialize Kalman filter if enabled
        self.kalman = None
        if getattr(config, 'USE_KALMAN', False):
            self.kalman = KalmanFilterCV2D(
                initial_xy=centroid,
                process_noise=config.KF_PROCESS_NOISE,
                measurement_noise=config.KF_MEASUREMENT_NOISE,
                initial_error=config.KF_INITIAL_ERROR
            )
        
        # Ellipse-fit values (MATLAB parity); fall back to bbox if not provided
        if major_axis is None:
            major_axis = float(max(bbox[2], bbox[3])) if bbox else 0.0
        if minor_axis is None:
            minor_axis = float(min(bbox[2], bbox[3])) if bbox else 0.0

        # Histories (keep complete history for reporting)
        self.position_history = [centroid]
        self.frame_history = [frame_index] if frame_index is not None else []
        self.bbox_history = [bbox]
        self.major_axis_history = [major_axis]
        self.minor_axis_history = [minor_axis]
        self.orientation_history = [orientation]
        self.eccentricity_history = [eccentricity]
        self.dist_from_mangrove_history = []  # OPTIONAL: populated only if mangrove mask enabled
        
        # Visual properties
        np.random.seed(track_id)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())
    
    def predict_next_position(self):
        """Predict next position using Kalman filter or simple linear prediction"""
        if getattr(self.config, 'USE_KALMAN', False) and self.kalman is not None:
            # Use Kalman filter prediction
            z_pred, _ = self.kalman.predict()
            return (float(z_pred[0]), float(z_pred[1]))
        else:
            # Fallback to simple linear prediction
            if len(self.position_history) >= 2:
                last_pos = self.position_history[-1]
                prev_pos = self.position_history[-2]
                
                # Linear prediction
                dx = last_pos[0] - prev_pos[0]
                dy = last_pos[1] - prev_pos[1]
                
                predicted_x = last_pos[0] + dx
                predicted_y = last_pos[1] + dy
                
                return (predicted_x, predicted_y)
            else:
                return self.centroid
    
    def update_with_detection(self, bbox, centroid, frame_index=None, distance_from_mangrove=0.0,
                              major_axis=None, minor_axis=None, orientation=0.0, eccentricity=0.0):
        """Update track with new detection"""
        self.bbox = bbox
        self.centroid = centroid
        self.age += 1
        self.hits += 1
        self.total_visible_count += 1
        self.invisible_count = 0
        self.position_history.append(centroid)
        self.bbox_history.append(bbox)
        if major_axis is None:
            major_axis = float(max(bbox[2], bbox[3])) if bbox else 0.0
        if minor_axis is None:
            minor_axis = float(min(bbox[2], bbox[3])) if bbox else 0.0
        self.major_axis_history.append(major_axis)
        self.minor_axis_history.append(minor_axis)
        self.orientation_history.append(orientation)
        self.eccentricity_history.append(eccentricity)
        self.dist_from_mangrove_history.append(distance_from_mangrove)
        if frame_index is not None:
            self.frame_history.append(frame_index)
    
    def update_without_detection(self):
        """Update when track is not detected"""
        self.age += 1
        self.invisible_count += 1
    
    def should_be_deleted(self):
        """Check if track should be deleted - MATLAB equivalent logic"""
        # Delete if invisible too long (MATLAB: invisibleForTooLong)
        if self.invisible_count >= self.config.MAX_INVISIBLE_COUNT:
            return True
        
        # Delete if young track with poor visibility ratio (MATLAB logic)
        if hasattr(self.config, 'AGE_THRESHOLD') and hasattr(self.config, 'MIN_VISIBILITY_RATIO'):
            visibility_ratio = self.total_visible_count / max(self.age, 1)
            if (self.age < self.config.AGE_THRESHOLD and 
                visibility_ratio < self.config.MIN_VISIBILITY_RATIO):
                return True
        else:
            # Fallback to old logic
            if self.age < 10 and self.hits < self.config.MIN_TRACK_HITS:
                return True
        
        return False
    
    def net_displacement(self):
        """Straight-line distance from first to last observed position (pixels).

        This distinguishes real fish (which travel across the frame) from
        stationary artifacts that jitter in place — an artifact oscillates
        but its net displacement stays near zero.
        """
        if len(self.position_history) < 2:
            return 0.0
        first = self.position_history[0]
        last = self.position_history[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        return (dx*dx + dy*dy) ** 0.5

    def recent_displacement(self, window_frames):
        """Straight-line displacement over the most recent N observations.

        This catches long-lived artifacts that pass lifetime displacement once
        but then remain effectively stationary for the rest of the run.
        """
        if len(self.position_history) < 2:
            return 0.0
        n = max(int(window_frames), 2)
        recent = self.position_history[-n:] if len(self.position_history) >= n else self.position_history
        first = recent[0]
        last = recent[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        return (dx*dx + dy*dy) ** 0.5

    def is_reliable(self):
        """Check if track is reliable for display.

        A track must have enough detections AND have moved enough to be
        considered a real fish (filters stationary sonar artifacts).

        The displacement check is age-gated: young tracks are shown based
        on hit count alone so real fish get boxes immediately.  The
        displacement filter only kicks in once the track has lived long
        enough that a genuine fish would have moved MIN_TRACK_DISPLACEMENT
        pixels — stationary artifacts will fail at that point.
        """
        if self.total_visible_count < self.config.MIN_TRACK_HITS:
            return False
        min_disp = getattr(self.config, 'MIN_TRACK_DISPLACEMENT', 0.0)
        if min_disp > 0:
            disp_age = getattr(self.config, 'MIN_DISPLACEMENT_AGE', 10)
            if self.age >= disp_age:
                # Keep lifetime movement gate.
                if self.net_displacement() < min_disp:
                    return False

                # Also require some recent movement so stationary bright
                # artifacts don't stay labeled as fish indefinitely.
                recent_min_disp = 0.35 * float(min_disp)
                if self.recent_displacement(disp_age) < recent_min_disp:
                    return False
        return True


class StaticPatternRemover:
    """Simple static pattern support for SimpleFishDetector"""
    
    def __init__(self, config):
        self.config = config
        self.static_pattern = None
        self.pattern_computed = False
        self.frame_accumulator = None
        self.frame_count = 0
        self.max_frames_for_pattern = getattr(config, 'STATIC_PATTERN_FRAMES', 200)
        
        # Try to load offline static pattern if provided
        try:
            if getattr(self.config, 'USE_STATIC_PATTERN', False) and getattr(self.config, 'STATIC_PATTERN_FILE', ""):
                path = self.config.STATIC_PATTERN_FILE
                if path.lower().endswith('.npy') and os.path.exists(path):
                    self.static_pattern = np.load(path).astype(np.uint8)
                    self.pattern_computed = True
                    print(f"Static pattern loaded from NPY: {path} (shape: {self.static_pattern.shape})")
                elif path.lower().endswith('.mat') and os.path.exists(path):
                    from scipy.io import loadmat
                    mat = loadmat(path)
                    # Try common keys
                    for key in ['Pattern', 'pattern', 'static_pattern', 'bluebar', 'P']:
                        if key in mat:
                            self.static_pattern = mat[key].astype(np.uint8)
                            self.pattern_computed = True
                            print(f"Static pattern loaded from MAT: {path} (shape: {self.static_pattern.shape})")
                            break
                else:
                    print(f"Static pattern file not found: {path}")
        except Exception as e:
            print(f"Warning: failed to load static pattern: {e}")
    
    def add_frame_for_pattern(self, frame):
        """Add frame to pattern computation"""
        if self.pattern_computed or self.frame_count >= self.max_frames_for_pattern:
            return
        
        self.frame_count += 1
        
        if self.frame_accumulator is None:
            self.frame_accumulator = frame.astype(np.float64)
        else:
            self.frame_accumulator += frame.astype(np.float64)
        
        # Compute average after enough frames
        if self.frame_count >= self.max_frames_for_pattern:
            self.static_pattern = (self.frame_accumulator / self.frame_count).astype(np.uint8)
            self.pattern_computed = True
            print(f"Static pattern computed from {self.frame_count} frames")
    
    def remove_pattern(self, frame):
        """Remove static pattern from frame"""
        if not self.pattern_computed or self.static_pattern is None:
            return frame
        
        if frame.shape != self.static_pattern.shape:
            return frame
        
        result = np.clip(frame.astype(np.int16) - self.static_pattern.astype(np.int16), 0, 255)
        return result.astype(np.uint8)
    
    def difference_mask(self, frame):
        """Create a binary mask of differences from static pattern"""
        if not self.pattern_computed or self.static_pattern is None:
            return None
        
        if frame.shape != self.static_pattern.shape:
            return None
        
        diff = np.abs(frame.astype(np.int16) - self.static_pattern.astype(np.int16)).astype(np.uint8)
        
        # Determine threshold
        mode = str(getattr(self.config, 'BW_THRESHOLD_MODE', 'auto')).lower()
        bw = getattr(self.config, 'BW_THRESHOLD', 0.0)
        
        if mode == 'absolute' or (mode == 'auto' and bw > 1.0):
            thr = int(min(max(bw, 0), 255))
        elif mode == 'relative' or (mode == 'auto' and 0.0 <= bw <= 1.0):
            thr = int(min(max(bw, 0.0), 1.0) * 255)
        else:
            thr = 25  # Default threshold
        
        _, binary_mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        return binary_mask


class FishDetector:
    """
    Simplified fish detector - no presets, just configurable parameters
    Now with static pattern support
    """
    
    def __init__(self, config):
        self.config = config
        
        # Static pattern support
        self.static_pattern_remover = StaticPatternRemover(config)
        
        # Tracking state
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        # Archive of all reliable tracks (MATLAB allTracks equivalent)
        self.all_tracks = []  # tracks that became reliable before being deleted
        
        # Mangrove distance mapper (OPTIONAL - initialized on first frame)
        self.mangrove_mapper = None
        self.validation_filter = None
        mangrove_file = getattr(config, 'MANGROVE_MASK_FILE', "")
        validation_file = getattr(config, 'VALIDATION_MASK_FILE', "")
        self.mangrove_mapper_pending = mangrove_file if mangrove_file else None
        self.validation_filter_pending = validation_file if validation_file else None
        
        # Statistics
        self.total_detections = 0
        
        kalman_status = "enabled" if getattr(self.config, 'USE_KALMAN', False) else "disabled"
        print(f"Fish Detector initialized (Kalman: {kalman_status})")
        self.config.print_summary()
    
    def detect_fish(self, frame, frame_index=None):
        """Detect fish with optional static-pattern differencing.

        When static pattern mode is active and a pattern is available, use
        BW_THRESHOLD/BW_THRESHOLD_MODE on frame-vs-pattern difference.
        Otherwise fall back to INTENSITY_THRESHOLD on the working frame.
        """
        self.frame_count += 1
        
        # Initialize mangrove mapper on first frame (now we know image shape)
        if self.mangrove_mapper_pending and self.mangrove_mapper is None:
            from mangrove_distance_mapper import MangroveDistanceMapper
            self.mangrove_mapper = MangroveDistanceMapper(
                self.mangrove_mapper_pending, 
                frame.shape[:2]
            )
            self.mangrove_mapper_pending = None
        
        # Initialize validation filter on first frame
        if self.validation_filter_pending and self.validation_filter is None:
            from mangrove_distance_mapper import ValidationMaskFilter
            self.validation_filter = ValidationMaskFilter(
                self.validation_filter_pending,
                frame.shape[:2]
            )
            self.validation_filter_pending = None
        
        # Build static pattern from initial frames if not yet computed
        if not self.static_pattern_remover.pattern_computed:
            self.static_pattern_remover.add_frame_for_pattern(frame)
        
        # Detection mask:
        # - Pattern mode: threshold absolute frame-vs-pattern difference
        # - Fallback: intensity threshold on corrected frame
        use_pattern = getattr(self.config, 'USE_STATIC_PATTERN', False)
        mask = None
        if use_pattern:
            mask = self.static_pattern_remover.difference_mask(frame)

        if mask is None:
            frame_corrected = self.static_pattern_remover.remove_pattern(frame) if use_pattern else frame
            threshold = int(getattr(self.config, 'INTENSITY_THRESHOLD', 40))
            _, mask = cv2.threshold(frame_corrected, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply ROI mask if enabled
        if self.config.ENABLE_ROI:
            mask = self._apply_roi_mask(mask)
        
        # Morphological operations
        mask = self._clean_mask(mask)
        
        # Find detections (MATLAB: blobAnalyser.step)
        detections = self._find_detections(mask)
        self.total_detections += len(detections)
        
        # Update tracks
        self._update_tracking(detections, frame_index=frame_index)
        
        # Return reliable tracks
        fish_detections = []
        for track in self.tracks:
            if track.is_reliable():
                fish_detections.append({
                    'bbox': track.bbox,
                    'centroid': track.centroid,
                    'id': track.id,
                    'confidence': min(track.hits / 5.0, 1.0),
                    'pattern_active': getattr(self.config, 'USE_STATIC_PATTERN', False) and self.static_pattern_remover.pattern_computed
                })
        
        return fish_detections
    
    def _apply_roi_mask(self, mask):
        """Apply Region of Interest mask"""
        height, width = mask.shape
        
        x1 = int(width * self.config.ROI_X_START)
        x2 = int(width * self.config.ROI_X_END)
        y1 = int(height * self.config.ROI_Y_START)
        y2 = int(height * self.config.ROI_Y_END)
        
        # Create ROI mask
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        
        # Apply ROI
        return cv2.bitwise_and(mask, roi_mask)
    
    def _clean_mask(self, mask):
        """Clean mask with morphological operations"""
        # Opening (remove noise)
        if self.config.MORPH_OPENING_SIZE > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.MORPH_OPENING_SIZE, self.config.MORPH_OPENING_SIZE)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing (fill holes)
        if self.config.MORPH_CLOSING_SIZE > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.MORPH_CLOSING_SIZE, self.config.MORPH_CLOSING_SIZE)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    @staticmethod
    def _ellipse_from_moments(blob_mask):
        """Compute ellipse properties from image moments (MATLAB regionprops equivalent).

        Returns (major_axis, minor_axis, orientation_deg, eccentricity) or None
        if the blob is too small for a meaningful fit.
        """
        m = cv2.moments(blob_mask, binaryImage=True)
        m00 = m['m00']
        if m00 < 3:
            return None

        # Normalized second central moments
        mu20 = m['mu20'] / m00
        mu02 = m['mu02'] / m00
        mu11 = m['mu11'] / m00

        common = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2)
        major = 2.0 * np.sqrt(2.0) * np.sqrt(max(mu20 + mu02 + common, 0.0))
        minor = 2.0 * np.sqrt(2.0) * np.sqrt(max(mu20 + mu02 - common, 0.0))
        orientation = np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02))
        eccentricity = np.sqrt(1.0 - (minor / major) ** 2) if major > 0 else 0.0

        return major, minor, orientation, eccentricity

    def _find_detections(self, mask):
        """Find fish detections in cleaned mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        detections = []
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            centroid = (centroids[i, 0], centroids[i, 1])
            
            # Filter by area
            if area < self.config.MIN_FISH_AREA or area > self.config.MAX_FISH_AREA:
                continue
            
            # Filter by aspect ratio
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                if (aspect_ratio < self.config.MIN_ASPECT_RATIO or 
                    aspect_ratio > self.config.MAX_ASPECT_RATIO):
                    continue
            
            # Compute ellipse properties from image moments (MATLAB parity)
            blob_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            ellipse = self._ellipse_from_moments(blob_mask)
            if ellipse is not None:
                major_axis, minor_axis, orientation, eccentricity = ellipse
            else:
                major_axis = float(max(w, h))
                minor_axis = float(min(w, h))
                orientation = 0.0
                eccentricity = 0.0

            detections.append({
                'bbox': (x, y, w, h),
                'centroid': centroid,
                'area': area,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'orientation': orientation,
                'eccentricity': eccentricity,
            })
        
        return detections
    
    def _predict_new_locations(self):
        """Predict new locations for all tracks (MATLAB: predictNewLocationsOfTracks).
        
        Must be called BEFORE assignment so that distance() uses the
        predicted state, not the stale corrected state from the previous frame.
        """
        use_kalman = getattr(self.config, 'USE_KALMAN', False)
        for track in self.tracks:
            if use_kalman and track.kalman is not None:
                try:
                    z_pred, _ = track.kalman.predict()
                    # Shift bbox center to predicted centroid (matches MATLAB)
                    if track.bbox:
                        x, y, w, h = track.bbox
                        track.bbox = (int(z_pred[0] - w / 2),
                                      int(z_pred[1] - h / 2), w, h)
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    def _update_tracking(self, detections, frame_index=None):
        """Update tracks with new detections - MATLAB equivalent tracking loop"""
        # Step 1: Predict new locations for ALL tracks (MATLAB: predictNewLocationsOfTracks)
        # This advances each Kalman state so distance() uses the predicted position.
        self._predict_new_locations()
        
        # Step 2: Assign detections to tracks using predicted state
        assignments = self._assign_detections_to_tracks(detections)
        
        # Update tracks (includes Kalman correction)
        self._apply_assignments(detections, assignments, frame_index=frame_index)
        
        # Delete old tracks
        self._delete_old_tracks()
        
        # Create new tracks (pass frame_index so initial detection has its frame recorded)
        self._create_new_tracks(detections, assignments, frame_index=frame_index)
    
    def _assign_detections_to_tracks(self, detections):
        """Assign detections to tracks using Hungarian algorithm with Mahalanobis distance (MATLAB equivalent)"""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        use_kalman = getattr(self.config, 'USE_KALMAN', False)
        gating_threshold = float(getattr(self.config, 'KF_GATING_THRESHOLD_D2', 9.21))
        cost_of_non_assignment = float(getattr(self.config, 'COST_OF_NON_ASSIGNMENT', 10.0))
        large_cost = 1e6
        
        # Build cost matrix
        cost_matrix = np.full((len(self.tracks), len(detections)), large_cost, dtype=np.float64)
        
        for i, track in enumerate(self.tracks):
            if use_kalman and track.kalman is not None:
                # Use Kalman filter distance (MATLAB equivalent)
                for j, detection in enumerate(detections):
                    try:
                        # Compute Mahalanobis distance with gating
                        distance = track.kalman.distance(detection['centroid'])
                        distance_squared = distance ** 2
                        
                        # Apply gating threshold
                        if distance_squared <= gating_threshold:
                            cost_matrix[i, j] = distance
                        # else: leave as large_cost (gated out)
                    except (np.linalg.LinAlgError, ValueError):
                        # Fallback to Euclidean if Kalman fails
                        predicted_pos = track.predict_next_position()
                        det_pos = detection['centroid']
                        distance = np.sqrt((predicted_pos[0] - det_pos[0])**2 + 
                                         (predicted_pos[1] - det_pos[1])**2)
                        cost_matrix[i, j] = distance
            else:
                # Fallback to Euclidean distance
                predicted_pos = track.predict_next_position()
                for j, detection in enumerate(detections):
                    det_pos = detection['centroid']
                    distance = np.sqrt((predicted_pos[0] - det_pos[0])**2 + 
                                     (predicted_pos[1] - det_pos[1])**2)
                    cost_matrix[i, j] = distance
        
        # Apply Hungarian algorithm
        try:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # Filter by cost of non-assignment threshold (MATLAB equivalent)
            valid_assignments = []
            assigned_tracks = set()
            assigned_detections = set()
            
            for t_idx, d_idx in zip(track_indices, detection_indices):
                if cost_matrix[t_idx, d_idx] <= cost_of_non_assignment:
                    valid_assignments.append((t_idx, d_idx))
                    assigned_tracks.add(t_idx)
                    assigned_detections.add(d_idx)
            
            # Find unassigned
            unassigned_tracks = [i for i in range(len(self.tracks)) if i not in assigned_tracks]
            unassigned_detections = [i for i in range(len(detections)) if i not in assigned_detections]
            
            return valid_assignments, unassigned_tracks, unassigned_detections
            
        except Exception as e:
            print(f"Warning: Assignment failed: {e}, falling back to no assignments")
            return [], list(range(len(self.tracks))), list(range(len(detections)))
    
    def _apply_assignments(self, detections, assignments, frame_index=None):
        """Apply track assignments with Kalman correction (MATLAB equivalent)"""
        valid_assignments, unassigned_tracks, unassigned_detections = assignments
        use_kalman = getattr(self.config, 'USE_KALMAN', False)
        
        # Update assigned tracks with Kalman correction
        for track_idx, detection_idx in valid_assignments:
            detection = detections[detection_idx]
            track = self.tracks[track_idx]
            
            if use_kalman and track.kalman is not None:
                try:
                    # Correct Kalman filter with measurement (MATLAB equivalent)
                    corrected_xy = track.kalman.correct(detection['centroid'])
                    centroid = (float(corrected_xy[0]), float(corrected_xy[1]))
                except (np.linalg.LinAlgError, ValueError):
                    # Fallback to raw detection if correction fails
                    centroid = detection['centroid']
            else:
                # Use raw detection centroid
                centroid = detection['centroid']
            
            # Calculate mangrove distance (MATLAB equivalent) 
            distance_px = 0.0
            if self.mangrove_mapper and self.mangrove_mapper.is_active():
                distance_px = self.mangrove_mapper.get_distance_at_position(
                    centroid[0], centroid[1]
                )
            
            # Update track with corrected position, distance, and ellipse properties
            track.update_with_detection(
                detection['bbox'], centroid, frame_index=frame_index,
                distance_from_mangrove=distance_px,
                major_axis=detection.get('major_axis'),
                minor_axis=detection.get('minor_axis'),
                orientation=detection.get('orientation', 0.0),
                eccentricity=detection.get('eccentricity', 0.0))
        
        # Update unassigned tracks -- predict() was already called in
        # _predict_new_locations, so we only mark them invisible here
        # (matches MATLAB updateUnassignedTracks: just increment age/invisible).
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].update_without_detection()
    
    def _delete_old_tracks(self):
        """Remove lost tracks (MATLAB: deleteLostTracks).
        
        Reliable tracks are archived into all_tracks before removal.
        IDs are never reused -- matches MATLAB's monotonic nextId counter.
        """
        kept_tracks = []
        for track in self.tracks:
            if track.should_be_deleted():
                if track.is_reliable():
                    self.all_tracks.append(track)
            else:
                kept_tracks.append(track)
        self.tracks = kept_tracks
    
    def _create_new_tracks(self, detections, assignments, frame_index=None):
        """Create new tracks from unassigned detections (MATLAB: createNewTracks).
        
        Each unassigned detection starts a fresh track with a new monotonic ID.
        """
        _, _, unassigned_detections = assignments
        
        for detection_idx in unassigned_detections:
            if len(self.tracks) >= self.config.MAX_TOTAL_TRACKS:
                break
            
            detection = detections[detection_idx]
            new_track = FishTrack(
                self.next_track_id,
                detection['bbox'],
                detection['centroid'],
                self.config,
                major_axis=detection.get('major_axis'),
                minor_axis=detection.get('minor_axis'),
                orientation=detection.get('orientation', 0.0),
                eccentricity=detection.get('eccentricity', 0.0),
                frame_index=frame_index
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
            
            if self.mangrove_mapper and self.mangrove_mapper.is_active():
                distance_px = self.mangrove_mapper.get_distance_at_position(
                    detection['centroid'][0], detection['centroid'][1]
                )
                new_track.dist_from_mangrove_history.append(distance_px)
            else:
                new_track.dist_from_mangrove_history.append(0.0)
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        if not isinstance(frame, np.ndarray):
            return frame
        
        # Ensure frame is writable
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # Draw ROI if enabled
        if self.config.ENABLE_ROI:
            frame = self._draw_roi(frame)
        
        # Draw fish detections
        for detection in detections:
            x, y, w, h = detection['bbox']
            fish_id = detection.get('id', '?')
            
            # Find track color
            color = self.config.BBOX_COLOR
            for track in self.tracks:
                if track.id == fish_id:
                    color = track.color
                    break
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.config.BBOX_THICKNESS)
            
            # Draw label
            if self.config.SHOW_FISH_IDS:
                label = f"Fish {fish_id}"
                label_size, _ = cv2.getTextSize(
                    label, self.config.LABEL_FONT, 
                    self.config.LABEL_FONT_SCALE, self.config.LABEL_THICKNESS
                )
                
                # Label background
                cv2.rectangle(frame, (x, y - label_size[1] - 3), 
                             (x + label_size[0] + 3, y), (0, 0, 0), -1)
                
                # Label text
                cv2.putText(frame, label, (x + 1, y - 2),
                           self.config.LABEL_FONT, self.config.LABEL_FONT_SCALE,
                           self.config.LABEL_COLOR, self.config.LABEL_THICKNESS)
        
        return frame
    
    def _draw_roi(self, frame):
        """Draw ROI overlay"""
        height, width = frame.shape[:2]
        
        x1 = int(width * self.config.ROI_X_START)
        x2 = int(width * self.config.ROI_X_END)
        y1 = int(height * self.config.ROI_Y_START)
        y2 = int(height * self.config.ROI_Y_END)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, "DETECTION ZONE", (x1 + 5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def reset(self):
        """Reset detector state"""
        self.tracks = []
        self.all_tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        self.static_pattern_remover = StaticPatternRemover(self.config)
        print("Fish detector reset")
    
    def get_statistics(self):
        """Get basic statistics"""
        reliable_tracks = sum(1 for track in self.tracks if track.is_reliable())
        return {
            'total_detections': self.total_detections,
            'active_tracks': len(self.tracks),
            'reliable_tracks': reliable_tracks
        }


def create_fish_detector_for_video(video_filename):
    """
    Create fish detector configured for a specific video
    
    Args:
        video_filename: Path to ARIS video file
    
    Returns:
        FishDetector: Configured detector
    """
    from fish_tracking_config import get_config_for_video
    
    config = get_config_for_video(video_filename)
    return FishDetector(config)


if __name__ == "__main__":
    # Demo the simple detector
    print("Simple Fish Detector Demo")
    print("=" * 40)
    
    # Test with default config
    from fish_tracking_config import FishTrackingConfig
    config = FishTrackingConfig()
    detector = FishDetector(config)
    
    print("Simple detector created successfully!")
    print("Key features:")
    print("  • Single configuration file")
    print("  • Per-video parameter tuning")
    print("  • No complex presets")
    print("  • Direct parameter control")
