#!/usr/bin/env python3
"""
Simple OpenCV ARIS Player
Streamlined player with unified configuration system
No presets, no complexity - just configurable fish tracking
"""

import os
import time
import cv2
import numpy as np
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from aris_sonar_processing import FinalCleanMapper
from fish_detector import FishDetector
from fish_tracking_config import FishTrackingConfig, get_config_for_video
from ddf_compat import read_file_layout, read_frame_header as parse_ddf_frame_header


def load_matlab_bluebar_colormap():
    """Load the exact MATLAB bluebar colormap and convert to OpenCV format"""
    try:
        # Try to load from the MATLAB file
        bluebar_path = './bluebar.mat'
        if os.path.exists(bluebar_path):
            import scipy.io
            bluebar_data = scipy.io.loadmat(bluebar_path)
            bluebar_cmap = bluebar_data['bluebar']
            # Convert to 8-bit and create OpenCV colormap
            colormap_8bit = (bluebar_cmap * 255).astype(np.uint8)
            return colormap_8bit
    except Exception as e:
        print(f"WARNING: Could not load bluebar.mat: {e}")
    # Fallback: Create approximation
    print("Creating bluebar approximation...")
    colors = []
    n_colors = 256
    for i in range(n_colors):
        t = i / (n_colors - 1)
        if t < 0.5:
            r = 0.0
            g = t * 0.5
            b = t * 2.0
        else:
            r = (t - 0.5) * 0.4
            g = 0.25 + (t - 0.5) * 0.75
            b = 1.0
        colors.append([int(b * 255), int(g * 255), int(r * 255)])
    return np.array(colors, dtype=np.uint8)


class CapMultiThreading:
    """Multi-threaded frame capture for maximum performance"""
    def __init__(self, filename, buffer_size=64):
        import threading
        from collections import deque
        self.filename = filename
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.capture_thread = None
        self.capturing = False
        self.file_info = None
        self.layout = None
        self.current_frame_num = 1
        self.max_frames = 0
        self.fid_lock = threading.Lock()
        # Open file and read header
        self.fid = open(filename, 'rb')
        self._read_header()

    def _read_header(self):
        """Read DDF metadata using version-aware compatibility parser."""
        self.layout = read_file_layout(self.fid)
        self.file_info = self.layout.to_file_info()
        self.max_frames = int(self.layout.numframes)

    def read_frame_header(self, frame_position):
        """Read detailed frame header with version-specific parsing."""
        return parse_ddf_frame_header(self.fid, self.layout, frame_position)

    def _capture_frames(self):
        """Background thread for frame capture"""
        import time as _time
        frame_size = self.layout.frame_data_length()
        frame_header_len = int(self.layout.frame_header_length)
        while self.capturing and self.current_frame_num <= self.max_frames:
            if len(self.frame_buffer) < self.buffer_size:
                with self.fid_lock:
                    frame_pos = self.layout.frame_header_offset(self.current_frame_num)
                    self.fid.seek(frame_pos)
                    self.fid.read(frame_header_len)
                    data_bytes = self.fid.read(frame_size)
                if len(data_bytes) == frame_size:
                    raw_data = np.frombuffer(data_bytes, dtype=np.uint8)
                    matlab_reshaped = raw_data.reshape(
                        self.file_info['numbeams'],
                        self.file_info['sampleperchannel'],
                        order='F'
                    )
                    if self.file_info['reverse'] == 0:
                        frame = np.fliplr(matlab_reshaped.T)
                    else:
                        frame = matlab_reshaped.T
                    self.frame_buffer.append((self.current_frame_num, frame))
                    self.current_frame_num += 1
                else:
                    break
            else:
                _time.sleep(0.001)
        self.capturing = False

    def start_capture(self):
        """Start the capture thread"""
        import threading
        self.capturing = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()

    def get_frame(self):
        """Get next frame from buffer"""
        import time as _time
        if self.frame_buffer:
            frame_num, frame = self.frame_buffer.popleft()
            return True, frame, frame_num
        elif self.capturing:
            _time.sleep(0.01)
            return self.get_frame()
        else:
            return False, None, 0

    def release(self):
        """Stop capture and close file"""
        self.capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.fid:
            self.fid.close()

    def get_frame_new(self, frame_number):
        """
        MATLAB equivalent: data = get_frame_new(data, framenumber)
        Read any specific frame by number
        """
        if frame_number < 1 or frame_number > self.file_info['numframes']:
            raise ValueError(f"Frame number {frame_number} out of range [1, {self.file_info['numframes']}]")
        data_length = self.layout.frame_data_length()
        frame_header_pos = self.layout.frame_header_offset(frame_number)
        frame_data_pos = self.layout.frame_data_offset(frame_number)
        with self.fid_lock:
            frame_header = self.read_frame_header(frame_header_pos)
            self.fid.seek(frame_data_pos)
            max_range = frame_header['windowstart'] + frame_header['windowlength']
            range_changed = False
            if (frame_header['windowstart'] != self.file_info['minrange'] or 
                max_range != self.file_info['maxrange']):
                self.file_info['minrange'] = frame_header['windowstart']
                self.file_info['maxrange'] = max_range
                range_changed = True
            frame_data = np.frombuffer(
                self.fid.read(data_length), 
                dtype=np.uint8
            ).reshape((self.file_info['numbeams'], self.file_info['sampleperchannel']), order='F')
            if self.file_info['reverse'] == 0:
                frame_data = frame_data.T
                frame_data = np.fliplr(frame_data)
            else:
                frame_data = frame_data.T
        return {
            'frame': frame_data,
            'frame_header': frame_header,
            'range_changed': range_changed,
            'minrange': self.file_info['minrange'],
            'maxrange': self.file_info['maxrange'],
            # Legacy convenience aliases for callers that read top-level fields.
            'sonartimestamp': frame_header.get('sonartimestamp', 0),
            'timestamp': frame_header.get('timestamp'),
            'datenum': frame_header.get('datenum'),
            'watertemp': frame_header.get('watertemp', 0.0),
            'salinity': frame_header.get('salinity', 0),
            'pressure': frame_header.get('pressure', 0.0),
        }

    def motion_correction(self, frame_data, velocity):
        """
        MATLAB equivalent: framenew = motion_correction(data, velocity)
        Corrects for platform forward velocity
        """
        if abs(velocity) < 1e-6:
            return frame_data.copy()
        n, m = frame_data.shape
        frame_new = np.zeros((n, m), dtype=np.uint8)
        max_range = self.file_info['maxrange']
        min_range = self.file_info['minrange']
        time = 2 * max_range / 1500
        delta_range = (max_range - min_range) / 512
        distance = velocity * time
        nbins = round(distance / delta_range)
        if m == 96:
            im = np.array([1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89]) - 1
            offset = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            imax = 8
        else:
            im = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]) - 1
            offset = np.array([0, 2, 1, 3])
            imax = 4
        nx = 0
        for indx in range(imax):
            beam_indices = im + offset[indx]
            if nx < 512:
                frame_new[nx:512, beam_indices] = frame_data[:512-nx, beam_indices]
            nx += nbins
        return frame_new


class OpenCVPlayer:
    """Simplified ARIS player with unified configuration"""
    
    def __init__(self, filename, enable_fish_detection=True, show_raw_window=False):
        self.filename = filename
        self.cap = None
        
        # Movie parameters
        self.first_frame = 1
        self.last_frame = 500
        self.current_frame = 1
        self.playing = False
        self.target_fps = 12.0
        
        # Performance tracking
        self.last_frame_time = None
        
        # Load colormap
        self.bluebar_cmap = load_matlab_bluebar_colormap()
        
        # Precomputed mapping data
        self.map_data = None
        self.mapscale = None  # MATLAB-style [hs, ws, i0, j0] for coordinate conversion
        
        # Fish detection
        self.enable_fish_detection = enable_fish_detection
        self.fish_detector = None
        # Master visibility switch for overlays (does NOT stop detection/tracking).
        self.show_detection_overlay = True

        # Video-specific config (single source of truth for geometry + tracking)
        self.video_config = self._load_video_config()
        self.fan_image_x_size, self.fan_half_angle_deg, self.fan_beam_smooth = self._fan_geometry_from_config()
        
        # UI options
        self.show_raw_window = show_raw_window
        
        # Saved tracking data for comparison
        self.saved_tracks = None
        self.saved_extent = None  # min/max for normalization
        self.show_saved_tracks = False
        self.load_saved_tracks()

        # Live analysis accumulation (for export after playback)
        self.frame_to_tracks = defaultdict(list)
        self.start_time = None
        self.playback_fps = None
        self.current_pass = 0
        self.pass_completed = False
        self.last_exported_pass = 0
        self.loop_just_reset = False
        
        # Track last complete pass stats
        self.last_complete_pass_stats = None
        
        if self.enable_fish_detection:
            print(f"Fish detection enabled for: {os.path.basename(filename)}")
        else:
            print(f"Video playback only (no fish detection)")

    def _load_video_config(self):
        """Load per-video config, falling back to defaults on any error."""
        try:
            return get_config_for_video(self.filename)
        except Exception as e:
            print(f"Warning: could not load video config ({e}); using defaults.")
            return FishTrackingConfig()

    def _fan_geometry_from_config(self):
        """Return sanitized fan-geometry parameters from config."""
        try:
            imagexsize = int(getattr(self.video_config, 'FAN_IMAGE_X_SIZE', 400))
        except Exception:
            imagexsize = 400
        try:
            half_angle = float(getattr(self.video_config, 'FAN_HALF_ANGLE_DEG', 14.0))
        except Exception:
            half_angle = 14.0
        try:
            smooth = int(getattr(self.video_config, 'FAN_BEAM_SMOOTH', 4))
        except Exception:
            smooth = 4

        imagexsize = max(32, imagexsize)
        smooth = max(1, smooth)
        return imagexsize, half_angle, smooth
    
    def precompute_mapping(self):
        """Precompute coordinate mapping for performance"""
        if self.map_data is not None:
            return
        
        print("⚡ Precomputing coordinate mapping...")
        
        if self.cap is None:
            self.cap = CapMultiThreading(self.filename)
        
        # Get first frame for range values
        first_frame_data = self.cap.get_frame_new(1)
        minrange = first_frame_data['minrange']
        maxrange = first_frame_data['maxrange']
        
        # Safety check -- catch zero/near-zero or inverted ranges
        if maxrange <= minrange or (maxrange - minrange) < 0.01:
            print(f"WARNING: Invalid range detected (min={minrange}, max={maxrange}), using defaults")
            minrange = 1.0
            maxrange = 5.0
        
        print(f"Using range: {minrange:.4f}m to {maxrange:.4f}m")
        
        # Create coordinate mapping
        nrows = self.cap.file_info['numbeams'] * self.fan_beam_smooth - self.fan_beam_smooth + 1
        half_angle = self.fan_half_angle_deg
        imagexsize = self.fan_image_x_size
        
        self.map_data = FinalCleanMapper.mapscan(
            imagexsize, maxrange, minrange,
            half_angle, nrows, self.cap.file_info['sampleperchannel']
        )
        
        # Store mapscale for MATLAB-style coordinate conversion
        self.mapscale = self.map_data.get('mapscale', None)
        if self.mapscale:
            hs, ws, i0, j0 = self.mapscale
            print(f"Mapscale computed (MATLAB-style):")
            print(f"   Height scale: {hs:.6f} m/pixel")
            print(f"   Width scale:  {ws:.6f} m/pixel")
            print(f"   Y origin (i0): {i0:.1f} pixels")
            print(f"   X origin (j0): {j0:.1f} pixels")
        
        print(
            f"Mapping precomputed: {imagexsize}x{self.map_data['iysize']} output "
            f"(half_angle={half_angle:.2f}, smooth={self.fan_beam_smooth})"
        )
    
    def load_saved_tracks(self):
        """Load saved tracking data for comparison if available"""
        try:
            import csv
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            tracking_file = os.path.join("exports/statistics", f"{base_name}_tracking_info.csv")
            
            if os.path.exists(tracking_file):
                self.saved_tracks = {}
                min_x, max_x = float('inf'), float('-inf')
                min_y, max_y = float('inf'), float('-inf')
                with open(tracking_file, 'r') as f:
                    reader = csv.DictReader(f)
                    frame_min, frame_max = None, None
                    for row in reader:
                        try:
                            frame_idx = int(row['Frame Index'])
                            fish_id = int(row['Fish ID'])
                            # Prefer pixel columns if present; else derive from meters
                            if 'X_px' in row and 'Y_px' in row and row['X_px'] and row['Y_px']:
                                x_px = int(float(row['X_px']))
                                y_px = int(float(row['Y_px']))
                            else:
                                # Fallback normalize later
                                x_px = None
                                y_px = None
                            x_pos = float(row.get('X Position (m)', 0.0) or 0.0)
                            y_pos = float(row.get('Y Position (m)', 0.0) or 0.0)
                            
                            if frame_idx not in self.saved_tracks:
                                self.saved_tracks[frame_idx] = []
                            
                            self.saved_tracks[frame_idx].append({
                                'fish_id': fish_id,
                                'x_m': x_pos,
                                'y_m': y_pos,
                                'x_px': x_px,
                                'y_px': y_px
                            })

                            # Track extents
                            if x_px is None or y_px is None:
                                # Only track meter extents if pixel columns are absent
                                if x_pos < min_x: min_x = x_pos
                                if x_pos > max_x: max_x = x_pos
                                if y_pos < min_y: min_y = y_pos
                                if y_pos > max_y: max_y = y_pos

                            # Frame coverage
                            if frame_min is None or frame_idx < frame_min:
                                frame_min = frame_idx
                            if frame_max is None or frame_idx > frame_max:
                                frame_max = frame_idx
                        except (ValueError, KeyError):
                            continue
                
                if self.saved_tracks:
                    total_points = sum(len(tracks) for tracks in self.saved_tracks.values())
                    # Avoid zero range
                    if max_x <= min_x: max_x = min_x + 1e-6
                    if max_y <= min_y: max_y = min_y + 1e-6
                    self.saved_extent = {
                        'min_x': min_x, 'max_x': max_x,
                        'min_y': min_y, 'max_y': max_y
                    }
                    self.saved_tracks_path = tracking_file
                    self.saved_frame_min = frame_min if frame_min is not None else 1
                    self.saved_frame_max = frame_max if frame_max is not None else self.saved_frame_min
                    from os.path import relpath
                    print(f"Saved tracks file: {relpath(self.saved_tracks_path)}")
                    print(f"   Frames covered: {self.saved_frame_min} - {self.saved_frame_max}")
                    print(f"Loaded {total_points} saved tracking points from offline analysis")
                    print(f"    Press 'T' during playback to toggle saved tracks overlay")
                else:
                    self.saved_tracks = None
            else:
                print(f"No saved tracking data found. Run offline analysis (Option R) first for comparison.")
                
        except Exception as e:
            print(f"Warning: Could not load saved tracks: {e}")
            self.saved_tracks = None
    
    def initialize_fish_detector(self):
        """Initialize fish detector with video-specific configuration"""
        if self.enable_fish_detection and self.fish_detector is None:
            try:
                self.fish_detector = FishDetector(self.video_config)
                print(f"Fish detector initialized for this video")
            except Exception as e:
                print(f"Failed to initialize fish detector: {e}")
                self.enable_fish_detection = False
    
    def draw_saved_tracks(self, image, frame_num):
        """Draw saved tracking data on the image"""
        if not self.saved_tracks or frame_num not in self.saved_tracks:
            return image
        
        # Ensure drawable buffer (some operations create non-contiguous views)
        if not isinstance(image, np.ndarray):
            return image
        if image.dtype != np.uint8 or not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image, dtype=np.uint8)

        # Normalize saved (meters) into image pixels using dataset extents
        height, width = image.shape[:2]
        ex = self.saved_extent or {'min_x':0.0,'max_x':1.0,'min_y':0.0,'max_y':1.0}
        rx = max(ex['max_x'] - ex['min_x'], 1e-6)
        ry = max(ex['max_y'] - ex['min_y'], 1e-6)
        
        for track in self.saved_tracks[frame_num]:
            # Prefer exact pixel coords if available
            if track.get('x_px') is not None and track.get('y_px') is not None:
                pixel_x = int(track['x_px'])
                pixel_y = int(track['y_px'])
            else:
                # Normalize from meters
                u = (track['x_m'] - ex['min_x']) / rx
                v = (track['y_m'] - ex['min_y']) / ry
                pixel_x = int(u * width)
                pixel_y = int(v * height)
            
            # Ensure coordinates are within image bounds
            pixel_x = max(0, min(width-1, pixel_x))
            pixel_y = max(0, min(height-1, pixel_y))
            
            # Draw saved track point in magenta for blue background contrast
            cv2.circle(image, (pixel_x, pixel_y), 10, (255, 0, 255), 2)  # Magenta ring
            cv2.circle(image, (pixel_x, pixel_y), 4, (255, 255, 255), -1)  # White center
            
            # Draw fish ID
            cv2.putText(image, f"S{track['fish_id']}", (pixel_x + 10, pixel_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # HUD indicating overlay active
        cv2.putText(image, "Saved tracks overlay ON (T)", (10, height-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return image
    
    def apply_colormap_opencv(self, image, colormap):
        """Apply MATLAB bluebar colormap"""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        if colormap is not None and len(colormap.shape) == 2 and colormap.shape[1] == 3:
            colored_rgb = colormap[image]
            colored_bgr = colored_rgb[:, :, [2, 1, 0]]
            return colored_bgr.astype(np.uint8)
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    def create_fan_image_fast(self, raw_frame):
        """Create fan-shaped image using precomputed mapping"""
        if self.map_data is None:
            self.precompute_mapping()
        
        # Apply beam smoothing
        if self.fan_beam_smooth > 1:
            if self.fan_beam_smooth == 4:
                frame = FinalCleanMapper.expand4(raw_frame)
            else:
                frame = FinalCleanMapper.smooth1(raw_frame, self.fan_beam_smooth, 'expand')
        else:
            frame = raw_frame.copy()
        frame[0, 0] = 0
        
        # Use precomputed mapping
        frame_flat = frame.flatten(order='F')
        svector_values = frame_flat[self.map_data['svector'] - 1]
        fan_image = svector_values.reshape(
            self.map_data['iysize'],
            self.fan_image_x_size,
            order='F'
        ).astype(np.uint8)
        
        return fan_image
    
    def run_player(self, first=1, last=500):
        """Run the simplified movie player"""
        print("SIMPLE ARIS MOVIE PLAYER")
        print("=" * 50)
        print(f"File: {self.filename}")
        print(f"Frames: {first} to {last}")
        
        try:
            # Create capture
            self.cap = CapMultiThreading(self.filename)
            
            # Print file info
            info = self.cap.file_info
            print(f"\nFile Info:")
            print(f"   Frames: {info['numframes']}")
            print(f"   Beams: {info['numbeams']}")
            print(f"   Samples: {info['sampleperchannel']}")
            print(f"   Frame rate: {info['framerate']:.1f} fps")
            
            # Validate frame range
            self.first_frame = max(1, first)
            self.last_frame = min(info['numframes'], last)
            
            print(f"\nPlaying frames {self.first_frame} to {self.last_frame}")
            
            # Precompute mapping (must happen before capture thread starts
            # to avoid race condition on the file handle)
            self.precompute_mapping()
            
            # Now start the capture thread (safe -- mapping already read frame 1)
            self.cap.start_capture()
            
            # Initialize fish detector
            self.initialize_fish_detector()

            # Initialize timing for exports
            self.start_time = datetime.now()
            try:
                self.playback_fps = float(info.get('framerate', self.target_fps))
            except Exception:
                self.playback_fps = self.target_fps
            
            # Start first pass
            self.current_pass = 1
            self.pass_completed = False
            print(f"Pass #{self.current_pass} started - statistics will export when complete")
            
            # Create windows
            if self.show_raw_window:
                cv2.namedWindow('Raw ARIS Data', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Fan Mapping', cv2.WINDOW_NORMAL)
            
            # Set window sizes
            if self.show_raw_window:
                cv2.resizeWindow('Raw ARIS Data', 600, 450)
            cv2.resizeWindow('Fan Mapping', 600, 450)
            
            # Position windows
            if self.show_raw_window:
                cv2.moveWindow('Raw ARIS Data', 100, 100)
                cv2.moveWindow('Fan Mapping', 720, 100)
            else:
                cv2.moveWindow('Fan Mapping', 100, 100)
            
            print(f"\nCONTROLS:")
            print(f"   SPACE: Play/Pause")
            print(f"   ESC: Exit and auto-export CSVs")
            print(f"   R: Reset to first frame")
            print(f"   D: Toggle overlay ON/OFF (detection keeps running)")
            print(f"   C: Clear fish detector (recompute background)")
            print(f"   E: Export statistics NOW (exports CSVs immediately)")
            print(f"   T: Toggle saved tracks overlay (shows exported results)")
            
            if self.enable_fish_detection:
                print(f"\nAuto-Export: Statistics will be saved to exports/statistics/ when playback completes")
                print(f"   • Global info: frame-by-frame fish counts")
                print(f"   • Tracking info: position, velocity, length per fish")
                print(f"   • Validation info: time-windowed statistics")
            
            # Main playback loop
            frame_count = 0
            pass_frame_count = 0  # Track frames per pass
            pass_detection_count = 0  # Track detections per pass
            start_time = time.time()
            pass_start_time = time.time()  # Track per-pass timing
            target_frame_time = 1.0 / self.target_fps
            
            while True:
                frame_start_time = time.time()
                
                if self.playing:
                    # Get frame
                    ret, raw_frame, frame_num = self.cap.get_frame()
                    
                    if not ret or frame_num > self.last_frame:
                        # Only process loop logic once per loop
                        if not self.loop_just_reset:
                            # Mark pass as complete and export
                            self.pass_completed = True
                            
                            # Save complete pass stats before resetting
                            pass_time = time.time() - pass_start_time
                            pass_fps = pass_frame_count / pass_time if pass_time > 0 else 0
                            self.last_complete_pass_stats = {
                                'pass_number': self.current_pass,
                                'frames': pass_frame_count,
                                'detections': pass_detection_count,
                                'fps': pass_fps,
                                'time': pass_time
                            }
                            
                            if self.enable_fish_detection and self.fish_detector is not None and len(self.frame_to_tracks) > 0:
                                print(f"\nPass #{self.current_pass} COMPLETE (frames {self.first_frame}-{self.last_frame})")
                                print(f"Exporting statistics from Pass #{self.current_pass}...")
                                try:
                                    self.export_analysis_results()
                                    self.last_exported_pass = self.current_pass
                                    print(f"Pass #{self.current_pass} exported successfully!\n")
                                except Exception as e:
                                    print(f"Export failed: {e}\n")
                            
                            # Check if this was a long video (auto-exit after one pass)
                            total_frames_processed = self.last_frame - self.first_frame + 1
                            if total_frames_processed >= 5000:
                                print(f"Large video processing complete ({total_frames_processed} frames)")
                                print(f"   Exiting automatically to prevent memory issues")
                                print(f"   (Videos with 5000+ frames exit after one pass)")
                                break  # Exit instead of looping
                            
                            # Prepare for next loop (smaller videos only)
                            print("📹 Looping to start...")
                            self.cap.current_frame_num = self.first_frame
                            
                            # Reset for next pass
                            if self.enable_fish_detection and self.fish_detector is not None:
                                self.fish_detector.tracks = []
                                self.fish_detector.next_track_id = 1
                                self.frame_to_tracks.clear()
                                self.current_pass += 1
                                self.pass_completed = False
                                # Reset per-pass counters
                                pass_frame_count = 0
                                pass_detection_count = 0
                                pass_start_time = time.time()
                                print(f"Pass #{self.current_pass} started (background model preserved)")
                            
                            self.loop_just_reset = True
                        continue
                    
                    if frame_num < self.first_frame:
                        continue
                    
                    self.current_frame = frame_num
                    
                    # Clear the loop reset flag once we start getting frames again
                    if self.loop_just_reset:
                        self.loop_just_reset = False
                    
                    # Auto-reset when loop wraps (frame number decreased)
                    if hasattr(self, '_last_frame_num') and frame_num < self._last_frame_num:
                        if self.enable_fish_detection and self.fish_detector is not None:
                            print("Loop detected: auto-resetting detector and stats")
                            self.fish_detector.reset()
                            self.frame_to_tracks.clear()
                    self._last_frame_num = frame_num

                    # Create fan image
                    fan_frame = self.create_fan_image_fast(raw_frame)
                    
                    # Apply colormap
                    raw_colored = self.apply_colormap_opencv(raw_frame, self.bluebar_cmap)
                    fan_colored = self.apply_colormap_opencv(fan_frame, self.bluebar_cmap)
                    
                    # Fish detection
                    fish_detections = []
                    if self.enable_fish_detection and self.fish_detector is not None:
                        try:
                            # Pass frame index so track histories are time-aligned
                            fish_detections = self.fish_detector.detect_fish(fan_frame, frame_index=frame_num)
                            if fish_detections and self.show_detection_overlay:
                                fan_colored = np.ascontiguousarray(fan_colored, dtype=np.uint8)
                                fan_colored = self.fish_detector.draw_detections(fan_colored, fish_detections)
                        except Exception as e:
                            print(f"Detection error: {e}")
                            self.enable_fish_detection = False

                    # Accumulate mapping for export (tracks updated this frame)
                    if self.enable_fish_detection and self.fish_detector is not None:
                        try:
                            for track in self.fish_detector.tracks:
                                if track.frame_history and track.frame_history[-1] == frame_num:
                                    obs_idx = len(track.position_history) - 1
                                    self.frame_to_tracks[frame_num].append((track.id, obs_idx))
                            # Track per-pass detection count
                            pass_detection_count += len(fish_detections)
                        except Exception:
                            pass
                    
                    # Increment per-pass frame counter
                    pass_frame_count += 1
                    
                    # Draw saved tracks overlay if enabled
                    if self.show_detection_overlay and self.show_saved_tracks and self.saved_tracks is not None:
                        fan_colored = self.draw_saved_tracks(fan_colored, frame_num)
                    
                    # Ensure arrays are contiguous
                    raw_colored = np.ascontiguousarray(raw_colored, dtype=np.uint8)
                    fan_colored = np.ascontiguousarray(fan_colored, dtype=np.uint8)
                    
                    # Add frame info
                    current_time = time.time()
                    if self.last_frame_time is not None:
                        frame_time = current_time - self.last_frame_time
                        fps = 1.0 / frame_time if frame_time > 0 else 0
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        # Frame and FPS text
                        frame_text = f'Frame: {frame_num}'
                        fps_text = f'FPS: {fps:.1f}'
                        
                        # Fish count text with pattern status
                        if self.enable_fish_detection and self.fish_detector is not None:
                            fish_text = f'Fish: {len(fish_detections)}'
                            # Check if pattern is active
                            pattern_active = any(d.get('pattern_active', False) for d in fish_detections)
                            if pattern_active:
                                fish_text += ' | Pattern: ON'
                            if not self.show_detection_overlay:
                                fish_text += ' | Overlay: OFF'
                        else:
                            fish_text = 'Detection: OFF'
                        
                        # Add text to both images
                        for img in [raw_colored, fan_colored]:
                            # Background rectangles and text
                            cv2.rectangle(img, (5, 5), (150, 75), (0, 0, 0), -1)
                            cv2.putText(img, frame_text, (10, 25), font, font_scale, (0, 255, 0), thickness)
                            cv2.putText(img, fps_text, (10, 45), font, font_scale, (0, 255, 0), thickness)
                            cv2.putText(img, fish_text, (10, 65), font, font_scale, (0, 255, 0), thickness)
                    
                    self.last_frame_time = current_time
                    
                    # Display frames
                    if self.show_raw_window:
                        cv2.imshow('Raw ARIS Data', raw_colored)
                    cv2.imshow('Fan Mapping', fan_colored)
                    
                    frame_count += 1
                    
                    # Print occasional statistics
                    if frame_count % 60 == 0:
                        elapsed = current_time - start_time
                        avg_fps = frame_count / elapsed
                        print(f"Frame {frame_num}: {avg_fps:.1f} avg FPS, {len(fish_detections)} fish detected")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE - Play/Pause
                    self.playing = not self.playing
                    print(f"{'Playing' if self.playing else 'Paused'}")
                elif key == ord('r'):  # R - Reset
                    self.cap.current_frame_num = self.first_frame
                    if self.enable_fish_detection and self.fish_detector is not None:
                        self.fish_detector.reset()
                    print("Reset to first frame")
                elif key == ord('d'):  # D - Toggle overlay visibility only
                    self.show_detection_overlay = not self.show_detection_overlay
                    print(f"Detection overlay: {'ON' if self.show_detection_overlay else 'OFF'} (tracking still running)")
                elif key == ord('c'):  # C - Clear fish detector
                    if self.enable_fish_detection and self.fish_detector is not None:
                        self.fish_detector.reset()
                        print("Fish detector cleared")
                elif key == ord('t'):  # T - Toggle saved tracks overlay
                    if self.saved_tracks is not None:
                        self.show_saved_tracks = not self.show_saved_tracks
                        print(f"Saved tracks overlay: {'ON' if self.show_saved_tracks else 'OFF'}")
                        if self.show_saved_tracks:
                            try:
                                from os.path import relpath
                                path_str = relpath(getattr(self, 'saved_tracks_path', '')) if hasattr(self, 'saved_tracks_path') else ''
                                if path_str:
                                    print(f"   Using file: {path_str} | Frames: {getattr(self,'saved_frame_min',1)}-{getattr(self,'saved_frame_max',1)}")
                            except Exception:
                                pass
                    else:
                        print("No saved tracks available. Run offline analysis first (Option R).")
                elif key == ord('e'):  # E - Export statistics from current playback
                    if self.enable_fish_detection and len(self.frame_to_tracks) > 0:
                        print(f"\nWarning: Manual export requested for Pass #{self.current_pass}")
                        if not self.pass_completed:
                            print(f"   Pass is INCOMPLETE ({len(self.frame_to_tracks)} frames so far)")
                            confirm = input("   Export incomplete data? (y/N): ").strip().lower()
                            if confirm != 'y':
                                print("   Export cancelled")
                                continue
                        try:
                            self.export_analysis_results()
                            print(f"Pass #{self.current_pass} data exported\n")
                        except Exception as e:
                            print(f"Export failed: {e}\n")
                    else:
                        print("Warning: No tracking data to export yet")
                # FPS limiting
                if self.playing:
                    frame_end_time = time.time()
                    frame_duration = frame_end_time - frame_start_time
                    sleep_time = target_frame_time - frame_duration
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            # Final statistics - use last COMPLETE pass if current pass is incomplete
            if self.pass_completed or self.last_complete_pass_stats is None:
                # Current pass completed OR no complete passes yet - show current pass
                pass_time = time.time() - pass_start_time
                pass_fps = pass_frame_count / pass_time if pass_time > 0 else 0
                stats_pass = self.current_pass
                stats_frames = pass_frame_count
                stats_detections = pass_detection_count
                stats_fps = pass_fps
                stats_time = pass_time
                is_complete = self.pass_completed
            else:
                # Current pass incomplete - show last complete pass stats
                stats = self.last_complete_pass_stats
                stats_pass = stats['pass_number']
                stats_frames = stats['frames']
                stats_detections = stats['detections']
                stats_fps = stats['fps']
                stats_time = stats['time']
                is_complete = True
            
            status = "COMPLETE" if is_complete else "INCOMPLETE"
            print(f"\nFinal Performance (Last {status} Pass #{stats_pass}):")
            print(f"   Pass frames: {stats_frames}")
            print(f"   Pass FPS: {stats_fps:.1f}")
            print(f"   Total playback time: {stats_time:.1f}s")
            
            if self.enable_fish_detection and self.fish_detector is not None:
                print(f"Fish Detection Stats (Pass #{stats_pass}):")
                print(f"   Pass detections: {stats_detections}")
                print(f"   Active tracks at exit: {len(self.fish_detector.tracks)}")

            # Only export if mid-pass when ESC pressed (incomplete pass)
            # Complete passes already exported automatically
            try:
                if self.enable_fish_detection and self.fish_detector is not None and len(self.frame_to_tracks) > 0:
                    if not self.pass_completed:
                        print(f"\nWarning: Pass #{self.current_pass} INCOMPLETE - only {len(self.frame_to_tracks)} frames processed")
                        print("Tip: Let playback complete a full pass for accurate statistics")
                        print("   (Incomplete pass data NOT exported)")
                    elif self.current_pass > self.last_exported_pass:
                        # Final pass wasn't exported yet (shouldn't happen but safety check)
                        print(f"\nExporting final pass #{self.current_pass}...")
                        self.export_analysis_results()
            except Exception as e:
                print(f"Warning: Export check error: {e}")
        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Simple movie player closed!")

    def export_analysis_results(self, output_dir="exports/statistics"):
        """Export MATLAB-style CSV statistics from the current playback state."""
        if self.fish_detector is None:
            print("No detector active - nothing to export")
            return
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        base_path = os.path.join(output_dir, base_name)

        # Determine timing
        fps = self.playback_fps or self.target_fps or 10.0

        # Choose reliable tracks using config threshold (active + archived)
        all_detector_tracks = list(self.fish_detector.tracks) + list(getattr(self.fish_detector, 'all_tracks', []))
        
        # Merge tracks with duplicate IDs (can happen with ID reuse)
        tracks_by_id = {}
        for t in all_detector_tracks:
            if t.is_reliable():
                if t.id not in tracks_by_id:
                    tracks_by_id[t.id] = t
                else:
                    # Merge histories from duplicate ID (append observations)
                    existing = tracks_by_id[t.id]
                    existing.position_history.extend(t.position_history)
                    existing.frame_history.extend(t.frame_history)
                    existing.bbox_history.extend(t.bbox_history)
                    existing.major_axis_history.extend(t.major_axis_history)
                    existing.minor_axis_history.extend(t.minor_axis_history)
                    existing.dist_from_mangrove_history.extend(t.dist_from_mangrove_history)
        
        reliable_tracks = sorted(tracks_by_id.values(), key=lambda t: t.id)
        reliable_ids = set(tracks_by_id.keys())

        # Normalize observation ordering per track by frame index after merge.
        for track in reliable_tracks:
            n = len(track.position_history)
            if n == 0:
                continue
            packed = []
            for i in range(n):
                frame_idx = track.frame_history[i] if i < len(track.frame_history) else (i + 1)
                packed.append((
                    int(frame_idx),
                    track.position_history[i],
                    track.bbox_history[i] if i < len(track.bbox_history) else track.bbox,
                    float(track.major_axis_history[i]) if i < len(track.major_axis_history) else 0.0,
                    float(track.minor_axis_history[i]) if i < len(track.minor_axis_history) else 0.0,
                    float(track.orientation_history[i]) if i < len(track.orientation_history) else 0.0,
                    float(track.eccentricity_history[i]) if i < len(track.eccentricity_history) else 0.0,
                    float(track.dist_from_mangrove_history[i]) if i < len(track.dist_from_mangrove_history) else 0.0
                ))
            packed.sort(key=lambda x: x[0])
            track.frame_history = [p[0] for p in packed]
            track.position_history = [p[1] for p in packed]
            track.bbox_history = [p[2] for p in packed]
            track.major_axis_history = [p[3] for p in packed]
            track.minor_axis_history = [p[4] for p in packed]
            track.orientation_history = [p[5] for p in packed]
            track.eccentricity_history = [p[6] for p in packed]
            track.dist_from_mangrove_history = [p[7] for p in packed]
        
        # Get mangrove area for global info (MATLAB equivalent)
        # Use mapscale for proper area calculation: area_m2 = pixels * hs * ws
        mangrove_area_cm2 = 0.0
        if (self.fish_detector.mangrove_mapper and 
            self.fish_detector.mangrove_mapper.is_active()):
            if self.mapscale:
                hs, ws, _, _ = self.mapscale
                pix_area_m2 = hs * ws  # Area of one pixel in m²
                pix_scale = (hs + ws) / 2  # Average for compatibility
            else:
                pix_scale = getattr(self.fish_detector.config, 'PIX_SCALE', 0.001)
                pix_area_m2 = pix_scale ** 2
            mangrove_area_cm2 = self.fish_detector.mangrove_mapper.get_mangrove_area_cm2(pix_scale)

        # Frame range exported should match processed playback range.
        if hasattr(self, 'first_frame') and hasattr(self, 'last_frame'):
            first_frame = self.first_frame
            last_frame = self.last_frame
        else:
            all_frame_indices = sorted(set(self.frame_to_tracks.keys()))
            if not all_frame_indices:
                return
            first_frame = min(all_frame_indices)
            last_frame = max(all_frame_indices)

        # Build per-frame time lookup from ARIS headers when available.
        frame_times = {}
        has_aris_timestamps = False
        for frame_idx in range(first_frame, last_frame + 1):
            dt_val = None
            try:
                frame_info = self.cap.get_frame_new(frame_idx)
                header = frame_info.get('frame_header', {})
                sonar_ts = header.get('sonartimestamp', 0)
                if sonar_ts and sonar_ts > 0:
                    dt_val = datetime.utcfromtimestamp(float(sonar_ts) * 1e-6)
                    has_aris_timestamps = True
            except Exception:
                dt_val = None
            frame_times[frame_idx] = dt_val

        if has_aris_timestamps:
            # Fill gaps by stepping with nominal FPS from nearest known times.
            step_dt = timedelta(seconds=1.0 / max(fps, 1e-6))
            known = [k for k, v in frame_times.items() if v is not None]
            if known:
                first_known = min(known)
                for idx in range(first_known - 1, first_frame - 1, -1):
                    frame_times[idx] = frame_times[idx + 1] - step_dt
                for idx in range(first_known + 1, last_frame + 1):
                    if frame_times[idx] is None:
                        frame_times[idx] = frame_times[idx - 1] + step_dt
        else:
            start_time = self.start_time or datetime.now()
            for frame_idx in range(first_frame, last_frame + 1):
                frame_times[frame_idx] = start_time + timedelta(seconds=(frame_idx - 1) / max(fps, 1e-6))

        # 1. Global info
        if self.mapscale:
            hs, ws, i0, j0 = self.mapscale  # [height_scale, width_scale, y_origin, x_origin]
            pix_scale_avg = (hs + ws) / 2
        else:
            hs = ws = 0.001
            i0 = j0 = 0
            pix_scale_avg = 0.001

        prey_predator_cm = float(getattr(self.fish_detector.config, 'PREY_PREDATOR_THRESHOLD_CM', 5.0))
        school_min_fishes = int(getattr(self.fish_detector.config, 'SCHOOL_MIN_FISHES', 10))

        def get_frame_observations(frame_idx):
            obs = []
            for tid, obs_idx in self.frame_to_tracks.get(frame_idx, []):
                if tid not in reliable_ids:
                    continue
                track = tracks_by_id.get(tid)
                if track is None:
                    continue
                if 0 <= obs_idx < len(track.position_history):
                    obs.append((track, obs_idx))
            return obs

        def mean_nearest_neighbor_distance(points_xy):
            if points_xy.shape[0] < 2:
                return None
            dif = points_xy[:, None, :] - points_xy[None, :, :]
            dists = np.sqrt(np.sum(dif * dif, axis=2))
            np.fill_diagonal(dists, np.inf)
            return float(np.mean(np.min(dists, axis=1)))

        global_file = f"{base_path}_global_info.csv"
        with open(global_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Datafile Name', 'Date (yyyy-mm-dd)', 'Time (HH:MM:SS.FFF)',
                             'Frame Index', 'Number of Fish', 'School Detected',
                             'Area Of Mangrove Habitat (cm²)', 'Number of Predators', 'Number of Preys'])
            
            for frame_idx in range(first_frame, last_frame + 1):
                frame_obs = get_frame_observations(frame_idx)
                num_fish = len(frame_obs)
                num_preys = 0
                num_predators = 0
                school_detected = False

                if num_fish > 0:
                    lengths_px = np.array(
                        [float(track.major_axis_history[oi]) for track, oi in frame_obs],
                        dtype=np.float64
                    )
                    lengths_cm = lengths_px * pix_scale_avg * 100.0
                    num_preys = int(np.sum(lengths_cm < prey_predator_cm))
                    num_predators = int(num_fish - num_preys)

                    if num_fish > school_min_fishes and num_fish >= 2:
                        pts = np.array([track.position_history[oi] for track, oi in frame_obs], dtype=np.float64)
                        mean_nn = mean_nearest_neighbor_distance(pts)
                        mean_major_px = float(np.mean(lengths_px)) if lengths_px.size > 0 else 0.0
                        if mean_nn is not None and mean_major_px > 0:
                            school_detected = bool(mean_nn < (0.5 * mean_major_px))

                frame_time = frame_times[frame_idx]
                
                writer.writerow([
                    os.path.basename(self.filename),
                    frame_time.strftime('%Y-%m-%d'),
                    frame_time.strftime('%H:%M:%S.%f')[:-3],
                    frame_idx,
                    num_fish,
                    school_detected,
                    int(mangrove_area_cm2),
                    num_predators,
                    num_preys
                ])

        # 2. Tracking info
        tracking_file = f"{base_path}_tracking_info.csv"
        with open(tracking_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Datafile Name', 'Frame Index', 'Fish ID', 'X Position (m)', 'Y Position (m)',
                             'X_px', 'Y_px', 'Velocity (cm/s)', 'Velocity Direction (degrees)',
                             'Acceleration (cm/s^2)', 'Fish Length (cm)', 'Fish Width (cm)',
                             'Orientation (degrees)', 'Eccentricity',
                             'Tail Beat Rate (s)', 'Distance from Mangrove (cm)'])
            for track in reliable_tracks:
                n = len(track.position_history)
                frame_hist = [track.frame_history[i] if i < len(track.frame_history) else (i + 1) for i in range(n)]
                fish_lengths_cm = [
                    (float(track.major_axis_history[i]) if i < len(track.major_axis_history) else 0.0) * pix_scale_avg * 100.0
                    for i in range(n)
                ]
                beat_rate = np.zeros(n, dtype=np.float64)
                if n > 2:
                    peaks = np.array([], dtype=np.int64)
                    try:
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(np.asarray(fish_lengths_cm, dtype=np.float64))
                    except Exception:
                        # Fallback local-maximum finder if scipy peak detection is unavailable.
                        peaks_list = []
                        for p in range(1, n - 1):
                            if fish_lengths_cm[p] > fish_lengths_cm[p - 1] and fish_lengths_cm[p] > fish_lengths_cm[p + 1]:
                                peaks_list.append(p)
                        peaks = np.asarray(peaks_list, dtype=np.int64)
                    if peaks.size > 0:
                        idx1 = np.concatenate(([0], peaks))
                        for p in range(1, len(idx1)):
                            cur = int(idx1[p])
                            prev = int(idx1[p - 1])
                            frame_gap = max(abs(frame_hist[cur] - frame_hist[prev]), 1)
                            beat_rate[cur] = (frame_gap * 2.0) / max(fps, 1e-6)

                for i in range(n):
                    frame_idx = frame_hist[i]
                    x_px = float(track.position_history[i][0])
                    y_px = float(track.position_history[i][1])
                    
                    # MATLAB-style coordinate conversion (relative to transducer)
                    # X_m = (pixel_x - j0) * ws  (centered on image)
                    # Y_m = (i0 - pixel_y) * hs  (distance from transducer, Y increases toward transducer)
                    if self.mapscale:
                        x_m = (x_px - j0) * ws
                        y_m = (i0 - y_px) * hs
                    else:
                        x_m = x_px * 0.001
                        y_m = y_px * 0.001
                    
                    # Kinematics using proper scale and actual frame gaps (MATLAB parity)
                    vel = 0.0
                    ang = 0.0
                    acc = 0.0
                    if i > 0:
                        # Calculate displacement in meters
                        x_px_prev = float(track.position_history[i-1][0])
                        y_px_prev = float(track.position_history[i-1][1])
                        
                        if self.mapscale:
                            x_m_prev = (x_px_prev - j0) * ws
                            y_m_prev = (i0 - y_px_prev) * hs
                        else:
                            x_m_prev = x_px_prev * 0.001
                            y_m_prev = y_px_prev * 0.001
                        
                        dx_m = x_m - x_m_prev
                        dy_m = y_m - y_m_prev
                        dist_m = (dx_m**2 + dy_m**2) ** 0.5
                        dist_cm = dist_m * 100  # meters to cm
                        
                        # Prefer frame-header timestamps; fallback to frame gap/FPS.
                        frame_gap = 1
                        if i < len(frame_hist) and (i - 1) < len(frame_hist):
                            frame_gap = max(abs(frame_hist[i] - frame_hist[i - 1]), 1)
                        dt = frame_gap / max(fps, 1e-6)
                        t_cur = frame_times.get(frame_hist[i])
                        t_prev = frame_times.get(frame_hist[i - 1])
                        if t_cur is not None and t_prev is not None:
                            dt_sec = (t_cur - t_prev).total_seconds()
                            if dt_sec > 0:
                                dt = dt_sec
                        vel = dist_cm / dt
                        if dist_m > 0:
                            ang = np.degrees(np.arctan2(dy_m, dx_m))
                        
                        if i > 1:
                            x_px_prev2 = float(track.position_history[i-2][0])
                            y_px_prev2 = float(track.position_history[i-2][1])
                            if self.mapscale:
                                x_m_prev2 = (x_px_prev2 - j0) * ws
                                y_m_prev2 = (i0 - y_px_prev2) * hs
                            else:
                                x_m_prev2 = x_px_prev2 * 0.001
                                y_m_prev2 = y_px_prev2 * 0.001
                            
                            dx_m_prev = x_m_prev - x_m_prev2
                            dy_m_prev = y_m_prev - y_m_prev2
                            dist_prev_cm = ((dx_m_prev**2 + dy_m_prev**2) ** 0.5) * 100
                            frame_gap_prev = 1
                            if (i - 1) < len(frame_hist) and (i - 2) < len(frame_hist):
                                frame_gap_prev = max(abs(frame_hist[i - 1] - frame_hist[i - 2]), 1)
                            dt_prev = frame_gap_prev / max(fps, 1e-6)
                            t_prev_cur = frame_times.get(frame_hist[i - 1])
                            t_prev_prev = frame_times.get(frame_hist[i - 2])
                            if t_prev_cur is not None and t_prev_prev is not None:
                                dt_prev_sec = (t_prev_cur - t_prev_prev).total_seconds()
                                if dt_prev_sec > 0:
                                    dt_prev = dt_prev_sec
                            vel_prev = dist_prev_cm / dt_prev
                            acc = (vel - vel_prev) / dt
                    
                    # Fish length and width using ellipse-fit major/minor axis
                    length_cm = 0.0
                    width_cm = 0.0
                    if i < len(track.major_axis_history):
                        length_px = float(track.major_axis_history[i])
                        length_cm = length_px * pix_scale_avg * 100  # pixels → meters → cm
                    if i < len(track.minor_axis_history):
                        width_px = float(track.minor_axis_history[i])
                        width_cm = width_px * pix_scale_avg * 100

                    # Orientation and eccentricity from ellipse fit
                    orientation_deg = 0.0
                    if i < len(getattr(track, 'orientation_history', [])):
                        orientation_deg = float(track.orientation_history[i])
                    eccentricity = 0.0
                    if i < len(getattr(track, 'eccentricity_history', [])):
                        eccentricity = float(track.eccentricity_history[i])

                    # Mangrove distance using proper pixel scale
                    distance_cm = 0.0
                    if i < len(track.dist_from_mangrove_history):
                        distance_px = track.dist_from_mangrove_history[i]
                        distance_cm = distance_px * pix_scale_avg * 100  # pixels → meters → cm
                    
                    writer.writerow([
                        os.path.basename(self.filename),
                        frame_idx,
                        track.id,
                        round(x_m, 6),
                        round(y_m, 6),
                        int(round(x_px)),
                        int(round(y_px)),
                        round(vel, 2),
                        round(ang, 1),
                        round(acc, 2),
                        round(length_cm, 1),
                        round(width_cm, 1),
                        round(orientation_deg, 1),
                        round(eccentricity, 3),
                        round(float(beat_rate[i]), 3),
                        round(distance_cm, 1)
                    ])

        # 3. Validation info (simple 5-minute buckets)
        validation_ranges_file = f"{base_path}_validation_info_ranges.csv"
        validation_max_file = f"{base_path}_validation_info_max_fishes_per_range.csv"

        # group frames by configurable period
        validation_period = float(getattr(self.fish_detector.config, 'VALIDATION_MAX_FISH_PERIOD_S', 300.0))
        validation_filter = getattr(self.fish_detector, 'validation_filter', None)
        frame_counts = {}
        for frame_idx in range(first_frame, last_frame + 1):
            frame_obs = get_frame_observations(frame_idx)
            count = 0
            for track, obs_idx in frame_obs:
                x_px = float(track.position_history[obs_idx][0])
                y_px = float(track.position_history[obs_idx][1])
                if validation_filter and validation_filter.is_active():
                    if not validation_filter.is_in_validation_area(x_px, y_px):
                        continue
                count += 1
            frame_counts[frame_idx] = count

        ranges = []
        current_start_t = None
        current_frames = []
        for frame_idx in range(first_frame, last_frame + 1):
            t_sec = (frame_times[frame_idx] - frame_times[first_frame]).total_seconds()
            if current_start_t is None:
                current_start_t = t_sec
            if (t_sec - current_start_t) >= validation_period:
                if current_frames:
                    ranges.append((current_frames[0], current_frames[-1], current_start_t, t_sec))
                current_start_t = t_sec
                current_frames = [frame_idx]
            else:
                current_frames.append(frame_idx)
        if current_frames:
            end_t = (current_frames[-1] - 1) / max(fps, 1e-6)
            ranges.append((current_frames[0], current_frames[-1], current_start_t, end_t))

        max_fish_rows = []
        with open(validation_ranges_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Datafile Name', 'Range ID', 'Date Start (yyyy-mm-dd)', 'Time Start (HH:MM:SS.FFF)',
                             'Date End (yyyy-mm-dd)', 'Time End (HH:MM:SS.FFF)', 'Elapsed time (seconds)',
                             'Frame Start', 'Frame End', 'Maximum Number of Fishes'])
            for idx, (f_start, f_end, t_start, t_end) in enumerate(ranges, 1):
                max_count = 0
                max_frames = []
                for fidx in range(f_start, f_end + 1):
                    count = frame_counts.get(fidx, 0)
                    if count > max_count:
                        max_count = count
                        max_frames = [fidx]
                    elif count == max_count:
                        max_frames.append(fidx)
                start_dt = frame_times[f_start]
                end_dt = frame_times[f_end]
                writer.writerow([
                    os.path.basename(self.filename), idx,
                    start_dt.strftime('%Y-%m-%d'), start_dt.strftime('%H:%M:%S.%f')[:-3],
                    end_dt.strftime('%Y-%m-%d'), end_dt.strftime('%H:%M:%S.%f')[:-3],
                    round(max(0.0, t_end - t_start), 1), f_start, f_end, max_count
                ])
                for mf in max_frames:
                    max_fish_rows.append((idx, mf, max_count, frame_times[mf]))

        with open(validation_max_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Datafile Name', 'Range ID', 'Frame Index', 'Date (yyyy-mm-dd)', 'Time (HH:MM:SS.FFF)', 'Maximum Number of Fishes'])
            for (rid, fidx, mcount, ftime) in max_fish_rows:
                writer.writerow([
                    os.path.basename(self.filename), rid, fidx,
                    ftime.strftime('%Y-%m-%d'), ftime.strftime('%H:%M:%S.%f')[:-3], mcount
                ])

        # 4. Summary statistics
        summary_file = f"{base_path}_summary.csv"
        total_frames = self.last_frame - self.first_frame + 1  # Frames actually processed
        frames_with_fish = len(self.frame_to_tracks)
        unique_fish = len(reliable_ids)
        total_observations = sum(len(t.position_history) for t in reliable_tracks)
        max_simultaneous = max((frame_counts.get(fidx, 0) for fidx in range(first_frame, last_frame + 1)), default=0)
        max_fish_id = max(reliable_ids) if reliable_ids else 0
        
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Datafile Name', os.path.basename(self.filename)])
            writer.writerow(['Total Frames Processed', total_frames])
            writer.writerow(['Frames with Fish Detected', frames_with_fish])
            writer.writerow(['Unique Fish Tracked', unique_fish])
            writer.writerow(['Highest Fish ID Assigned', max_fish_id])
            writer.writerow(['Total Tracking Observations', total_observations])
            writer.writerow(['Max Simultaneous Fish', max_simultaneous])
            writer.writerow(['Average Observations per Fish', round(total_observations / max(unique_fish, 1), 1)])
            writer.writerow(['Config: MIN_TRACK_HITS', getattr(self.fish_detector.config, 'MIN_TRACK_HITS', 'N/A')])
            writer.writerow(['Config: AGE_THRESHOLD', getattr(self.fish_detector.config, 'AGE_THRESHOLD', 'N/A')])
            writer.writerow(['Config: MIN_VISIBILITY_RATIO', getattr(self.fish_detector.config, 'MIN_VISIBILITY_RATIO', 'N/A')])
            writer.writerow(['Config: MAX_INVISIBLE_COUNT', getattr(self.fish_detector.config, 'MAX_INVISIBLE_COUNT', 'N/A')])
            writer.writerow(['Config: PREY_PREDATOR_THRESHOLD_CM', getattr(self.fish_detector.config, 'PREY_PREDATOR_THRESHOLD_CM', 'N/A')])
            writer.writerow(['Config: SCHOOL_MIN_FISHES', getattr(self.fish_detector.config, 'SCHOOL_MIN_FISHES', 'N/A')])
            writer.writerow(['Config: VALIDATION_MAX_FISH_PERIOD_S', getattr(self.fish_detector.config, 'VALIDATION_MAX_FISH_PERIOD_S', 'N/A')])
            writer.writerow(['Timing Source', 'ARIS headers' if has_aris_timestamps else 'Playback start + FPS fallback'])

        from os.path import relpath
        print(f"   Exported: {relpath(global_file)}")
        print(f"   Exported: {relpath(tracking_file)}")
        print(f"   Exported: {relpath(validation_ranges_file)}")
        print(f"   Exported: {relpath(validation_max_file)}")
        print(f"   Exported: {relpath(summary_file)}")

        # Reload saved tracks so overlay (T) reflects the just-exported results
        try:
            self.load_saved_tracks()
            print("   Saved tracks overlay updated (press 'T' to toggle)")
        except Exception:
            pass


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = '../ARIS Data/DH1112_2015-11-08_115514.aris'
    
    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        return
    
    print("Simple ARIS Player Demo")
    print("=" * 40)
    
    player = OpenCVPlayer(filename, enable_fish_detection=True)
    player.run_player(first=1, last=200)


if __name__ == "__main__":
    main()
