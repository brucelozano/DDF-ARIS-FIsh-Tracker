#!/usr/bin/env python3
"""
ARIS/DDF Utilities - Export and batch processing functions
MATLAB equivalents: arisreader_generateavi, convert_sonar_to_mat
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from opencv_player import CapMultiThreading
from aris_sonar_processing import FinalCleanMapper

SUPPORTED_SONAR_EXTENSIONS = (".aris", ".ddf")


def discover_sonar_files(input_dir, file_pattern=None):
    """Discover sonar files (.aris/.ddf) in a single directory."""
    directory = Path(input_dir)
    if file_pattern:
        return sorted(directory.glob(file_pattern))

    discovered = []
    for extension in SUPPORTED_SONAR_EXTENSIONS:
        discovered.extend(directory.glob(f"*{extension}"))

    unique_files = {path.resolve(): path for path in discovered}
    return sorted(unique_files.values())


class ARISExporter:
    """Export ARIS data to various formats"""
    
    def __init__(self, filename):
        self.filename = filename
        self.cap = CapMultiThreading(filename)
        self.mapper = FinalCleanMapper()
        self.bluebar_cmap = self.load_matlab_bluebar_colormap()
        
    def read_frame_direct(self, frame_number):
        """
        Direct frame reading that bypasses CapMultiThreading issues
        Returns clean MATLAB-equivalent frame data
        """
        import numpy as np
        
        if frame_number < 1 or frame_number > self.cap.file_info['numframes']:
            raise ValueError(
                f"Frame number {frame_number} out of range [1, {self.cap.file_info['numframes']}]"
            )

        # Read file info from the existing CapMultiThreading instance
        numbeams = self.cap.file_info['numbeams']
        sampleperchannel = self.cap.file_info['sampleperchannel']
        reverse = self.cap.file_info['reverse']

        data_length = numbeams * sampleperchannel
        layout = getattr(self.cap, 'layout', None)
        if layout is not None:
            frame_data_pos = layout.frame_data_offset(frame_number)
        else:
            file_header_len = int(self.cap.file_info.get('fileheaderlength', 1024))
            frame_header_len = int(self.cap.file_info.get('frameheaderlength', 1024))
            frame_data_pos = file_header_len + (frame_number - 1) * (frame_header_len + data_length) + frame_header_len
        
        # Direct file reading (no multi-threading)
        with open(self.filename, 'rb') as fid:
            # Seek to frame data position
            fid.seek(frame_data_pos)
            
            # Read frame data
            raw_bytes = fid.read(data_length)
            
            if len(raw_bytes) != data_length:
                raise ValueError(f"Could not read complete frame {frame_number}")
            
            # Process exactly like MATLAB
            raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            reshaped = raw_array.reshape(numbeams, sampleperchannel, order='F')
            
            # Apply orientation (MATLAB equivalent)
            if reverse == 0:
                frame_data = np.fliplr(reshaped.T)  # Transpose and flip
            else:
                frame_data = reshaped.T  # Just transpose
                
        return frame_data
    
    def load_matlab_bluebar_colormap(self):
        """Load MATLAB bluebar colormap"""
        try:
            from scipy.io import loadmat
            mat_data = loadmat('bluebar.mat')
            bluebar = mat_data['bluebar']
            # Convert to 8-bit RGB
            bluebar_8bit = (bluebar * 255).astype(np.uint8)
            return bluebar_8bit
        except Exception as e:
            print(f"Warning: Could not load bluebar.mat: {e}")
            return None
    
    def apply_colormap_to_grayscale(self, gray_image):
        """Apply bluebar colormap to grayscale image"""
        if self.bluebar_cmap is not None:
            # Use the colormap to convert grayscale to color
            colored_image = self.bluebar_cmap[gray_image]
            # Convert RGB to BGR for OpenCV
            return colored_image[:, :, [2, 1, 0]]
        else:
            # Fallback to grayscale
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
    def generate_avi(self, output_file, display_type='cartesian', frame_range=None, fps=30):
        """
        MATLAB equivalent: arisreader_generateavi(filein, fileout, type, index)
        Generate AVI video from ARIS frames
        
        Args:
            output_file: Output AVI filename
            display_type: 'polar', 'cartesian', or 'raw'
            frame_range: [start, end] frame numbers (1-based), defaults to full file like MATLAB
            fps: Video framerate
        """
        print(f"Generating AVI: {output_file}")
        print(f"   Type: {display_type}, FPS: {fps}")
        
        # MATLAB behavior: default to full file [1, numframes]
        if frame_range is None:
            frame_range = [1, self.cap.file_info['numframes']]
        
        start_frame, end_frame = frame_range
        total_frames = end_frame - start_frame + 1
        
        # Get first frame to determine dimensions using clean frame reading
        first_frame = self.read_frame_direct(start_frame)
        
        if display_type == 'raw':
            # Raw sonar data (polar display)
            height, width = first_frame.shape
            is_color = False
        elif display_type == 'polar':
            # Polar display (same as raw but with different processing)
            height, width = first_frame.shape
            is_color = False
        else:  # cartesian
            # Fan-shaped mapping - get dimensions from actual mapping (like MATLAB make_first_image)
            temp_data = {
                'frame': first_frame,
                'numbeams': self.cap.file_info['numbeams'],
                'sampleperchannel': self.cap.file_info['sampleperchannel'],
                'minrange': self.cap.file_info['minrange'],
                'maxrange': self.cap.file_info['maxrange']
            }
            # MATLAB uses make_first_image(data, 4, 400)
            test_img = self.mapper.make_first_image(temp_data, smooth=4, imagexsize=400)
            height, width = test_img.shape
            is_color = False
        
        # Initialize video writer (like MATLAB's 'Uncompressed AVI')
        # Try uncompressed first, fallback to MJPG if not supported
        try:
            fourcc = cv2.VideoWriter_fourcc(*'DIB ')  # Uncompressed RGB
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), is_color)
            if not video_writer.isOpened():
                raise Exception("Uncompressed not supported")
        except:
            print("   Note: Using MJPG compression (uncompressed not available)")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Fallback to compression
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), is_color)
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_file}")
        
        print(f"   Video dimensions: {width}x{height}")
        print(f"   Processing {total_frames} frames...")
        
        # MATLAB optimization: prepare mapping for cartesian display once
        if display_type == 'cartesian':
            # Like MATLAB's make_first_image(data, 4, 400) setup
            mapping_data = {
                'frame': first_frame,
                'numbeams': self.cap.file_info['numbeams'],
                'sampleperchannel': self.cap.file_info['sampleperchannel'],
                'minrange': self.cap.file_info['minrange'],
                'maxrange': self.cap.file_info['maxrange']
            }
            # Pre-compute the mapping (like MATLAB does once)
            first_mapped_image = self.mapper.make_first_image(mapping_data, smooth=4, imagexsize=400)
        
        # Process frames (like MATLAB's loop)
        for i, frame_num in enumerate(range(start_frame, end_frame + 1)):
            # Use clean frame reading (fix the corruption issue)
            current_frame = self.read_frame_direct(frame_num)
            
            if display_type == 'raw' or display_type == 'polar':
                # Raw/polar sonar data (like MATLAB's polar mode)
                img = current_frame.astype(np.uint8)
                
            else:  # cartesian
                if i == 0:
                    # First frame: use pre-computed mapping
                    img = first_mapped_image.astype(np.uint8)
                else:
                    # Subsequent frames: like MATLAB's make_new_image optimization
                    # For now, we'll use make_first_image but this could be optimized later
                    temp_data = {
                        'frame': current_frame,
                        'numbeams': self.cap.file_info['numbeams'],
                        'sampleperchannel': self.cap.file_info['sampleperchannel'],
                        'minrange': self.cap.file_info['minrange'],
                        'maxrange': self.cap.file_info['maxrange']
                    }
                    img = self.mapper.make_first_image(temp_data, smooth=4, imagexsize=400).astype(np.uint8)
            
            # Ensure correct format for video writer
            if len(img.shape) == 2:
                # Convert grayscale to the expected format
                if is_color:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            video_writer.write(img)
            
            # Progress update
            if (i + 1) % 10 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"   Progress: {progress:.1f}% ({i + 1}/{total_frames} frames)")
        
        video_writer.release()
        print(f"AVI export complete: {output_file}")
        return output_file
    
    def convert_to_images(self, output_dir, frame_range=None, image_format='png'):
        """
        MATLAB equivalent: convert_sonar_to_mat with image output
        Export frames as individual images
        """
        print(f"Converting to images: {output_dir}")
        
        if frame_range is None:
            frame_range = [1, min(50, self.cap.file_info['numframes'])]
        
        start_frame, end_frame = frame_range
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        for frame_num in range(start_frame, end_frame + 1):
            # Use direct frame reading to avoid CapMultiThreading corruption
            frame = self.read_frame_direct(frame_num)
            
            # Get actual frame range values (not file header values)
            frame_header_data = self.cap.get_frame_new(frame_num)
            minrange = frame_header_data['minrange']
            maxrange = frame_header_data['maxrange']
            
            # Safety check for invalid range values
            if maxrange <= minrange or (maxrange - minrange) <= 0:
                print(f"   WARNING: Invalid range for frame {frame_num} (min={minrange:.4f}, max={maxrange:.4f})")
                print("   Using default range values...")
                minrange = 1.0
                maxrange = 5.0
            
            frame_data = {
                'frame': frame,
                'minrange': minrange,
                'maxrange': maxrange
            }
            
            # Raw image - rotate to match MATLAB orientation (vertical)
            raw_frame = frame_data['frame']
            # MATLAB displays as (range samples, beams) vertically
            raw_frame_oriented = raw_frame.T  # Transpose to (128, 800) for vertical display
            
            # Apply colormap to raw image
            raw_colored = self.apply_colormap_to_grayscale(raw_frame_oriented)
            raw_file = f"{output_dir}/frame_{frame_num:04d}_raw.{image_format}"
            cv2.imwrite(raw_file, raw_colored)
            exported_files.append(raw_file)
            
            # Fan mapping - use proper MATLAB-equivalent coordinate mapping
            temp_data = {
                'frame': frame_data['frame'],
                'numbeams': self.cap.file_info['numbeams'],
                'sampleperchannel': self.cap.file_info['sampleperchannel'],
                'minrange': frame_data['minrange'],
                'maxrange': frame_data['maxrange']
            }
            # Use the static method directly to ensure correct processing
            fan_img = FinalCleanMapper.make_first_image(temp_data, smooth=4, imagexsize=400)
            
            # Apply colormap to fan image
            fan_colored = self.apply_colormap_to_grayscale(fan_img.astype(np.uint8))
            fan_file = f"{output_dir}/frame_{frame_num:04d}_fan.{image_format}"
            cv2.imwrite(fan_file, fan_colored)
            exported_files.append(fan_file)
            
            if frame_num % 10 == 1:
                print(f"   Exported frame {frame_num}")
        
        print(f"Exported {len(exported_files)} images to {output_dir}")
        return exported_files
    
    def convert_to_numpy(self, output_file, frame_range=None):
        """
        MATLAB equivalent: convert_sonar_to_mat with MAT output
        Export frames as NumPy archive (Python equivalent of MAT)
        """
        print(f"💾 Converting to NumPy: {output_file}")
        
        if frame_range is None:
            frame_range = [1, min(100, self.cap.file_info['numframes'])]
        
        start_frame, end_frame = frame_range
        total_frames = end_frame - start_frame + 1
        
        # Pre-allocate arrays
        frame_shape = (self.cap.file_info['sampleperchannel'], self.cap.file_info['numbeams'])
        frames_array = np.zeros((total_frames, *frame_shape), dtype=np.uint8)
        timestamps = []
        frame_numbers = []
        
        print(f"   Collecting {total_frames} frames...")
        
        for i, frame_num in enumerate(range(start_frame, end_frame + 1)):
            frame_data = self.cap.get_frame_new(frame_num)
            frames_array[i] = frame_data['frame']
            timestamps.append(frame_data['frame_header'].get('sonartimestamp', 0))
            frame_numbers.append(frame_num)
            
            if (i + 1) % 25 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"   Progress: {progress:.1f}%")
        
        # Save data
        np.savez_compressed(output_file,
            frames=frames_array,
            timestamps=np.array(timestamps),
            frame_numbers=np.array(frame_numbers),
            file_info=self.cap.file_info,
            minrange=self.cap.file_info['minrange'],
            maxrange=self.cap.file_info['maxrange'],
            numbeams=self.cap.file_info['numbeams'],
            sampleperchannel=self.cap.file_info['sampleperchannel']
        )
        
        print(f"NumPy export complete: {output_file}")
        return output_file
    
    # ===== STATIC PATTERN (MATLAB parity) =====
    def compute_static_pattern(self, frame_limit=None, imagexsize=400, smooth=4, half_angle=14.0):
        """
        Compute average static pattern in the cartesian (fan) domain.
        Matches MATLAB's compute_average_pattern_ddf: averages ALL frames.
        
        Args:
            frame_limit: Number of frames to average (None = all frames, matching MATLAB)
            imagexsize: Fan image width (must match detection mapping)
            smooth: Beam smoothing factor
            half_angle: Half field-of-view in degrees for mapscan
        Returns:
            pattern (float64 ndarray): Averaged fan image in 0..255 scale
        """
        from aris_sonar_processing import FinalCleanMapper
        import numpy as _np
        
        total_frames = self.cap.file_info['numframes']
        num_frames = total_frames if frame_limit is None else min(frame_limit, total_frames)
        if num_frames <= 0:
            raise ValueError("No frames available to compute static pattern")
        imagexsize = max(32, int(imagexsize))
        smooth = max(1, int(smooth))
        half_angle = float(half_angle)
        
        print(f"   Averaging {num_frames} of {total_frames} frames (MATLAB uses all)...")
        accumulator = None
        map_data = None
        current_minrange = None
        current_maxrange = None
        map_rebuild_count = 0
        nrows = self.cap.file_info['numbeams'] * smooth - smooth + 1
        
        for i in range(1, num_frames + 1):
            frame_info = self.cap.get_frame_new(i)
            frame = frame_info['frame']
            minrange = frame_info['minrange']
            maxrange = frame_info['maxrange']

            # MATLAB behavior: build map once and only rebuild if range changes.
            range_changed = bool(frame_info.get('range_changed', False))
            if (
                map_data is None
                or range_changed
                or minrange != current_minrange
                or maxrange != current_maxrange
            ):
                map_data = FinalCleanMapper.mapscan(
                    imagexsize,
                    maxrange,
                    minrange,
                    half_angle,
                    nrows,
                    self.cap.file_info['sampleperchannel']
                )
                current_minrange = minrange
                current_maxrange = maxrange
                map_rebuild_count += 1

            if smooth > 1:
                if smooth == 4:
                    frame = FinalCleanMapper.expand4(frame)
                else:
                    frame = FinalCleanMapper.smooth1(frame, smooth, 'expand')
            frame = frame.copy()
            frame[0, 0] = 0

            frame_flat = frame.flatten(order='F')
            svector_values = frame_flat[map_data['svector'] - 1]
            fan_img = svector_values.reshape(
                map_data['iysize'],
                imagexsize,
                order='F'
            )

            if accumulator is None:
                accumulator = _np.zeros_like(fan_img, dtype=_np.float64)
            accumulator += fan_img.astype(_np.float64)
            if i % 100 == 0 or i == num_frames:
                print(f"   Averaged {i}/{num_frames} frames...")
        
        pattern = accumulator / num_frames
        print(f"Static pattern computed: {pattern.shape} (map rebuilt {map_rebuild_count}x)")
        return pattern
    
    def save_static_pattern(self, output_dir, pattern, base_name=None, also_mat=True):
        """
        Save static pattern to NPY (runtime) and MAT (MATLAB-style) under output_dir.
        Returns dict with paths.
        """
        from pathlib import Path as _Path
        from scipy.io import savemat as _savemat
        import numpy as _np
        
        out_dir = _Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if base_name is None:
            base_name = _Path(self.filename).stem

        pattern = _np.asarray(pattern, dtype=_np.float64)
        if pattern.size == 0:
            raise ValueError("Pattern is empty")

        if _np.nanmax(pattern) <= 1.0:
            pattern_norm = _np.clip(pattern, 0.0, 1.0)
            pattern_uint8 = _np.clip(_np.rint(pattern_norm * 255.0), 0, 255).astype(_np.uint8)
        else:
            pattern_clipped = _np.clip(pattern, 0.0, 255.0)
            pattern_uint8 = _np.clip(_np.rint(pattern_clipped), 0, 255).astype(_np.uint8)
            pattern_norm = pattern_clipped / 255.0
        
        npy_path = out_dir / f"{base_name}_pattern.npy"
        _np.save(str(npy_path), pattern_uint8)
        result = {'npy': str(npy_path)}
        
        if also_mat:
            mat_path = out_dir / f"{base_name}_pattern.mat"
            # Keep MATLAB variable name and normalized floating-point scale.
            _savemat(str(mat_path), {'Pattern': pattern_norm.astype(_np.float64)})
            result['mat'] = str(mat_path)
        
        print(f"Static pattern saved: {result}")
        return result
    
    def close(self):
        """Close file handles"""
        self.cap.release()


class ARISBatchProcessor:
    """Batch processing of multiple ARIS/DDF files"""
    
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def process_directory(self, output_dir, file_pattern=None,
                         export_type="images", frame_limit=50):
        """
        MATLAB equivalent: convert_sonar_to_mat batch processing
        Process all ARIS/DDF files in a directory
        
        Args:
            output_dir: Output directory
            file_pattern: Optional file pattern to match (e.g., "*.aris")
            export_type: "images", "avi", or "numpy"
            frame_limit: Maximum frames per file
        """
        print(f"Batch processing: {self.input_dir}")
        if file_pattern:
            print(f"   Pattern: {file_pattern}")
        else:
            print("   Pattern: default (*.aris + *.ddf)")
        print(f"   Export type: {export_type}")
        
        sonar_files = discover_sonar_files(self.input_dir, file_pattern=file_pattern)
        if not sonar_files:
            if file_pattern:
                print(f"ERROR: No files found matching {file_pattern}")
            else:
                print("ERROR: No .aris/.ddf files found")
            return []
        
        print(f"   Found {len(sonar_files)} files")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, aris_file in enumerate(sonar_files):
            print(f"\n📄 Processing file {i+1}/{len(sonar_files)}: {aris_file.name}")
            
            try:
                exporter = ARISExporter(str(aris_file))
                
                # Create file-specific output directory
                file_output_dir = output_path / aris_file.stem
                
                if export_type == "images":
                    exported = exporter.convert_to_images(
                        str(file_output_dir), 
                        frame_range=[1, frame_limit]
                    )
                elif export_type == "avi":
                    output_file = str(output_path / f"{aris_file.stem}.avi")
                    exported = exporter.generate_avi(
                        output_file, 
                        display_type='cartesian',
                        frame_range=[1, frame_limit]
                    )
                elif export_type == "numpy":
                    output_file = str(output_path / f"{aris_file.stem}.npz")
                    exported = exporter.convert_to_numpy(
                        output_file,
                        frame_range=[1, frame_limit]
                    )
                else:
                    raise ValueError(f"Unknown export type: {export_type}")
                
                results.append({
                    'input_file': str(aris_file),
                    'output': exported,
                    'status': 'success'
                })
                
                exporter.close()
                print(f"Completed: {aris_file.name}")
                
            except Exception as e:
                print(f"ERROR processing {aris_file.name}: {e}")
                results.append({
                    'input_file': str(aris_file),
                    'output': None,
                    'status': f'error: {e}'
                })
        
        # Save processing summary
        summary_file = output_path / "batch_processing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ARIS/DDF Batch Processing Summary\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Export type: {export_type}\n")
            f.write(f"Frame limit: {frame_limit}\n\n")
            
            f.write(f"Results:\n")
            for result in results:
                f.write(f"  {result['input_file']}: {result['status']}\n")
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\nBatch processing complete!")
        print(f"   Successful: {success_count}/{len(sonar_files)} files")
        print(f"   Summary: {summary_file}")
        
        return results


# Convenience functions for direct use
def export_aris_to_avi(input_file, output_file, display_type='cartesian', 
                      frame_range=None, fps=12):
    """Quick export ARIS to AVI"""
    exporter = ARISExporter(input_file)
    try:
        return exporter.generate_avi(output_file, display_type, frame_range, fps)
    finally:
        exporter.close()


def export_aris_to_images(input_file, output_dir, frame_range=None):
    """Quick export ARIS to images"""
    exporter = ARISExporter(input_file)
    try:
        return exporter.convert_to_images(output_dir, frame_range)
    finally:
        exporter.close()


def batch_process_aris_directory(input_dir, output_dir, export_type="images", 
                                frame_limit=50, file_pattern=None):
    """Quick batch processing"""
    processor = ARISBatchProcessor(input_dir)
    return processor.process_directory(
        output_dir,
        file_pattern=file_pattern,
        export_type=export_type,
        frame_limit=frame_limit
    )


class ARISTimeIndexer:
    """
    Python equivalent of arisreader_timeindex.m
    Reads multiple .aris/.ddf files in a directory and generates time index files.
    
    Returns:
    T = [matlabtime pingnumber filenumber]
    d = filename structure referring to filenumber above
    """
    
    def __init__(self):
        self.file_info = []
        
    def generate_time_index(self, data_dir):
        """
        MATLAB equivalent: [T,d] = arisreader_timeindex(dataDir)
        
        Args:
            data_dir: Directory containing .aris/.ddf files
            
        Returns:
            T: Time index array [matlabtime, pingnumber, filenumber]
            d: File structure array with filenames
        """
        import os
        import numpy as np
        
        print(f"Generating time index for directory: {data_dir}")
        
        # Check if directory exists
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")
            
        sonar_files = [str(path) for path in discover_sonar_files(data_dir)]
        if not sonar_files:
            raise ValueError(f"No .aris/.ddf files found in directory: {data_dir}")
            
        print(f"Found {len(sonar_files)} .aris/.ddf files")
        
        # Sort files for consistent ordering
        sonar_files.sort()
        d = [os.path.basename(f) for f in sonar_files]
        
        # Calculate total frames across all files
        N = np.zeros(len(sonar_files) + 1, dtype=int)
        Tsub = []
        
        for i, filename in enumerate(sonar_files):
            print(f"File {i+1} of {len(sonar_files)}: {os.path.basename(filename)}")
            
            # Read time data from this file
            T_file = self._read_aris_time(filename)
            Tsub.append(T_file)
            N[i+1] = T_file.shape[0]
            
        # Combine all time data
        total_frames = np.sum(N)
        T = np.full((total_frames, 3), np.nan)
        N = np.cumsum(N)
        
        for i in range(len(N) - 1):
            start_idx = N[i]
            end_idx = N[i+1]
            T[start_idx:end_idx, 2] = i + 1  # File number (1-based)
            T[start_idx:end_idx, 0:2] = Tsub[i]  # Time and ping number
            
        print(f"Time index generated: {total_frames} total frames across {len(sonar_files)} files")
        
        return T, d
        
    def _read_aris_time(self, filename):
        """
        MATLAB equivalent: ReadARISTime(filename)
        Read time data from a single ARIS file
        """
        # Use our existing frame reader
        cap = CapMultiThreading(filename)
        
        # Get file info
        file_info = cap.file_info
        numframes = file_info['numframes']
        
        # Initialize time array
        T = np.full((numframes, 2), np.nan)

        # Prefer parsed header timestamps when available; fallback to synthetic FPS time.
        first_frame = cap.get_frame_new(1)
        first_header = first_frame.get('frame_header', {})
        first_datenum = first_header.get('datenum', first_frame.get('datenum'))
        if first_datenum is not None:
            T[0, 0] = float(first_datenum)
        else:
            T[0, 0] = 0.0
        T[0, 1] = 1  # First ping number
        
        for i in range(1, numframes):
            frame_data = cap.get_frame_new(i + 1)
            frame_header = frame_data.get('frame_header', {})
            
            # Try to use actual timestamp if available
            datenum_value = frame_header.get('datenum', frame_data.get('datenum'))
            if datenum_value is not None:
                T[i, 0] = float(datenum_value)
            else:
                # Use synthetic time (MATLAB approach for ARIS)
                # Time step based on frame rate
                T[i, 0] = i * (1.0 / file_info.get('framerate', 12.0))
                
            T[i, 1] = i + 1  # Ping number (1-based)
            
        cap.release()  # CapMultiThreading uses release(), not close()
        return T
        
    def save_time_index(self, data_dir, T, d, output_filename="T.mat", create_csv=True):
        """
        Save time index to .mat file (MATLAB equivalent) and optionally CSV
        
        Args:
            data_dir: Directory to save files
            T: Time index array [matlabtime, pingnumber, filenumber]
            d: File list
            output_filename: Output filename for .mat file
            create_csv: If True, also create a CSV version
        """
        import scipy.io
        import os
        
        output_path = os.path.join(data_dir, output_filename)
        
        # Create structure similar to MATLAB
        time_data = {
            'T': T,
            'd': d,
            'description': 'Time index: T = [matlabtime, pingnumber, filenumber]'
        }
        
        # Save .mat file
        scipy.io.savemat(output_path, time_data)
        print(f"💾 Time index saved to: {output_path}")
        
        # Also create CSV version if requested
        if create_csv:
            csv_path = self.save_time_index_csv(data_dir, T, d)
            return output_path, csv_path
        
        return output_path
    
    def save_time_index_csv(self, data_dir, T, d, output_filename="time_index.csv"):
        """
        Save time index to CSV file for easy analysis
        
        Args:
            data_dir: Directory to save file
            T: Time index array [matlabtime, pingnumber, filenumber]
            d: File list
            output_filename: Output filename for CSV file
        """
        import pandas as pd
        import os
        
        output_path = os.path.join(data_dir, output_filename)
        
        # Create DataFrame with descriptive column names
        df = pd.DataFrame(T, columns=['Time_seconds', 'Ping_number', 'File_number'])
        
        # Add additional useful columns
        df['Time_minutes'] = df['Time_seconds'] / 60
        df['Time_hours'] = df['Time_seconds'] / 3600
        
        # Add filename for each frame
        filenames = []
        for fn in df['File_number']:
            file_idx = int(fn) - 1  # Convert to 0-based index
            if file_idx < len(d):
                filenames.append(d[file_idx])
            else:
                filenames.append(f"file_{int(fn)}")
        df['Filename'] = filenames
        
        # Calculate relative time within each file
        df['Time_in_file_seconds'] = df.groupby('File_number')['Time_seconds'].transform(lambda x: x - x.min())
        
        # Add frame rate information
        if len(T) > 1:
            total_time = T[-1, 0] - T[0, 0]
            frame_rate = len(T) / (total_time + 1/12)  # Add one frame interval
            df['Estimated_FPS'] = frame_rate
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        print(f"CSV time index saved to: {output_path}")
        print(f"   Columns: {list(df.columns)}")
        
        return output_path


if __name__ == "__main__":
    # Demo usage
    example_file = "example_data/DH1112_2015-11-08_115514.aris"
    
    if os.path.exists(example_file):
        print("ARIS/DDF Export Utilities Demo")
        print("=" * 50)
        
        # Export to AVI
        print("\nExporting to AVI...")
        export_aris_to_avi(example_file, "demo_output.avi", 
                          frame_range=[1, 25], fps=10)
        
        # Export to images
        print("\nExporting to images...")
        export_aris_to_images(example_file, "demo_images", 
                             frame_range=[1, 10])
        
        print("\nDemo complete!")
    else:
        print("ERROR: Example file not found. Place an ARIS/DDF file to test.")