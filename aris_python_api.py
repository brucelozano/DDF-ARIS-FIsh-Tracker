#!/usr/bin/env python3
"""
Complete ARIS Reader - Full MATLAB Equivalent
Provides all MATLAB functions in a single, organized interface
"""

import os
import sys
import argparse
import numpy as np
from opencv_player import OpenCVPlayer, CapMultiThreading
from aris_utilities import ARISExporter, ARISBatchProcessor, export_aris_to_avi, export_aris_to_images, batch_process_aris_directory


class CompleteARISReader:
    """
    Complete ARIS reader with all MATLAB equivalent functions
    Provides a unified interface for all ARIS operations
    """
    
    def __init__(self, filename):
        self.filename = filename
        self.cap = CapMultiThreading(filename)
        self.current_data = None
        
    def get_frame_first(self):
        """
        MATLAB equivalent: data = get_frame_first(filename)
        Read first frame and file info
        """
        frame_data = self.cap.get_frame_new(1)
        
        self.current_data = {
            'version': self.cap.file_info.get('version', 5),
            'minrange': frame_data['minrange'],
            'maxrange': frame_data['maxrange'],
            'numbeams': self.cap.file_info['numbeams'],
            'numframes': self.cap.file_info['numframes'],
            'sampleperchannel': self.cap.file_info['sampleperchannel'],
            'frame': frame_data['frame'],
            'framenumber': 1,
            'reverse': self.cap.file_info['reverse'],
            'flag': 0,
            'framerate': self.cap.file_info['framerate'],
            'serialnumber': self.cap.file_info['serialnumber'],
            'receivergain': self.cap.file_info['receivergain']
        }
        
        print(f"get_frame_first: Frame 1, shape {frame_data['frame'].shape}")
        return self.current_data
    
    def get_frame_new(self, frame_number):
        """
        MATLAB equivalent: data = get_frame_new(data, framenumber)
        Read any frame by number
        """
        frame_data = self.cap.get_frame_new(frame_number)
        
        if self.current_data is not None:
            # Update existing data structure
            self.current_data.update({
                'frame': frame_data['frame'],
                'framenumber': frame_number,
                'minrange': frame_data['minrange'],
                'maxrange': frame_data['maxrange'],
                'flag': 1 if frame_data['range_changed'] else 0
            })
        else:
            # Create new data structure
            self.current_data = {
                'version': self.cap.file_info.get('version', 5),
                'minrange': frame_data['minrange'],
                'maxrange': frame_data['maxrange'],
                'numbeams': self.cap.file_info['numbeams'],
                'numframes': self.cap.file_info['numframes'],
                'sampleperchannel': self.cap.file_info['sampleperchannel'],
                'frame': frame_data['frame'],
                'framenumber': frame_number,
                'reverse': self.cap.file_info['reverse'],
                'flag': 1 if frame_data['range_changed'] else 0
            }
        
        print(f"get_frame_new: Frame {frame_number}, shape {frame_data['frame'].shape}")
        return self.current_data
    
    def make_first_image(self, smooth=4, imagexsize=400):
        """
        MATLAB equivalent: data = make_first_image(data, smooth, imagexsize)
        Create processed image from frame data
        """
        if self.current_data is None:
            raise ValueError("Must call get_frame_first() first")
        
        from aris_sonar_processing import FinalCleanMapper
        mapper = FinalCleanMapper()
        
        # Apply beam smoothing if requested
        frame = self.current_data['frame']
        if smooth > 1:
            frame = mapper.expand4(frame)  # Use static method
        
        # Use the proper MATLAB-equivalent coordinate mapping
        image = mapper.make_first_image(self.current_data, smooth=smooth, imagexsize=imagexsize)
        
        self.current_data['image'] = image
        print(f"make_first_image: Image shape {image.shape}, smooth={smooth}")
        return self.current_data
    
    def make_new_image(self, frame_data):
        """
        MATLAB equivalent: data = make_new_image(data, frame_in)
        Update image with new frame data
        """
        if self.current_data is None or 'image' not in self.current_data:
            raise ValueError("Must call make_first_image() first")
        
        # Create temporary data structure for the new frame
        temp_data = self.current_data.copy()
        temp_data['frame'] = frame_data
        
        # Use the proper MATLAB-equivalent coordinate mapping
        from aris_sonar_processing import FinalCleanMapper
        mapper = FinalCleanMapper()
        image = mapper.make_first_image(temp_data, smooth=4, imagexsize=self.current_data['image'].shape[1])
        
        self.current_data['image'] = image
        print(f"make_new_image: Updated image shape {image.shape}")
        return self.current_data
    
    def motion_correction(self, velocity):
        """
        MATLAB equivalent: framenew = motion_correction(data, velocity)
        Apply motion correction for platform velocity
        """
        if self.current_data is None:
            raise ValueError("Must have frame data loaded")
        
        corrected_frame = self.cap.motion_correction(self.current_data['frame'], velocity)
        print(f"motion_correction: Applied velocity {velocity} m/s")
        return corrected_frame
    
    def runmovie3(self, first_frame=1, last_frame=500):
        """
        MATLAB equivalent: runmovie3 script
        Play movie sequence with OpenCV
        """
        print(f"Starting movie player: frames {first_frame} to {last_frame}")
        player = OpenCVPlayer(self.filename)
        player.run_player(first_frame, last_frame)
    
    def generate_avi(self, output_file, display_type='cartesian', frame_range=None, fps=12):
        """
        MATLAB equivalent: arisreader_generateavi(filein, fileout, type, index)
        Export ARIS data to AVI video
        """
        exporter = ARISExporter(self.filename)
        try:
            return exporter.generate_avi(output_file, display_type, frame_range, fps)
        finally:
            exporter.close()
    
    def convert_to_images(self, output_dir, frame_range=None):
        """
        MATLAB equivalent: convert_sonar_to_mat with image output
        Export frames as images
        """
        # Use our existing CapMultiThreading instance instead of creating new ARISExporter
        # to ensure consistent frame reading
        from aris_utilities import ARISExporter
        from pathlib import Path
        import cv2
        import numpy as np
        
        print(f"Converting to images: {output_dir}")
        
        if frame_range is None:
            frame_range = [1, min(50, self.cap.file_info['numframes'])]
        
        start_frame, end_frame = frame_range
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load bluebar colormap directly
        try:
            from scipy.io import loadmat
            mat_data = loadmat('bluebar.mat')
            bluebar_cmap = (mat_data['bluebar'] * 255).astype(np.uint8)
        except:
            bluebar_cmap = None
        
        def apply_colormap(gray_image):
            """Apply bluebar colormap to grayscale image"""
            if bluebar_cmap is not None:
                colored_image = bluebar_cmap[gray_image]
                return colored_image[:, :, [2, 1, 0]]  # Convert RGB to BGR
            else:
                return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        exported_files = []
        
        for frame_num in range(start_frame, end_frame + 1):
            # Use OUR CapMultiThreading instance for consistent frame reading
            frame_data = self.cap.get_frame_new(frame_num)
            
            # Raw image - rotate to match MATLAB orientation (vertical)
            raw_frame = frame_data['frame']
            raw_frame_oriented = raw_frame.T  # Transpose to (128, 800) for vertical display
            
            # Apply colormap to raw image
            raw_colored = apply_colormap(raw_frame_oriented)
            raw_file = f"{output_dir}/frame_{frame_num:04d}_raw.png"
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
            from aris_sonar_processing import FinalCleanMapper
            fan_img = FinalCleanMapper.make_first_image(temp_data, smooth=4, imagexsize=400)
            
            # Apply colormap to fan image
            fan_colored = apply_colormap(fan_img.astype(np.uint8))
            fan_file = f"{output_dir}/frame_{frame_num:04d}_fan.png"
            cv2.imwrite(fan_file, fan_colored)
            exported_files.append(fan_file)
            
            if frame_num % 10 == 1:
                print(f"   Exported frame {frame_num}")
        
        print(f"Exported {len(exported_files)} images to {output_dir}")
        return exported_files
    
    def convert_to_numpy(self, output_file, frame_range=None):
        """
        MATLAB equivalent: convert_sonar_to_mat with MAT output
        Export data as NumPy archive
        """
        exporter = ARISExporter(self.filename)
        try:
            return exporter.convert_to_numpy(output_file, frame_range)
        finally:
            exporter.close()
    
    def close(self):
        """Close file handles"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_aris_beam_from_angle(self, angles):
        """
        Python equivalent of get_aris_beam_from_angle.m
        Converts angles to beam numbers using ARIS distortion pattern
        """
        from aris_sonar_processing import FinalCleanMapper
        return FinalCleanMapper.get_aris_beam_from_angle(angles)
    
    def smooth1(self, inarray, factor, method='expand'):
        """
        Python equivalent of smooth1.m
        Advanced beam interpolation with multiple methods
        """
        from aris_sonar_processing import FinalCleanMapper
        return FinalCleanMapper.smooth1(inarray, factor, method)
    
    def generate_time_index(self, data_dir):
        """
        Python equivalent of arisreader_timeindex.m
        Generate time index for multiple ARIS files in a directory
        """
        from aris_utilities import ARISTimeIndexer
        indexer = ARISTimeIndexer()
        return indexer.generate_time_index(data_dir)
    
    def get_extended_frame_header(self, frame_number):
        """
        Extended frame header parsing including GPS, accelerometer, and 3D data
        Based on test_aris_reader.m comprehensive header reading
        """
        # Get standard frame header first
        frame_data = self.cap.get_frame_new(frame_number)
        frame_header = frame_data.get('frame_header', {})
        
        # Extract additional fields if available in frame header
        extended_header = {
            # Standard fields (already available)
            'frameindex': frame_header.get('framenumber', frame_number),
            'timestamp': frame_header.get('timestamp', None),
            'sonartimestamp': frame_header.get('sonartimestamp', None),
            'datenum': frame_header.get('datenum', None),
            'depth': frame_header.get('depth', None),
            'compassheading': frame_header.get('compassheading', None),
            'compasspitch': frame_header.get('compasspitch', None),
            'compassroll': frame_header.get('compassroll', None),
            'watertemp': frame_header.get('watertemp', None),
            'salinity': frame_header.get('salinity', None),
            'pressure': frame_header.get('pressure', None),
            
            # Extended fields (would need to be parsed from raw frame header)
            # Note: These would require extending the frame header parsing
            # in opencv_native_player.py to include all fields from test_aris_reader.m
            'latitude': frame_header.get('latitude', None),  # GPS latitude
            'longitude': frame_header.get('longitude', None),  # GPS longitude
            'sonarposition': frame_header.get('sonarposition', None),  # Special for PNNL
            'beamtilt': None,  # Beam tilt angle
            'targetrange': None,  # Target range
            'targetbearing': None,  # Target bearing
            'accellx': None,  # X-axis acceleration
            'accelly': None,  # Y-axis acceleration
            'accellz': None,  # Z-axis acceleration
            'sonarx': None,  # Sonar X location for 3D processing
            'sonary': None,  # Sonar Y location for 3D processing
            'sonarz': None,  # Sonar Z location for 3D processing
            'sonarpan': frame_header.get('sonarpan', None),  # X2 pan output
            'sonartilt': frame_header.get('sonartilt', None),  # X2 tilt output
            'sonarroll': frame_header.get('sonarroll', None),  # X2 roll output
            'tmatrix': None,  # 3D processing transformation matrix (16 floats)
        }
        
        return extended_header


def main():
    """Command line interface for ARIS operations"""
    parser = argparse.ArgumentParser(description='Complete ARIS/DDF Reader - MATLAB Equivalent')
    parser.add_argument('command', choices=['play', 'export-avi', 'export-images', 'batch'], 
                       help='Operation to perform')
    parser.add_argument('input', help='Input ARIS/DDF file or directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--frames', '-f', nargs=2, type=int, default=[1, 50],
                       help='Frame range [start end] (default: 1 50)')
    parser.add_argument('--fps', type=int, default=12, help='Video FPS (default: 12)')
    parser.add_argument('--type', choices=['raw', 'polar', 'cartesian'], default='cartesian',
                       help='Display type for AVI export (default: cartesian)')
    parser.add_argument('--limit', type=int, default=50, help='Frame limit for batch processing')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'play':
            # Play movie
            if not os.path.exists(args.input):
                print(f"ERROR: File not found: {args.input}")
                return 1
            
            with CompleteARISReader(args.input) as reader:
                reader.runmovie3(args.frames[0], args.frames[1])
        
        elif args.command == 'export-avi':
            # Export to AVI
            if not args.output:
                args.output = os.path.splitext(args.input)[0] + '.avi'
            
            with CompleteARISReader(args.input) as reader:
                reader.generate_avi(args.output, args.type, args.frames, args.fps)
        
        elif args.command == 'export-images':
            # Export to images
            if not args.output:
                args.output = os.path.splitext(args.input)[0] + '_images'
            
            with CompleteARISReader(args.input) as reader:
                reader.convert_to_images(args.output, args.frames)
        
        elif args.command == 'batch':
            # Batch processing
            if not args.output:
                args.output = args.input + '_output'
            
            processor = ARISBatchProcessor(args.input)
            processor.process_directory(args.output, export_type='images', frame_limit=args.limit)
        
        print("Operation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    # Demo usage if run directly
    if len(sys.argv) == 1:
        print("Complete ARIS Reader - MATLAB Equivalent")
        print("=" * 60)
        print("Available operations:")
        print("  python aris_python_api.py play file.aris")
        print("  python aris_python_api.py export-avi file.aris --output video.avi")
        print("  python aris_python_api.py export-images file.aris --output images/")
        print("  python aris_python_api.py batch directory/ --output results/")
        print()
        
        # Try demo with example file
        example_file = "example_data/DH1112_2015-11-08_115514.aris"
        if os.path.exists(example_file):
            print("Demo with example file:")
            print(f"   {example_file}")
            print()
            
            try:
                with CompleteARISReader(example_file) as reader:
                    print("MATLAB Workflow Demo:")
                    
                    # Complete MATLAB workflow
                    data = reader.get_frame_first()
                    data = reader.make_first_image(smooth=4, imagexsize=400)
                    data = reader.get_frame_new(25)
                    data = reader.make_new_image(data['frame'])
                    
                    # Motion correction
                    corrected = reader.motion_correction(velocity=2.5)
                    
                    print("\nAll MATLAB functions working!")
                    print(f"   Frame shape: {data['frame'].shape}")
                    print(f"   Image shape: {data['image'].shape}")
                    print(f"   Motion corrected: {corrected.shape}")
                    
            except Exception as e:
                print(f"ERROR: Demo failed: {e}")
        else:
            print("INFO: Place an ARIS file in example_data/ to try the demo")
    else:
        sys.exit(main()) 