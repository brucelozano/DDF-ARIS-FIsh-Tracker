#!/usr/bin/env python3
"""
Interactive Mangrove Mask Creator
Create mangrove/habitat masks from ARIS frames for distance tracking.
MATLAB equivalent: polygonal_interactive_segmentation.m
"""

import os
import sys
import cv2
import numpy as np
from aris_python_api import CompleteARISReader
from aris_sonar_processing import FinalCleanMapper


class InteractiveMaskCreator:
    """
    Interactive tool to create mangrove masks from ARIS frames.
    MATLAB equivalent: polygonal_interactive_segmentation.m
    """
    
    def __init__(self, aris_file, frame_number=10, imagexsize=400):
        self.aris_file = aris_file
        self.frame_number = frame_number
        self.imagexsize = imagexsize
        self.mask = None
        self.image = None
        self.points = []
        self.drawing = False
        self.display_image = None
        
    def load_frame(self):
        """Load and prepare reference frame"""
        print(f"\nLoading frame {self.frame_number} from {os.path.basename(self.aris_file)}...")
        
        with CompleteARISReader(self.aris_file) as reader:
            frame_data = reader.get_frame_new(self.frame_number)
            
            # Create fan image (same as detection pipeline)
            temp_data = {
                'frame': frame_data['frame'],
                'numbeams': reader.cap.file_info['numbeams'],
                'sampleperchannel': reader.cap.file_info['sampleperchannel'],
                'minrange': frame_data['minrange'],
                'maxrange': frame_data['maxrange']
            }
            
            self.image = FinalCleanMapper.make_first_image(
                temp_data, smooth=4, imagexsize=self.imagexsize
            )
            
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            print(f"Frame loaded: {self.image.shape}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.drawing = True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Preview line
            img_copy = self.display_image.copy()
            if len(self.points) > 0:
                cv2.line(img_copy, self.points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow('Mangrove Selection', img_copy)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Close polygon
            if len(self.points) >= 3:
                # Fill polygon
                pts = np.array(self.points, dtype=np.int32)
                cv2.fillPoly(self.mask, [pts], 255)
                print(f"Polygon added ({len(self.points)} points)")
            
            # Reset for next polygon
            self.points = []
            self.drawing = False
    
    def create_mask_interactive(self):
        """Interactive mask creation"""
        print("\n" + "=" * 60)
        print("Interactive Mangrove Mask Creator")
        print("=" * 60)
        print("\nInstructions:")
        print("  - LEFT CLICK to add polygon points")
        print("  - RIGHT CLICK to close current polygon")
        print("  - Press 'a' to add another region")
        print("  - Press 's' to SAVE and finish")
        print("  - Press 'c' to clear all")
        print("  - Press 'q' to quit without saving")
        print("\nTip: Draw around mangrove/structure areas")
        print("=" * 60)
        
        self.load_frame()
        
        # Create display image
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        cv2.namedWindow('Mangrove Selection')
        cv2.setMouseCallback('Mangrove Selection', self.mouse_callback)
        
        while True:
            # Overlay mask
            display = self.display_image.copy()
            mask_colored = cv2.applyColorMap(self.mask, cv2.COLORMAP_JET)
            display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)
            
            # Draw current polygon points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i-1], pt, (0, 255, 0), 2)
            
            # Instructions overlay
            cv2.putText(display, "LEFT: add point | RIGHT: close polygon", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "S: save | C: clear | Q: quit", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Mangrove Selection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                break
            elif key == ord('c'):  # Clear
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.points = []
                print("Mask cleared")
            elif key == ord('a'):  # Add another region
                self.points = []
                print("Ready for next polygon")
            elif key == ord('q'):  # Quit
                print("Cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Convert to boolean
        binary_mask = self.mask > 127
        
        print(f"\nMask created: {np.sum(binary_mask)} pixels marked as mangrove")
        print(f"Coverage: {np.sum(binary_mask)/binary_mask.size*100:.1f}%")
        return binary_mask
    
    def save_mask(self, output_path):
        """Save mask as PNG"""
        if self.mask is not None:
            cv2.imwrite(output_path, self.mask)
            print(f"\nMask saved: {output_path}")
            return True
        return False


def create_mask_from_pattern(pattern_file, threshold=0.6, output_path="mangrove_mask.png"):
    """
    Automatic mangrove segmentation from static pattern.
    MATLAB equivalent: segment_mangrove_from_pattern.m
    
    Args:
        pattern_file: Path to .npy or .mat pattern file
        threshold: Intensity threshold (0-1), default 0.6
        output_path: Where to save mask PNG
    """
    import numpy as np
    import cv2
    
    print("\n" + "=" * 60)
    print("Automatic Mangrove Mask from Static Pattern")
    print("=" * 60)
    
    # Load pattern
    if pattern_file.endswith('.npy'):
        pattern = np.load(pattern_file)
        print(f"Loaded NPY pattern: {pattern_file}")
    elif pattern_file.endswith('.mat'):
        import scipy.io
        mat = scipy.io.loadmat(pattern_file)
        pattern = mat['Pattern']
        print(f"Loaded MAT pattern: {pattern_file}")
    else:
        print(f"Error: Unsupported file format. Use .npy or .mat")
        return None
    
    # Normalize to 0-1 if needed
    if pattern.max() > 1.0:
        pattern = pattern.astype(np.float32) / 255.0
    
    # Threshold (MATLAB equivalent)
    mangrove_mask = pattern >= threshold
    
    # Save as PNG
    mask_uint8 = (mangrove_mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_uint8)
    
    print(f"\nMask created:")
    print(f"   Threshold: {threshold}")
    print(f"   Mangrove pixels: {np.sum(mangrove_mask)} / {mangrove_mask.size}")
    print(f"   Coverage: {np.sum(mangrove_mask)/mangrove_mask.size*100:.1f}%")
    print(f"   Saved: {output_path}")
    
    return mangrove_mask


def main():
    """Command-line interface for mask creation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create mangrove masks for ARIS fish tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mask creation from ARIS file:
  python create_mangrove_mask.py file.aris --frame 50 --output mangrove_mask.png
  
  # Automatic mask from static pattern:
  python create_mangrove_mask.py --pattern static_pattern.npy --threshold 0.6
        """
    )
    
    parser.add_argument('aris_file', nargs='?', help='ARIS file path (for interactive mode)')
    parser.add_argument('--frame', type=int, default=10, help='Frame number to use (default: 10)')
    parser.add_argument('--output', '-o', default='mangrove_mask.png', help='Output mask file')
    parser.add_argument('--pattern', help='Create mask from static pattern file (.npy or .mat)')
    parser.add_argument('--threshold', type=float, default=0.6, 
                       help='Pattern threshold for automatic mode (default: 0.6)')
    parser.add_argument('--imagexsize', type=int, default=400, 
                       help='Image width for fan mapping (default: 400)')
    
    args = parser.parse_args()
    
    # Pattern mode
    if args.pattern:
        if not os.path.exists(args.pattern):
            print(f"Error: Pattern file not found: {args.pattern}")
            return 1
        
        mask = create_mask_from_pattern(args.pattern, args.threshold, args.output)
        if mask is not None:
            print(f"\nSuccess! Use this in your config JSON:")
            print(f'  "MANGROVE_MASK_FILE": "{args.output}"')
            return 0
        return 1
    
    # Interactive mode
    if not args.aris_file:
        parser.print_help()
        return 1
    
    if not os.path.exists(args.aris_file):
        print(f"Error: ARIS file not found: {args.aris_file}")
        return 1
    
    try:
        creator = InteractiveMaskCreator(args.aris_file, args.frame, args.imagexsize)
        mask = creator.create_mask_interactive()
        
        if mask is not None:
            creator.save_mask(args.output)
            print(f"\nSuccess! Add this to your config JSON:")
            print(f'  "MANGROVE_MASK_FILE": "{args.output}"')
            return 0
        return 1
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

