#!/usr/bin/env python3
"""
Mangrove Distance Mapper
Computes and maintains signed distance field from mangrove mask.
MATLAB equivalent: obj.mangroveDistanceImage
"""

import os
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


class MangroveDistanceMapper:
    """
    Computes and maintains signed distance field from mangrove mask.
    
    Distance convention (MATLAB-compatible):
    - Positive values: Outside mangrove (in water)
    - Negative values: Inside mangrove (in habitat)
    - Zero: Exactly on the edge
    """
    
    def __init__(self, mask_path=None, image_shape=None):
        """
        Args:
            mask_path: Path to binary PNG mask (optional)
            image_shape: (height, width) of sonar images (optional, can be set later)
        """
        self.mask = None
        self.distance_field = None
        self.mangrove_area_pixels = 0
        self.mask_path = mask_path
        
        if mask_path and image_shape:
            self.load_and_compute(mask_path, image_shape)
    
    def load_and_compute(self, mask_path, image_shape):
        """
        Load mask and compute signed distance field.
        MATLAB equivalent: bwdist(mangroveImage, 'euclidean')
        
        Args:
            mask_path: Path to binary PNG mask
            image_shape: (height, width) tuple
        """
        if not os.path.exists(mask_path):
            print(f"Warning: Mangrove mask not found: {mask_path}")
            return False
        
        # Load mask image
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print(f"Error: Could not load mangrove mask: {mask_path}")
            return False
        
        # Resize to match sonar image if needed
        if mask_img.shape != image_shape:
            print(f"Resizing mask from {mask_img.shape} to {image_shape}")
            mask_img = cv2.resize(mask_img, (image_shape[1], image_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Convert to boolean mask
        self.mask = mask_img > 127  # Binary threshold
        self.mangrove_area_pixels = np.sum(self.mask)
        
        # MATLAB equivalent: bwdist(mangroveImage, 'euclidean')
        # Distance FROM mangrove (positive outside)
        dist_outside = distance_transform_edt(~self.mask)
        
        # MATLAB equivalent: bwdist(~mangroveImage, 'euclidean')
        # Distance INSIDE mangrove (will be negative)
        dist_inside = distance_transform_edt(self.mask)
        
        # Combine: positive outside, negative inside
        # MATLAB: obj.mangroveDistanceImage(mangroveImage) = -distImgInv(mangroveImage)
        self.distance_field = dist_outside.astype(np.float32)
        self.distance_field[self.mask] = -dist_inside[self.mask]
        
        print(f"Mangrove distance field computed:")
        print(f"   Mask shape: {self.mask.shape}")
        print(f"   Mangrove pixels: {self.mangrove_area_pixels}")
        print(f"   Distance range: [{self.distance_field.min():.1f}, {self.distance_field.max():.1f}] pixels")
        
        return True
    
    def get_distance_at_position(self, x, y):
        """
        Get signed distance at fish position.
        MATLAB equivalent: obj.mangroveDistanceImage(round(y), round(x))
        
        Args:
            x, y: Fish centroid position (pixels)
        
        Returns:
            float: Distance in pixels (negative=inside, positive=outside)
        """
        if self.distance_field is None:
            return 0.0
        
        # Clamp to valid image coordinates
        h, w = self.distance_field.shape
        y_int = int(np.clip(round(y), 0, h - 1))
        x_int = int(np.clip(round(x), 0, w - 1))
        
        return float(self.distance_field[y_int, x_int])
    
    def get_mangrove_area_cm2(self, pix_scale):
        """
        Calculate total mangrove area in cm².
        MATLAB equivalent: sum(sum(mask)) * obj.pixAreaInMeters * 100
        
        Args:
            pix_scale: Meters per pixel
        
        Returns:
            float: Area in cm²
        """
        if self.mask is None:
            return 0.0
        
        # MATLAB: obj.pixAreaInMeters = obj.data.mapscale(1)*obj.data.mapscale(2)
        # Python: assume uniform scale (non-anisotropic approximation)
        area_m2 = self.mangrove_area_pixels * (pix_scale ** 2)
        area_cm2 = area_m2 * 10000  # m² to cm²
        return area_cm2
    
    def is_active(self):
        """Check if mangrove mapper is active"""
        return self.distance_field is not None
    
    def visualize_distance_field(self, output_path=None):
        """
        Create visualization of signed distance field.
        Useful for debugging and verification.
        
        Args:
            output_path: Optional path to save visualization
            
        Returns:
            numpy array: Colored visualization image
        """
        if self.distance_field is None:
            print("Warning: No distance field to visualize")
            return None
        
        # Normalize for visualization
        dist_vis = self.distance_field.copy()
        
        # Color coding: blue=inside (negative), red=outside (positive)
        vmin, vmax = dist_vis.min(), dist_vis.max()
        dist_normalized = (dist_vis - vmin) / (vmax - vmin) if vmax > vmin else dist_vis
        dist_colored = cv2.applyColorMap((dist_normalized * 255).astype(np.uint8), 
                                         cv2.COLORMAP_JET)
        
        # Overlay mask boundary in white
        edges = cv2.Canny((self.mask * 255).astype(np.uint8), 100, 200)
        dist_colored[edges > 0] = [255, 255, 255]  # White boundary
        
        # Add legend text
        cv2.putText(dist_colored, "Blue = Inside Mangrove (negative)", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(dist_colored, "Red = Outside Mangrove (positive)", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(dist_colored, f"Range: [{vmin:.1f}, {vmax:.1f}] pixels", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, dist_colored)
            print(f"Visualization saved: {output_path}")
        
        return dist_colored


class ValidationMaskFilter:
    """
    Spatial filter for fish counting in validation areas.
    MATLAB equivalent: param.ValidationMaskImageFile
    """
    
    def __init__(self, mask_path=None, image_shape=None):
        """
        Args:
            mask_path: Path to binary PNG validation mask (optional)
            image_shape: (height, width) of sonar images (optional)
        """
        self.mask = None
        self.mask_path = mask_path
        
        if mask_path and image_shape:
            self.load_mask(mask_path, image_shape)
    
    def load_mask(self, mask_path, image_shape):
        """Load validation mask"""
        if not os.path.exists(mask_path):
            print(f"Warning: Validation mask not found: {mask_path}")
            return False
        
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print(f"Error: Could not load validation mask: {mask_path}")
            return False
        
        # Resize if needed
        if mask_img.shape != image_shape:
            mask_img = cv2.resize(mask_img, (image_shape[1], image_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        self.mask = mask_img > 127
        print(f"Validation mask loaded: {np.sum(self.mask)} validation pixels")
        return True
    
    def is_in_validation_area(self, x, y):
        """
        Check if position is within validation area.
        MATLAB equivalent: if isempty(mask) || mask(pos(2), pos(1))
        
        Args:
            x, y: Position in pixels
            
        Returns:
            bool: True if in validation area (or no mask = count everywhere)
        """
        if self.mask is None:
            return True  # No mask = count everywhere
        
        h, w = self.mask.shape
        y_int = int(np.clip(round(y), 0, h - 1))
        x_int = int(np.clip(round(x), 0, w - 1))
        
        return bool(self.mask[y_int, x_int])
    
    def is_active(self):
        """Check if validation filter is active"""
        return self.mask is not None


# Utility function for quick distance field visualization
def visualize_mangrove_mask(mask_path, output_path="mangrove_distance_vis.png"):
    """
    Quick utility to visualize a mangrove mask's distance field.
    
    Args:
        mask_path: Path to mask PNG
        output_path: Where to save visualization
    """
    # Load mask to get shape
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"Error: Could not load {mask_path}")
        return
    
    # Create mapper and visualize
    mapper = MangroveDistanceMapper(mask_path, mask_img.shape[:2])
    if mapper.is_active():
        mapper.visualize_distance_field(output_path)
        print(f"Visualization complete: {output_path}")
    else:
        print("Failed to create distance field")


if __name__ == "__main__":
    # Demo/test
    print("Mangrove Distance Mapper - Demo")
    print("=" * 50)
    
    # Create test mask
    test_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_mask, (50, 50), (150, 150), 255, -1)  # 100x100 square
    cv2.imwrite("test_mangrove_mask.png", test_mask)
    print("Created test mask: test_mangrove_mask.png")
    
    # Test mapper
    mapper = MangroveDistanceMapper("test_mangrove_mask.png", (200, 200))
    
    if mapper.is_active():
        # Test some positions
        print("\nTesting distance calculations:")
        test_positions = [
            (100, 100, "Center of mangrove (inside)"),
            (100, 30, "Above mangrove (outside)"),
            (100, 50, "Top edge"),
            (30, 100, "Left side (outside)"),
        ]
        
        for x, y, desc in test_positions:
            dist = mapper.get_distance_at_position(x, y)
            print(f"   {desc:30s}: {dist:+7.1f} pixels")
        
        # Create visualization
        mapper.visualize_distance_field("test_distance_field.png")
        print("\nVisualization saved: test_distance_field.png")
        
        # Test area calculation
        area = mapper.get_mangrove_area_cm2(pix_scale=0.01)  # 1cm per pixel
        print(f"\nMangrove area: {area:.1f} cm²")
    
    print("\nDemo complete!")

