#!/usr/bin/env python3
"""
FINAL CLEAN ARIS READER: Raw ARIS Data + MATLAB Fan-shaped Mapping
"""

import numpy as np
import matplotlib.pyplot as plt

class FinalCleanARISReader:
    """Clean ARIS reader for final output"""
    
    def __init__(self, filename):
        self.filename = filename
        self.cap = None
        self.version = None
        self.numframes = 0
        self.framerate = 0.0
        self.resolution = 0
        self.numbeams = 0
        self.samplerate = 0.0
        self.sampleperchannel = 0
        self.receivergain = 0
        self.windowstart = 0.0
        self.windowlength = 0.0
        self.reverse = 0
        self.serialnumber = 0
        
    def open(self):
        """Open file using the version-aware runtime reader."""
        if self.cap is None:
            from opencv_player import CapMultiThreading
            self.cap = CapMultiThreading(self.filename)

        info = self.cap.file_info
        self.version = info.get('version', 0)
        self.numframes = info.get('numframes', 0)
        self.framerate = info.get('framerate', 0.0)
        self.resolution = info.get('resolution', 0)
        self.numbeams = info.get('numbeams', 0)
        self.samplerate = info.get('samplerate', 0.0)
        self.sampleperchannel = info.get('sampleperchannel', 0)
        self.receivergain = info.get('receivergain', 0)
        self.windowstart = info.get('windowstart', 0.0)
        self.windowlength = info.get('windowlength', 0.0)
        self.reverse = info.get('reverse', 0)
        self.serialnumber = info.get('serialnumber', 0)
        return self
        
    def get_frame_first(self):
        """Get first frame - MATLAB equivalent"""
        if not self.cap:
            self.open()

        frame_info = self.cap.get_frame_new(1)
        frame = frame_info['frame']
        minrange = frame_info['minrange']
        maxrange = frame_info['maxrange']
            
        return {
            'frame': frame,
            'numbeams': self.numbeams,
            'sampleperchannel': self.sampleperchannel,
            'minrange': minrange,
            'maxrange': maxrange,
            'version': self.version
        }
    
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class FinalCleanMapper:
    """Clean coordinate mapping for fan-shaped output"""
    
    @staticmethod
    def lens_distortion(nbeams, theta):
        """ARIS lens distortion correction"""
        original_beams = nbeams
        
        if nbeams > 200:
            test_orig_4 = (nbeams + 3) / 4
            if abs(test_orig_4 - round(test_orig_4)) < 0.01:
                original_beams = round(test_orig_4)
            else:
                test_orig_8 = (nbeams + 7) / 8
                if abs(test_orig_8 - round(test_orig_8)) < 0.01:
                    original_beams = round(test_orig_8)
        
        if original_beams == 128:
            factor = 1.35
            a = [0.0030, -0.0055, 2.6829, 48.04]
        elif original_beams == 96:
            factor = 1.012
            a = [0.0030, -0.0055, 2.6829, 48.04]
        elif original_beams == 48:
            factor = 1.0
            a = [0.0015, -0.0036, 1.3351, 24.0976]
        else:
            halffov = 14.0
            c2 = (nbeams - 1) / (2 * halffov)
            beamnum = np.fix((theta + halffov) * c2 + 1.5).astype(int)
            return np.clip(beamnum, 1, nbeams)
        
        beam_scale = nbeams / original_beams
        beamnum = np.round(factor * beam_scale * (
            a[0] * theta**3 + a[1] * theta**2 + a[2] * theta + a[3]
        ) + 1).astype(int)
        
        return np.clip(beamnum, 1, nbeams)
    
    @staticmethod
    def expand4(inframe8):
        """4x beam interpolation"""
        m, n = inframe8.shape
        nout = 4*n - 3
        outframe = np.zeros((m, nout))
        inframe = inframe8.astype(np.float64)
        
        outframe[:, 0::4] = inframe
        if nout >= 4:
            outframe[:, 1:nout-3:4] = 0.75*inframe[:, 0:n-1] + 0.25*inframe[:, 1:n]
        if nout >= 3:
            outframe[:, 2:nout-2:4] = 0.50*inframe[:, 0:n-1] + 0.50*inframe[:, 1:n]
        if nout >= 2:
            outframe[:, 3:nout-1:4] = 0.25*inframe[:, 0:n-1] + 0.75*inframe[:, 1:n]
        
        return outframe.astype(np.uint8)
    
    @staticmethod
    def smooth1(inarray, factor, method='expand'):
        """
        Python equivalent of smooth1.m
        Expands input sample array by interpolating "virtual beams" between real ones.
        
        Args:
            inarray: Input frame from data file (512 x n_beams)
            factor: Expansion factor (4 or 8) - output spacing of adjacent real beams
            method: Interpolation method ('expand', 'linear', 'cubic', 'nearest')
            
        Returns:
            outarray: Expanded array with interpolated beams
        """
        import numpy as np
        from scipy import interpolate
        
        # Use custom expand4 for the expand method with factor 4
        if method == 'expand' and factor == 4:
            return FinalCleanMapper.expand4(inarray)
        
        # For other methods, use scipy interpolation
        m, n = inarray.shape
        nout = n * factor - factor + 1  # Output width dimension
        outarray = np.zeros((512, nout), dtype=np.uint8)
        
        # Convert to double for interpolation
        inarray_double = inarray.astype(np.float64)
        
        # Original beam positions
        x_original = np.arange(1, n + 1)
        
        # New interpolated positions
        if factor == 4:
            x_new = np.arange(1, n + 1, 0.25)
        elif factor == 8:
            x_new = np.arange(1, n + 1, 0.125)
        else:
            # General case
            step = 1.0 / factor
            x_new = np.arange(1, n + 1, step)
        
        # Ensure we don't exceed the original range
        x_new = x_new[x_new <= n]
        
        # Interpolate each range bin (row)
        for row_idx in range(512):
            row_data = inarray_double[row_idx, :]
            
            if method == 'linear':
                interpolated = np.interp(x_new, x_original, row_data)
            elif method == 'cubic':
                # Use cubic spline interpolation
                if len(x_original) >= 4:  # Need at least 4 points for cubic
                    f = interpolate.interp1d(x_original, row_data, kind='cubic', 
                                           fill_value='extrapolate')
                    interpolated = f(x_new)
                else:
                    # Fall back to linear if not enough points
                    interpolated = np.interp(x_new, x_original, row_data)
            elif method == 'nearest':
                # Nearest neighbor interpolation
                f = interpolate.interp1d(x_original, row_data, kind='nearest', 
                                       fill_value='extrapolate')
                interpolated = f(x_new)
            else:
                # Default to linear
                interpolated = np.interp(x_new, x_original, row_data)
            
            # Store interpolated data (truncate to fit output array)
            n_output = min(len(interpolated), nout)
            outarray[row_idx, :n_output] = np.clip(interpolated[:n_output], 0, 255).astype(np.uint8)
        
        return outarray
    
    @staticmethod
    def mapscan(ixsize, rmax, rmin, halffov, nbeams, nbins):
        """MATLAB mapscan equivalent with clean cone edges"""
        degtorad = 3.14159/180.0
        radtodeg = 180.0/3.14159
        
        # Safety check for invalid range
        if rmax <= rmin or (rmax - rmin) <= 0:
            raise ValueError(f"Invalid range: rmin={rmin:.4f}, rmax={rmax:.4f}. Range difference must be > 0")
        
        d2 = rmax*np.cos(halffov*degtorad)
        d3 = rmin*np.cos(halffov*degtorad)
        c1 = (nbins-1)/(rmax-rmin)
        
        gamma = ixsize/(2*rmax*np.sin(halffov*degtorad))
        iysize = int(gamma*(rmax - d3) + 0.5)
        svector = np.zeros(ixsize*iysize, dtype=int)
        
        ix = np.arange(1, ixsize+1, dtype=np.float64)
        x = ((ix-1) - ixsize/2)/gamma
        
        for iy in range(1, iysize+1):
            y = rmax - (iy-1)/gamma
            r = np.sqrt(y**2 + x**2)
            theta = radtodeg*np.arctan2(x, y)
            
            binnum = np.fix((r - rmin)*c1 + 1.5).astype(int)
            beamnum = FinalCleanMapper.lens_distortion(nbeams, theta)
            
            # Clean cone edges: strict field of view limits
            valid_mask = (beamnum > 0) & (beamnum <= nbeams) & (binnum > 0) & (binnum <= nbins)
            angle_mask = np.abs(theta) <= halffov
            range_mask = (r >= rmin) & (r <= rmax)
            final_mask = valid_mask & angle_mask & range_mask
            
            pos = np.zeros_like(beamnum, dtype=int)
            pos[final_mask] = (beamnum[final_mask] - 1) * nbins + binnum[final_mask]
            
            for i in range(len(ix)):
                idx = int((ix[i]-1)*iysize + iy - 1)
                if 0 <= idx < len(svector):
                    svector[idx] = pos[i]
        
        svector[svector == 0] = 1
        
        # Calculate mapscale for pixel-to-meter conversion (MATLAB equivalent)
        # These parameters convert from pixel space to meter space relative to transducer
        halffov_rad = halffov * degtorad
        ws = 2 * rmax * np.sin(14.25 * degtorad) / ixsize  # width scale (meters/pixel)
        hs = (rmax - rmin * np.cos(14.25 * degtorad)) / iysize  # height scale (meters/pixel)
        i0 = rmax / hs  # Y origin: transducer position in pixel space
        j0 = ixsize / 2  # X origin: center of image in pixel space
        mapscale = [hs, ws, i0, j0]  # [height_scale, width_scale, y_origin, x_origin]
        
        return {
            'iysize': iysize,
            'svector': svector,
            'ixsize': ixsize,
            'mapscale': mapscale,
            'minrange': rmin,
            'maxrange': rmax
        }
    
    @staticmethod
    def make_first_image(data, smooth=4, imagexsize=400):
        """Create MATLAB-equivalent fan-shaped image"""
        frame = data['frame'].copy()
        
        if smooth > 1:
            frame = FinalCleanMapper.expand4(frame)
            
        frame[0, 0] = 0
        
        nrows = data['numbeams']*smooth - smooth + 1
        half_angle = 14.0
        
        map_data = FinalCleanMapper.mapscan(
            imagexsize, data['maxrange'], data['minrange'], 
            half_angle, nrows, data['sampleperchannel']
        )
        
        frame_flat = frame.flatten(order='F')
        svector_values = frame_flat[map_data['svector'] - 1]
        image = svector_values.reshape(map_data['iysize'], imagexsize, order='F')
        
        return image.astype(np.uint8)

    @staticmethod
    def get_aris_beam_from_angle(angles):
        """
        Python equivalent of get_aris_beam_from_angle.m
        Converts angles to beam numbers using ARIS distortion pattern
        
        Args:
            angles: Array of angles in degrees
            
        Returns:
            beams: Array of beam numbers corresponding to input angles
        """
        import numpy as np
        
        # ARIS distortion pattern from MATLAB
        # Format: [beam_number, left_angle, center_angle, right_angle]
        aris_distortion_pattern = np.array([
            [0, -13.661, -13.762, -13.553],
            [1, -13.445, -13.553, -13.337],
            [2, -13.229, -13.337, -13.112],
            [3, -12.995, -13.112, -12.878],
            [4, -12.761, -12.878, -12.644],
            [5, -12.527, -12.644, -12.410],
            [6, -12.293, -12.410, -12.176],
            [7, -12.058, -12.176, -11.932],
            [8, -11.806, -11.932, -11.680],
            [9, -11.554, -11.680, -11.428],
            [10, -11.302, -11.428, -11.176],
            [11, -11.050, -11.176, -10.924],
            [12, -10.798, -10.924, -10.672],
            [13, -10.545, -10.672, -10.419],
            [14, -10.293, -10.419, -10.167],
            [15, -10.041, -10.167, -9.906],
            [16, -9.771, -9.906, -9.636],
            [17, -9.501, -9.636, -9.366],
            [18, -9.231, -9.366, -9.096],
            [19, -8.960, -9.096, -8.825],
            [20, -8.690, -8.825, -8.546],
            [21, -8.402, -8.546, -8.258],
            [22, -8.114, -8.258, -7.970],
            [23, -7.826, -7.970, -7.682],
            [24, -7.538, -7.682, -7.393],
            [25, -7.249, -7.393, -7.105],
            [26, -6.961, -7.105, -6.808],
            [27, -6.655, -6.808, -6.502],
            [28, -6.349, -6.502, -6.196],
            [29, -6.043, -6.196, -5.890],
            [30, -5.736, -5.890, -5.583],
            [31, -5.430, -5.583, -5.277],
            [32, -5.124, -5.277, -4.971],
            [33, -4.818, -4.971, -4.656],
            [34, -4.494, -4.656, -4.332],
            [35, -4.169, -4.332, -4.007],
            [36, -3.845, -4.007, -3.683],
            [37, -3.521, -3.683, -3.359],
            [38, -3.197, -3.359, -3.035],
            [39, -2.873, -3.035, -2.710],
            [40, -2.548, -2.710, -2.386],
            [41, -2.224, -2.386, -2.053],
            [42, -1.882, -2.053, -1.711],
            [43, -1.540, -1.711, -1.369],
            [44, -1.198, -1.369, -1.026],
            [45, -0.855, -1.026, -0.684],
            [46, -0.513, -0.684, -0.342],
            [47, -0.171, -0.342, 0.000],
            [48, 0.171, 0.000, 0.342],
            [49, 0.513, 0.342, 0.684],
            [50, 0.855, 0.684, 1.026],
            [51, 1.198, 1.026, 1.369],
            [52, 1.540, 1.369, 1.711],
            [53, 1.882, 1.711, 2.053],
            [54, 2.224, 2.053, 2.386],
            [55, 2.548, 2.386, 2.710],
            [56, 2.873, 2.710, 3.035],
            [57, 3.197, 3.035, 3.359],
            [58, 3.521, 3.359, 3.683],
            [59, 3.845, 3.683, 4.007],
            [60, 4.169, 4.007, 4.332],
            [61, 4.494, 4.332, 4.656],
            [62, 4.818, 4.656, 4.971],
            [63, 5.124, 4.971, 5.277],
            [64, 5.430, 5.277, 5.583],
            [65, 5.736, 5.583, 5.890],
            [66, 6.043, 5.890, 6.196],
            [67, 6.349, 6.196, 6.502],
            [68, 6.655, 6.502, 6.808],
            [69, 6.961, 6.808, 7.105],
            [70, 7.249, 7.105, 7.393],
            [71, 7.538, 7.393, 7.682],
            [72, 7.826, 7.682, 7.970],
            [73, 8.114, 7.970, 8.258],
            [74, 8.402, 8.258, 8.546],
            [75, 8.690, 8.546, 8.825],
            [76, 8.960, 8.825, 9.096],
            [77, 9.231, 9.096, 9.366],
            [78, 9.501, 9.366, 9.636],
            [79, 9.771, 9.636, 9.906],
            [80, 10.041, 9.906, 10.167],
            [81, 10.293, 10.167, 10.419],
            [82, 10.545, 10.419, 10.672],
            [83, 10.798, 10.672, 10.924],
            [84, 11.050, 10.924, 11.176],
            [85, 11.302, 11.176, 11.428],
            [86, 11.554, 11.428, 11.680],
            [87, 11.806, 11.680, 11.932],
            [88, 12.058, 11.932, 12.176],
            [89, 12.293, 12.176, 12.410],
            [90, 12.527, 12.410, 12.644],
            [91, 12.761, 12.644, 12.878],
            [92, 12.995, 12.878, 13.112],
            [93, 13.229, 13.112, 13.337],
            [94, 13.445, 13.337, 13.553],
            [95, 13.661, 13.553, 13.762]
        ])
        
        # Convert angles to array if not already
        angles = np.asarray(angles)
        beams = np.zeros(len(angles), dtype=int)
        
        # For each angle, find the corresponding beam
        for j, angle in enumerate(angles):
            # Find the beam where angle falls within the range
            for i, (beam_num, left_angle, center_angle, right_angle) in enumerate(aris_distortion_pattern):
                if angle < right_angle:
                    beams[j] = i + 1  # +1 to match MATLAB's 1-based indexing
                    break
            else:
                # If angle is beyond the last beam's range
                beams[j] = len(aris_distortion_pattern)  # No -1 needed now
                
        return beams

def process_aris_file(filename):
    """Process ARIS file and return raw data + fan-shaped mapping"""
    
    print("📡 FINAL CLEAN ARIS READER")
    print("=" * 50)
    
    try:
        reader = FinalCleanARISReader(filename)
        
        # Get raw ARIS data
        data = reader.get_frame_first()
        raw_frame = data['frame']
        
        # Create fan-shaped mapping
        fan_image = FinalCleanMapper.make_first_image(data, smooth=4, imagexsize=400)
        
        # Use upward-opening orientation (original, no flip)
        upward_fan = fan_image  # Based on testing, original orientation opens upward
        
        print(f"Raw ARIS data: {raw_frame.shape} (Range × Beam)")
        print(f"Fan mapping: {upward_fan.shape} (Range × Cross-range)")
        print(f"   File: {data['numbeams']} beams, {data['sampleperchannel']} samples")
        print(f"   Range: {data['minrange']:.3f}m to {data['maxrange']:.3f}m")
        
        reader.close()
        
        return {
            'raw_data': raw_frame,
            'fan_image': upward_fan,
            'info': {
                'numbeams': data['numbeams'],
                'sampleperchannel': data['sampleperchannel'],
                'minrange': data['minrange'],
                'maxrange': data['maxrange']
            }
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def display_final_results(result, save_prefix="final_aris"):
    """Display the two final outputs: raw data + fan mapping"""
    
    if result is None:
        print("ERROR: No results to display")
        return
    
    raw_data = result['raw_data']
    fan_image = result['fan_image']
    info = result['info']
    
    # Create final clean display
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Raw ARIS Data (Polar: Range × Beam)
    axes[0].imshow(raw_data, cmap='bone', aspect='auto', vmin=30, vmax=200,
                   interpolation='none', origin='upper')
    axes[0].set_title('Raw ARIS Data\n(Polar: Range × Beam)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Beam', fontsize=12)
    axes[0].set_ylabel('Range Sample', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # MATLAB Fan-shaped Mapping (Cartesian)
    axes[1].imshow(fan_image, cmap='bone', vmin=30, vmax=200,
                   interpolation='none', origin='upper', aspect='auto')
    axes[1].set_title('MATLAB Mapping\n(Fan-shaped Cartesian)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cross-range (pixels)', fontsize=12)
    axes[1].set_ylabel('Range (pixels)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add info text
    info_text = (f"ARIS Data Summary:\n"
                f"• {info['numbeams']} beams\n"
                f"• {info['sampleperchannel']} samples/beam\n"
                f"• Range: {info['minrange']:.3f}m - {info['maxrange']:.3f}m\n"
                f"• Raw shape: {raw_data.shape}\n"
                f"• Fan shape: {fan_image.shape}")
    
    fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_complete.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save individual images
    plt.figure(figsize=(10, 8))
    plt.imshow(raw_data, cmap='bone', aspect='auto', vmin=30, vmax=200,
               interpolation='none', origin='upper')
    plt.title('Raw ARIS Data (Polar: Range × Beam)', fontsize=14)
    plt.xlabel('Beam')
    plt.ylabel('Range Sample')
    plt.colorbar(label='Intensity')
    plt.savefig(f'{save_prefix}_raw_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(fan_image, cmap='bone', vmin=30, vmax=200,
               interpolation='none', origin='upper', aspect='auto')
    plt.title('MATLAB Fan-shaped Mapping (Cartesian)', fontsize=14)
    plt.xlabel('Cross-range (pixels)')
    plt.ylabel('Range (pixels)')
    plt.colorbar(label='Intensity')
    plt.savefig(f'{save_prefix}_fan_mapping.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Complete view: {save_prefix}_complete.png")
    print(f"Raw data: {save_prefix}_raw_data.png")
    print(f"Fan mapping: {save_prefix}_fan_mapping.png")

if __name__ == "__main__":
    filename = "../ARIS Data/DH1112_2015-11-08_115514.aris"
    
    # Process ARIS file
    result = process_aris_file(filename)
    
    if result is not None:
        # Display final clean results
        display_final_results(result, save_prefix="final_clean_aris")
        
        print(f"\nFINAL CLEAN ARIS READER SUCCESS!")
        print(f"   Raw ARIS Data: {result['raw_data'].shape} (polar coordinates)")
        print(f"   Fan Mapping: {result['fan_image'].shape} (cartesian coordinates)")
        print(f"   Ready for analysis and display!")
    else:
        print("ERROR: Processing failed") 