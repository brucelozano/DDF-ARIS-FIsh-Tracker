#!/usr/bin/env python3
"""
ARIS Python Player - Main Entry Point
Complete ARIS reader with all MATLAB equivalent functions
"""

import os
import sys
from opencv_player import OpenCVPlayer
from aris_python_api import CompleteARISReader

VIDEO_EXTENSIONS = (".aris", ".ddf")
DEFAULT_SCAN_ROOTS = (".", "..")


def _is_supported_video_file(filename):
    """Check if file has a supported ARIS/DDF extension."""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def _display_path(path):
    """Show path relative to current working directory when possible."""
    try:
        relative = os.path.relpath(path, os.getcwd())
        return "." if relative == "." else relative
    except ValueError:
        return path


def find_aris_files(directory=".", recursive=True):
    """Find ARIS/DDF files in a directory."""
    normalized_dir = os.path.abspath(directory)
    if not os.path.isdir(normalized_dir):
        return []

    video_files = []

    if recursive:
        skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
        for root, dirs, files in os.walk(normalized_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for file_name in files:
                if _is_supported_video_file(file_name):
                    video_files.append(os.path.join(root, file_name))
    else:
        for file_name in os.listdir(normalized_dir):
            file_path = os.path.join(normalized_dir, file_name)
            if os.path.isfile(file_path) and _is_supported_video_file(file_name):
                video_files.append(file_path)

    return sorted(set(video_files))


def discover_video_folders(search_roots=None):
    """Discover folders that contain ARIS/DDF files."""
    if search_roots is None:
        search_roots = DEFAULT_SCAN_ROOTS

    folder_map = {}
    seen_files = set()

    for root in search_roots:
        for file_path in find_aris_files(root, recursive=True):
            abs_path = os.path.abspath(file_path)
            if abs_path in seen_files:
                continue
            seen_files.add(abs_path)
            folder = os.path.dirname(abs_path)
            folder_map.setdefault(folder, []).append(abs_path)

    for file_list in folder_map.values():
        file_list.sort()

    return dict(
        sorted(
            folder_map.items(),
            key=lambda item: _display_path(item[0]).lower()
        )
    )


def _select_file_from_folder(folder_path, file_list):
    """Prompt for file selection within one folder."""
    folder_display = _display_path(folder_path)
    print(f"\nFiles in folder: {folder_display}")
    print("-" * 40)

    for i, file_path in enumerate(file_list, 1):
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {i}. {os.path.basename(file_path)} ({file_size:.1f} MB)")

    print("  B. Back to folder list")
    print("  0. Cancel")

    while True:
        choice = input(f"\nSelect file (1-{len(file_list)}, B, 0): ").strip()

        if choice == '0':
            return None
        if choice.lower() == 'b':
            return "__BACK__"

        try:
            index = int(choice) - 1
        except ValueError:
            print("Please enter a number, B, or 0!")
            continue

        if 0 <= index < len(file_list):
            return file_list[index]

        print("Invalid selection!")

def select_aris_file():
    """Interactive ARIS/DDF file selection with folder-first workflow."""
    print("\nARIS/DDF File Selection")
    print("=" * 30)

    folder_map = discover_video_folders()
    folder_paths = list(folder_map.keys())

    while True:
        if folder_paths:
            print("\nAvailable folders with ARIS/DDF files:")
            for i, folder_path in enumerate(folder_paths, 1):
                print(f"  {i}. {_display_path(folder_path)} ({len(folder_map[folder_path])} files)")
            print(f"  {len(folder_paths)+1}. Enter custom folder path")
            print(f"  {len(folder_paths)+2}. Enter full file path")
            print("  0. Cancel")

            choice = input(f"\nSelect option (0-{len(folder_paths)+2}): ").strip()

            if choice == '0':
                return None

            if choice == str(len(folder_paths) + 1):
                custom_folder = input("Enter folder path: ").strip()
                if not custom_folder:
                    print("No folder entered.")
                    continue
                if not os.path.isdir(custom_folder):
                    print("Folder not found!")
                    continue

                files_in_folder = find_aris_files(custom_folder, recursive=False)
                if not files_in_folder:
                    files_in_folder = find_aris_files(custom_folder, recursive=True)
                    if files_in_folder:
                        print("No files directly in folder. Showing files found in subfolders.")
                    else:
                        print("No ARIS/DDF files found in that folder.")
                        continue

                selected = _select_file_from_folder(os.path.abspath(custom_folder), files_in_folder)
                if selected == "__BACK__":
                    continue
                return selected

            if choice == str(len(folder_paths) + 2):
                custom_path = input("Enter full path to ARIS/DDF file: ").strip()
                if not custom_path:
                    print("No path entered.")
                    continue
                if not os.path.exists(custom_path):
                    print("File not found!")
                    continue
                if not _is_supported_video_file(custom_path):
                    print("Warning: file does not end with .aris or .ddf")
                return custom_path

            try:
                folder_index = int(choice) - 1
            except ValueError:
                print("Please enter a number!")
                continue

            if not (0 <= folder_index < len(folder_paths)):
                print("Invalid selection!")
                continue

            selected_folder = folder_paths[folder_index]
            selected = _select_file_from_folder(selected_folder, folder_map[selected_folder])
            if selected == "__BACK__":
                continue
            return selected

        else:
            print("No ARIS/DDF folders found automatically.")
            print("  1. Enter custom folder path")
            print("  2. Enter full file path")
            print("  0. Cancel")
            choice = input("\nSelect option (0-2): ").strip()

            if choice == '0':
                return None
            if choice == '1':
                custom_folder = input("Enter folder path: ").strip()
                if not custom_folder or not os.path.isdir(custom_folder):
                    print("Folder not found!")
                    continue
                files_in_folder = find_aris_files(custom_folder, recursive=True)
                if not files_in_folder:
                    print("No ARIS/DDF files found in that folder.")
                    continue
                selected = _select_file_from_folder(os.path.abspath(custom_folder), files_in_folder)
                if selected == "__BACK__":
                    continue
                return selected
            if choice == '2':
                custom_path = input("Enter full path to ARIS/DDF file: ").strip()
                if custom_path and os.path.exists(custom_path):
                    return custom_path
                print("File not found!")
                continue

            print("Invalid selection!")
            continue

def show_menu():
    """Display main menu options"""
    print("ARIS/DDF Python Reader - Complete MATLAB Equivalent")
    print("=" * 60)
    print("Available operations:")
    print("  1. Play ARIS/DDF movie (interactive viewer)")
    print("  2. Run MATLAB workflow demo") 
    print("  3. Export to AVI video")
    print("  4. Export first frame to images (MATLAB equivalent)")
    print("  5. Batch process directory")
    print("  6. Test all functions")
    print("  0. Exit")
    print("-" * 60)


def _fan_geometry_from_config(config):
    """Extract sanitized fan-geometry values from a FishTrackingConfig."""
    imagexsize = max(32, int(getattr(config, 'FAN_IMAGE_X_SIZE', 400)))
    smooth = max(1, int(getattr(config, 'FAN_BEAM_SMOOTH', 4)))
    half_angle = float(getattr(config, 'FAN_HALF_ANGLE_DEG', 14.0))
    return imagexsize, smooth, half_angle


def _compute_and_save_pattern_for_file(video_file, output_dir, imagexsize, smooth, half_angle):
    """Compute and save one static pattern, returning output paths."""
    from aris_utilities import ARISExporter

    base_name = os.path.splitext(os.path.basename(video_file))[0]
    exporter = ARISExporter(video_file)
    try:
        pattern = exporter.compute_static_pattern(
            imagexsize=imagexsize,
            smooth=smooth,
            half_angle=half_angle
        )
        saved = exporter.save_static_pattern(output_dir, pattern, base_name=base_name, also_mat=True)
    finally:
        exporter.close()

    return saved


def _build_batch_pattern_file_list(folder_path, recursive=True):
    """Get .ddf files for batch pattern generation."""
    candidates = find_aris_files(folder_path, recursive=recursive)
    return [path for path in candidates if path.lower().endswith('.ddf')]


def _default_pattern_path_for_video(video_file):
    """Return the most likely pattern file path for a video, if it exists."""
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    npy_path = os.path.join('exports', 'patterns', f"{base_name}_pattern.npy")
    mat_path = os.path.join('exports', 'patterns', f"{base_name}_pattern.mat")
    if os.path.exists(npy_path):
        return npy_path
    if os.path.exists(mat_path):
        return mat_path
    return npy_path


def _load_pattern_array(pattern_path):
    """Load a static pattern from .npy or .mat into a 2D float32 array."""
    import numpy as np

    if pattern_path.lower().endswith('.npy'):
        pattern = np.load(pattern_path)
    elif pattern_path.lower().endswith('.mat'):
        from scipy.io import loadmat
        mat = loadmat(pattern_path)
        if 'Pattern' not in mat:
            raise ValueError("MAT file does not contain 'Pattern'")
        pattern = mat['Pattern']
    else:
        raise ValueError("Unsupported pattern format. Use .npy or .mat")

    pattern = np.asarray(pattern, dtype=np.float32)
    if pattern.ndim == 3 and pattern.shape[-1] == 1:
        pattern = pattern[:, :, 0]
    pattern = np.squeeze(pattern)
    if pattern.ndim != 2:
        raise ValueError(f"Expected 2D pattern, got shape {pattern.shape}")
    return pattern


def _save_pattern_png(pattern_path):
    """
    Save one PNG rendering of the pattern in the same directory.
    Returns saved PNG path.
    """
    import numpy as np
    import cv2

    pattern = _load_pattern_array(pattern_path)

    finite = np.isfinite(pattern)
    if not finite.any():
        raise ValueError("Pattern has no finite values")
    p = pattern.copy()
    p[~finite] = 0.0
    pmin = float(np.min(p[finite]))
    pmax = float(np.max(p[finite]))
    pmean = float(np.mean(p[finite]))

    if pmax <= 1.0 and pmin >= 0.0:
        p_u8 = np.clip(np.rint(p * 255.0), 0, 255).astype(np.uint8)
    elif pmax > 255.0 or pmin < 0.0:
        denom = pmax - pmin
        if denom <= 1e-9:
            p_u8 = np.zeros_like(p, dtype=np.uint8)
        else:
            p_norm = (p - pmin) / denom
            p_u8 = np.clip(np.rint(p_norm * 255.0), 0, 255).astype(np.uint8)
    else:
        p_u8 = np.clip(np.rint(p), 0, 255).astype(np.uint8)

    png_path = os.path.splitext(pattern_path)[0] + ".png"
    cv2.imwrite(png_path, p_u8)

    print("Pattern PNG stats:")
    print(f"   Shape: {pattern.shape}")
    print(f"   Min/Max: {pmin:.6f} / {pmax:.6f}")
    print(f"   Mean: {pmean:.6f}")
    print(f"   Saved PNG: {_display_path(png_path)}")

    return png_path


def _list_pattern_files_in_folder(folder_path):
    """
    Return one source pattern file per stem in folder.
    Preference: .npy over .mat when both exist.
    """
    if not os.path.isdir(folder_path):
        return []

    npy_by_stem = {}
    mat_by_stem = {}
    for entry in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, entry)
        if not os.path.isfile(full_path):
            continue
        lower = entry.lower()
        stem, ext = os.path.splitext(entry)
        if ext == ".npy":
            npy_by_stem[stem] = full_path
        elif ext == ".mat":
            mat_by_stem[stem] = full_path

    stems = sorted(set(npy_by_stem) | set(mat_by_stem))
    selected = []
    for stem in stems:
        # Prefer .npy if both are available.
        selected.append(npy_by_stem.get(stem) or mat_by_stem.get(stem))
    return selected


def main():
    print("ARIS/DDF Python Reader - Interactive Menu")
    print("=" * 50)
    
    # Check for command line argument first
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print(f"ERROR: File not found: {filename}")
            return
        if not filename.lower().endswith(('.aris', '.ddf')):
            print(f"WARNING: File doesn't have .aris/.ddf extension: {filename}")
        print(f"Using command line file: {os.path.basename(filename)}")
    else:
        # Auto-detect ARIS/DDF files from current project tree
        candidates = find_aris_files(".")
        if not candidates:
            # Also look one level up for users launching from a subfolder
            candidates = find_aris_files("..")
        filename = candidates[0] if candidates else None
        
        if not filename:
            print("ERROR: No ARIS/DDF files found!")
            print("Usage options:")
            print("  1. python aris_reader.py /full/path/to/your_file.aris")
            print("     (or .ddf)")
            print("  2. Use option 9 to select a file interactively")
            print("  3. Place an .aris/.ddf file somewhere under this project")
            return
        
        print(f"Using auto-detected file: {os.path.basename(filename)}")
    
    print()
    
    while True:
        try:
            print("\nAvailable Options:")
            print("  1. Play movie (with fish tracking + auto-export CSVs)")
            print("  2. Export to AVI video")
            print("  3. Export first frame to images (MATLAB equivalent)")
            print("  4. Batch processing demo")
            print("  5. Test all MATLAB functions")
            print("  6. Generate time index (.mat + .csv) for directory")
            print("  7. Test beam angle conversion")
            print("  8. Test advanced interpolation (smooth1)")
            print("  9. Select different ARIS/DDF file")
            print("  V. Visualize Tracking Data (from CSV results)")
            print("  P. Compute static pattern (single or batch DDF)")
            print("  0. Exit")
            print()
            
            choice = input("Select option (0-9, V, P): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
                
            elif choice == '1':
                # Unified playback + export (MATLAB-style workflow)
                print("ARIS Movie Player with Fish Tracking + Auto-Export")
                print("=" * 60)
                print("MATLAB-style workflow: Play → Track → Export CSVs")
                print()
                
                # Simple fish detection toggle
                enable_fish = input("Enable fish detection? (Y/n): ").strip().lower() not in ['n', 'no']
                
                if enable_fish:
                    print("\nFish tracking enabled")
                    print("   • Dual-method detection (moving + bright objects)")
                    print("   • Static pattern support if configured")
                    print("   • Parameters from config_[videoname].json")
                    print("   • Press 'D' during playback to hide/show detection overlay")
                    print("   • Press 'E' to export CSVs anytime")
                    print("   • Auto-exports on exit (ESC)")
                else:
                    print("\n📹 Video playback only (no fish detection or export)")
                
                # Ask for frame range
                frame_range = input("\nEnter number of frames to process (default 500, or 'all' for entire video): ").strip().lower()
                
                # Determine last frame
                if frame_range == 'all':
                    # Get total frames from file
                    from opencv_player import CapMultiThreading
                    temp_cap = CapMultiThreading(filename)
                    total_frames = temp_cap.file_info['numframes']
                    temp_cap.release()
                    last_frame = total_frames
                    print(f"Processing all {total_frames} frames")
                else:
                    try:
                        last_frame = int(frame_range) if frame_range else 500
                    except ValueError:
                        last_frame = 500
                        print(f"Invalid input, using default: {last_frame}")
                
                # Use player with unified export
                print(f"\nStarting playback (frames 1-{last_frame})...")
                player = OpenCVPlayer(filename, enable_fish_detection=enable_fish, show_raw_window=False)
                player.run_player(first=1, last=last_frame)
                
            elif choice == '2':
                # Export to AVI - use fixed ARISExporter like MATLAB
                print("Exporting to AVI video...")
                print("   Using MATLAB-equivalent export (full file)")
                
                # Create filename prefix from input file
                base_name = os.path.splitext(os.path.basename(filename))[0]
                output_file = f"{base_name}_export.avi"
                
                # Use ARISExporter directly with fixed frame reading
                from aris_utilities import ARISExporter
                exporter = ARISExporter(filename)
                # MATLAB default: full file, cartesian type, 30 FPS
                exporter.generate_avi(output_file, display_type='cartesian', fps=30)
                exporter.close()
                
                print(f"AVI exported: {output_file}")
                print(f"   MATLAB-equivalent: full file, cartesian fan display")
                print(f"   Organized with filename prefix: {base_name}_")
                
            elif choice == '3':
                # Export first frame to images - use fixed ARISExporter
                print("Exporting first frame to images...")
                print("   Creating raw and fan-shaped images")
                
                # Create organized output directory with filename prefix
                base_name = os.path.splitext(os.path.basename(filename))[0]
                output_dir = f"{base_name}_first_frame"
                
                # Use ARISExporter directly with fixed frame reading  
                from aris_utilities import ARISExporter
                exporter = ARISExporter(filename)
                files = exporter.convert_to_images(frame_range=[1, 1], output_dir=output_dir)
                exporter.close()
                
                print(f"Images exported to: {output_dir}/")
                print(f"   Organized with filename prefix: {base_name}_")
                for file_path in files:
                    print(f"   - {os.path.basename(file_path)}")
                    
            elif choice == '4':
                # Batch processing demo - IMPROVED: Actually do batch processing
                print("Batch processing demo...")
                print("   Processing first 10 frames of all ARIS/DDF files in example_data/")
                
                try:
                    from aris_utilities import ARISBatchProcessor
                    
                    # Check if we have a directory with multiple files
                    data_dir = "example_data"
                    if os.path.exists(data_dir):
                        # Create organized output directory with timestamp
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = f"batch_demo_{timestamp}"
                        
                        processor = ARISBatchProcessor(data_dir)
                        results = processor.process_directory(
                            output_dir=output_dir,
                            export_type="images", 
                            frame_limit=10
                        )
                        print(f"Batch processing complete! Processed {len(results)} files.")
                        print(f"   Output organized in: {output_dir}/")
                    else:
                        print("   No example_data directory found for batch demo")
                        print("   Batch processing interface ready for your data")
                        
                except Exception as e:
                    print(f"   Batch demo error: {e}")
                    print("   Batch processing interface ready")
                
            elif choice == '5':
                # Test all functions (consolidated from old options 2 and 6)
                print("Testing all MATLAB equivalent functions...")
                test_all_functions(filename)

            elif choice.upper() == 'P':
                print("Static Pattern Generation")
                print("=" * 50)
                print("  1. Current file only")
                print("  2. All DDF files in a folder")
                print("  3. Export pattern PNG(s)")
                print("  0. Cancel")
                pattern_mode = input("Select pattern mode (0-3, default 1): ").strip()
                if pattern_mode == "":
                    pattern_mode = "1"
                if pattern_mode == "0":
                    print("Pattern generation cancelled.")
                    continue

                try:
                    from fish_tracking_config import get_config_for_video

                    # Use current file config as the geometry source for consistency.
                    # This avoids hardcoded values and keeps batch outputs comparable.
                    config = get_config_for_video(filename)
                    imagexsize, smooth, half_angle = _fan_geometry_from_config(config)
                    output_dir = os.path.join('exports', 'patterns')
                    if pattern_mode in {"1", "2"}:
                        print("MATLAB averages ALL frames for best results.")
                        print(f"Pattern geometry: width={imagexsize}, half_angle={half_angle:.2f}, smooth={smooth}")

                    if pattern_mode == "1":
                        saved = _compute_and_save_pattern_for_file(
                            filename, output_dir, imagexsize, smooth, half_angle
                        )
                        print(f"Saved static pattern files:\n  NPY: {saved.get('npy')}\n  MAT: {saved.get('mat', 'n/a')}")
                        print("Update your config JSON with:\n  \"USE_STATIC_PATTERN\": true,\n  \"STATIC_PATTERN_FILE\": \"" + saved.get('npy') + "\"  (or .mat)\n  \"BW_THRESHOLD\": 0.12,\n  \"BW_THRESHOLD_MODE\": \"relative\"")
                    elif pattern_mode == "2":
                        default_folder = os.path.dirname(os.path.abspath(filename)) or "."
                        folder_input = input(f"Folder to scan for DDF files [default: {default_folder}]: ").strip()
                        target_folder = folder_input if folder_input else default_folder
                        if not os.path.isdir(target_folder):
                            print(f"Error: folder not found: {target_folder}")
                            continue

                        recursive = input("Include subfolders? (Y/n): ").strip().lower() not in ["n", "no"]
                        ddf_files = _build_batch_pattern_file_list(target_folder, recursive=recursive)
                        if not ddf_files:
                            print("No .ddf files found in the selected folder.")
                            continue

                        print(f"Found {len(ddf_files)} DDF files. Starting batch pattern generation...")
                        print(f"Output directory: {_display_path(output_dir)}")
                        success_count = 0
                        failure_count = 0

                        for idx, ddf_file in enumerate(ddf_files, 1):
                            print(f"\n[{idx}/{len(ddf_files)}] Processing {_display_path(ddf_file)}")
                            try:
                                saved = _compute_and_save_pattern_for_file(
                                    ddf_file, output_dir, imagexsize, smooth, half_angle
                                )
                                success_count += 1
                                print(f"   Saved: {_display_path(saved.get('npy', ''))}")
                            except Exception as file_error:
                                failure_count += 1
                                print(f"   Failed: {file_error}")

                        print("\nBatch pattern generation complete.")
                        print(f"   Success: {success_count}")
                        print(f"   Failed: {failure_count}")
                    elif pattern_mode == "3":
                        print("Pattern PNG Export")
                        print("-" * 50)
                        print("  1. Single pattern file path")
                        print("  2. All pattern files in a folder")
                        print("  0. Cancel")
                        export_mode = input("Select export mode (0-2, default 1): ").strip()
                        if export_mode == "":
                            export_mode = "1"
                        if export_mode == "0":
                            print("Pattern PNG export cancelled.")
                            continue

                        if export_mode == "1":
                            default_pattern = _default_pattern_path_for_video(filename)
                            pattern_input = input(
                                f"Pattern file path [default: {default_pattern}]: "
                            ).strip()
                            pattern_path = pattern_input if pattern_input else default_pattern
                            if not os.path.exists(pattern_path):
                                print(f"Pattern file not found: {pattern_path}")
                                continue
                            _save_pattern_png(pattern_path)
                        elif export_mode == "2":
                            default_folder = os.path.dirname(_default_pattern_path_for_video(filename)) or os.path.join('exports', 'patterns')
                            folder_input = input(
                                f"Folder containing pattern files [default: {default_folder}]: "
                            ).strip()
                            target_folder = folder_input if folder_input else default_folder
                            if not os.path.isdir(target_folder):
                                print(f"Folder not found: {target_folder}")
                                continue

                            pattern_files = _list_pattern_files_in_folder(target_folder)
                            if not pattern_files:
                                print("No .npy/.mat pattern files found in that folder.")
                                continue

                            print(f"Found {len(pattern_files)} pattern files. Exporting PNGs...")
                            success_count = 0
                            failure_count = 0
                            for idx, pattern_file in enumerate(pattern_files, 1):
                                print(f"\n[{idx}/{len(pattern_files)}] {_display_path(pattern_file)}")
                                try:
                                    png_path = _save_pattern_png(pattern_file)
                                    success_count += 1
                                    print(f"   Wrote: {_display_path(png_path)}")
                                except Exception as export_error:
                                    failure_count += 1
                                    print(f"   Failed: {export_error}")

                            print("\nPattern PNG export complete.")
                            print(f"   Success: {success_count}")
                            print(f"   Failed: {failure_count}")
                        else:
                            print("Invalid export mode. Choose 0, 1, or 2.")
                    else:
                        print("Invalid pattern mode. Choose 0, 1, 2, or 3.")
                except Exception as e:
                    print(f"ERROR: Pattern compute/save failed: {e}")
                
            elif choice == '6':
                # Generate time index in both formats
                print("Generating time index (.mat + .csv formats)...")
                test_time_index_generation()
                
            elif choice == '7':
                # Test beam angle conversion
                print("Testing beam angle conversion...")
                test_beam_angle_conversion(filename)
                
            elif choice == '8':
                # Test advanced interpolation
                print("Testing advanced interpolation (smooth1)...")
                test_advanced_interpolation(filename)
                
            elif choice == '9':
                # Select different ARIS/DDF file
                print("File selection...")
                new_filename = select_aris_file()
                if new_filename:
                    filename = new_filename
                    print(f"\nSwitched to: {os.path.basename(filename)}")
                else:
                    print("File selection cancelled")
            
            elif choice.upper() == 'V':
                # Visualize Tracking Data
                print("Visualize Tracking Data - Analysis Results Viewer")
                print("=" * 55)
                print("Creates comprehensive visualizations from offline analysis:")
                print("  • Fish trajectory plots with velocity vectors")
                print("  • Activity heatmaps showing movement patterns") 
                print("  • Behavior analysis (velocity, size distributions)")
                print("  • Summary reports with statistics")
                print()
                
                try:
                    from simple_visualizer import interactive_visualizer
                    interactive_visualizer()
                except ImportError:
                    print("Error: simple_visualizer module not found")
                except Exception as e:
                    print(f"Error during visualization: {str(e)}")
            
            elif choice.upper() == 'A':
                print("Analysis mode has been removed to simplify the player setup.")
                
            else:
                print("ERROR: Invalid choice. Please select 0-9, V, or P.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            print("Press Enter to continue...")
            input()

def test_time_index_generation():
    """Test the new time indexing functionality"""
    print("Time Index Generation Test")
    print("-" * 30)
    
    # Test with example_data directory
    data_dir = "example_data"
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        return
        
    try:
        from aris_utilities import ARISTimeIndexer
        indexer = ARISTimeIndexer()
        
        print(f"Generating time index for: {data_dir}")
        T, d = indexer.generate_time_index(data_dir)
        
        print(f"Time index generated successfully!")
        print(f"   Total frames: {len(T)}")
        print(f"   Files processed: {len(d)}")
        print(f"   Duration: {T[-1,0] - T[0,0]:.1f} seconds")
        print(f"   Frame rate: {len(T)/(T[-1,0] - T[0,0] + 1/12):.1f} FPS")
        
        print(f"\nSample data (first 5 rows):")
        print(f"   [Time(s), Ping#, File#]")
        for i in range(min(5, len(T))):
            print(f"   [{T[i,0]:8.3f}, {T[i,1]:5.0f}, {T[i,2]:3.0f}]")
            
        # Save both .mat and .csv files
        mat_path, csv_path = indexer.save_time_index(data_dir, T, d, "demo_time_index.mat")
        print(f"\nFiles created:")
        print(f"   MATLAB format: {mat_path}")
        print(f"   CSV format: {csv_path}")
        
        # Show CSV advantages
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nCSV advantages:")
        print(f"   • Easy to open in Excel/Google Sheets")
        print(f"   • {len(df.columns)} useful columns: {', '.join(df.columns[:4])}...")
        print(f"   • Human-readable format")
        
    except Exception as e:
        print(f"ERROR: {e}")

def test_beam_angle_conversion(filename):
    """Test the new beam angle conversion functionality"""
    print("Beam Angle Conversion Test")
    print("-" * 30)
    
    try:
        from aris_python_api import CompleteARISReader
        
        with CompleteARISReader(filename) as reader:
            # Test some sample angles
            test_angles = [-10, -5, 0, 5, 10]
            print(f"Testing angles: {test_angles}")
            
            beams = reader.get_aris_beam_from_angle(test_angles)
            print(f"Corresponding beams: {beams}")
            
            print("\nDetailed mapping:")
            for angle, beam in zip(test_angles, beams):
                print(f"   Angle {angle:6.1f}° → Beam {beam:2d}")
                
            print("Beam angle conversion working correctly!")
            
    except Exception as e:
        print(f"ERROR: {e}")

def test_advanced_interpolation(filename):
    """Test the new advanced interpolation functionality"""
    print("Advanced Interpolation Test")
    print("-" * 30)
    
    try:
        from aris_python_api import CompleteARISReader
        
        with CompleteARISReader(filename) as reader:
            # Get a frame to test with
            frame_data = reader.get_frame_new(1)
            raw_frame = frame_data['frame']
            
            print(f"Original frame size: {raw_frame.shape}")
            
            # Test different interpolation methods
            methods = ['expand', 'linear', 'cubic', 'nearest']
            factor = 4
            
            for method in methods:
                try:
                    interpolated = reader.smooth1(raw_frame, factor, method)
                    print(f"   {method:8s}: {raw_frame.shape} → {interpolated.shape} OK")
                except Exception as e:
                    print(f"   {method:8s}: ERROR: {e}")
                    
            print("Advanced interpolation testing complete!")
            
    except Exception as e:
        print(f"ERROR: {e}")

def test_all_functions(filename):
    """Test all MATLAB equivalent functions"""
    print("Testing all MATLAB equivalent functions...")
    
    try:
        # Create organized output directory with filename prefix
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        with CompleteARISReader(filename) as reader:
            # Test basic reading
            print("  ✓ Testing get_frame_first...")
            data = reader.get_frame_first()
            
            print("  ✓ Testing make_first_image...")
            data = reader.make_first_image(smooth=4, imagexsize=300)
            
            print("  ✓ Testing get_frame_new...")
            data = reader.get_frame_new(10)
            
            print("  ✓ Testing make_new_image...")
            data = reader.make_new_image(data['frame'])
            
            print("  ✓ Testing motion_correction...")
            corrected = reader.motion_correction(velocity=1.5)
            
            print("  ✓ Testing export functions...")
            # Small test exports with organized naming
            reader.convert_to_images(f"{base_name}_test_images", frame_range=[1, 3])
            reader.convert_to_numpy(f"{base_name}_test_data.npz", frame_range=[1, 5])
            
        print("All functions tested successfully!")
        print("   Core reading: OK")
        print("   Image processing: OK") 
        print("   Motion correction: OK")
        print("   Export functions: OK")
        print(f"   Test outputs organized with prefix: {base_name}_")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 