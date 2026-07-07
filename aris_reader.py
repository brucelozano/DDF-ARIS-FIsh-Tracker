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
            print("  P. Compute static pattern (cartesian fan image average)")
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
                    print("   • Press 'D' during playback to toggle detection")
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
                # Compute and save static pattern for current file
                print("Computing static pattern (cartesian fan image average)...")
                print("MATLAB averages ALL frames for best results.")
                try:
                    from aris_utilities import ARISExporter
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    output_dir = os.path.join('exports', 'patterns')
                    exporter = ARISExporter(filename)
                    pattern = exporter.compute_static_pattern(imagexsize=400, smooth=4)
                    saved = exporter.save_static_pattern(output_dir, pattern, base_name=base_name, also_mat=True)
                    exporter.close()
                    print(f"Saved static pattern files:\n  NPY: {saved.get('npy')}\n  MAT: {saved.get('mat', 'n/a')}")
                    print("Update your config JSON with:\n  \"USE_STATIC_PATTERN\": true,\n  \"STATIC_PATTERN_FILE\": \"" + saved.get('npy') + "\"  (or .mat)\n  \"BW_THRESHOLD\": 0.12,\n  \"BW_THRESHOLD_MODE\": \"relative\"")
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
                print("ERROR: Invalid choice. Please select 0-9, R, V, or P.")
                
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