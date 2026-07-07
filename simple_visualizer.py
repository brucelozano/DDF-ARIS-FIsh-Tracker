#!/usr/bin/env python3
"""
Simple Tracking Data Visualizer
Creates basic visualizations from offline analysis CSV files using only matplotlib
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_tracking_data(tracking_file):
    """Load tracking data from CSV file"""
    if not os.path.exists(tracking_file):
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")
    
    data = []
    with open(tracking_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    'frame': int(row['Frame Index']),
                    'fish_id': int(row['Fish ID']),
                    'x': float(row['X Position (m)']),
                    'y': float(row['Y Position (m)']),
                    'velocity': float(row['Velocity (cm/s)']),
                    'direction': float(row['Velocity Direction (degrees)']),
                    'length': float(row['Fish Length (cm)'])
                })
            except (ValueError, KeyError):
                continue  # Skip malformed rows
    
    if len(data) == 0:
        raise ValueError("No valid tracking data found in file")
    
    unique_fish = len(set(row['fish_id'] for row in data))
    print(f"Loaded {len(data)} tracking points for {unique_fish} unique fish")
    return data


def create_trajectory_plot(data, base_name):
    """Create simple trajectory plot"""
    plt.figure(figsize=(12, 8))
    
    # Group data by fish ID
    fish_tracks = {}
    for row in data:
        fish_id = row['fish_id']
        if fish_id not in fish_tracks:
            fish_tracks[fish_id] = []
        fish_tracks[fish_id].append(row)
    
    # Sort each track by frame
    for fish_id in fish_tracks:
        fish_tracks[fish_id].sort(key=lambda x: x['frame'])
    
    # Plot trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, len(fish_tracks)))
    
    for i, (fish_id, track) in enumerate(fish_tracks.items()):
        if len(track) < 2:
            continue
            
        x_pos = [point['x'] for point in track]
        y_pos = [point['y'] for point in track]
        
        # Plot trajectory (no label - too many fish for legend)
        plt.plot(x_pos, y_pos, color=colors[i % len(colors)], alpha=0.5, linewidth=1)
        
        # Mark start and end (smaller markers for cleaner look)
        plt.scatter(x_pos[0], y_pos[0], color=colors[i % len(colors)], marker='o', s=20, 
                   edgecolor='black', linewidth=0.5, zorder=5, alpha=0.7)
        plt.scatter(x_pos[-1], y_pos[-1], color=colors[i % len(colors)], marker='s', s=20, 
                   edgecolor='black', linewidth=0.5, zorder=5, alpha=0.7)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Fish Trajectories - {base_name}\n({len(fish_tracks)} fish tracked, Circles=Start, Squares=End)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = "exports/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_trajectories.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved trajectory plot: {output_file}")
    return output_file


def create_heatmap(data, base_name):
    """Create density heatmap with logarithmic scale for better visualization"""
    from matplotlib.colors import LogNorm
    
    plt.figure(figsize=(10, 8))
    
    x_pos = [row['x'] for row in data]
    y_pos = [row['y'] for row in data]
    
    # Use more bins and logarithmic scale to show variation across density ranges
    h, xedges, yedges, img = plt.hist2d(
        x_pos, y_pos, 
        bins=80,  # More bins for finer resolution
        cmap='inferno',  # Better colormap for density
        norm=LogNorm(vmin=1),  # Log scale to show low-density areas
        cmin=1  # Don't show empty bins
    )
    plt.colorbar(label='Fish Detection Density (log scale)')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Fish Activity Heatmap - {base_name}\n({len(data)} tracking points)')
    plt.grid(True, alpha=0.3, color='white')
    
    # Save plot
    output_dir = "exports/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved heatmap: {output_file}")
    return output_file


def create_summary_report(data, base_name):
    """Create text summary report"""
    # Calculate statistics
    velocities = [row['velocity'] for row in data if row['velocity'] > 0]
    lengths = [row['length'] for row in data if row['length'] > 0]
    unique_fish = len(set(row['fish_id'] for row in data))
    frames = [row['frame'] for row in data]
    
    # Track lengths
    fish_tracks = {}
    for row in data:
        fish_id = row['fish_id']
        if fish_id not in fish_tracks:
            fish_tracks[fish_id] = 0
        fish_tracks[fish_id] += 1
    
    track_lengths = list(fish_tracks.values())
    
    # Create report
    report_lines = [
        f"FISH TRACKING ANALYSIS REPORT",
        f"=" * 50,
        f"Dataset: {base_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "BASIC STATISTICS:",
        f"   Total tracking points: {len(data)}",
        f"   Unique fish tracked: {unique_fish}",
        f"   Frame range: {min(frames)} - {max(frames)}",
        f"   Duration: {max(frames) - min(frames) + 1} frames",
        ""
    ]
    
    if velocities:
        report_lines.extend([
            "🏊 VELOCITY ANALYSIS:",
            f"   Mean velocity: {np.mean(velocities):.2f} cm/s",
            f"   Max velocity: {max(velocities):.2f} cm/s",
            f"   Velocity std: {np.std(velocities):.2f} cm/s",
            ""
        ])
    
    if lengths:
        report_lines.extend([
            "SIZE ANALYSIS:",
            f"   Mean fish length: {np.mean(lengths):.2f} cm",
            f"   Max fish length: {max(lengths):.2f} cm",
            f"   Length std: {np.std(lengths):.2f} cm",
            ""
        ])
    
    if track_lengths:
        report_lines.extend([
            "TRACK QUALITY:",
            f"   Mean track length: {np.mean(track_lengths):.1f} frames",
            f"   Longest track: {max(track_lengths)} frames",
            f"   Tracks >10 frames: {sum(1 for x in track_lengths if x > 10)}",
            f"   Tracks >20 frames: {sum(1 for x in track_lengths if x > 20)}"
        ])
    
    # Save and display report
    report_text = "\n".join(report_lines)
    output_dir = "exports/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_summary_report.txt")
    
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved summary report: {output_file}")
    return output_file


def find_analysis_files(statistics_dir="exports/statistics"):
    """Find all available analysis files"""
    files = {}
    if not os.path.exists(statistics_dir):
        return files
        
    for filename in os.listdir(statistics_dir):
        if filename.endswith('_tracking_info.csv'):
            base_name = filename.replace('_tracking_info.csv', '')
            tracking_file = os.path.join(statistics_dir, filename)
            
            # Check if file has data (more than just header)
            try:
                with open(tracking_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Header + at least one data row
                        files[base_name] = tracking_file
            except:
                continue
                
    return files


def visualize_dataset(base_name, tracking_file):
    """Create all visualizations for a dataset"""
    print(f"Creating visualizations for: {base_name}")
    
    try:
        # Load data
        data = load_tracking_data(tracking_file)
        
        # Create visualizations
        create_trajectory_plot(data, base_name)
        create_heatmap(data, base_name)
        create_summary_report(data, base_name)
        
        print(f"\nAll visualizations complete!")
        print(f"Output directory: exports/visualizations/")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")


def interactive_visualizer():
    """Interactive visualization menu"""
    print("Fish Tracking Data Visualizer")
    print("=" * 40)
    
    # Find available datasets
    files = find_analysis_files()
    
    if not files:
        print("No tracking data found in exports/statistics/")
        print("   Run offline analysis first (Option R)")
        return
    
    print("Available datasets:")
    datasets = list(files.keys())
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    print("  0. Visualize all datasets")
    
    try:
        choice = input(f"\nSelect dataset (0-{len(datasets)}): ").strip()
        
        if choice == '0':
            # Visualize all datasets
            for base_name, tracking_file in files.items():
                print(f"\n{'='*60}")
                visualize_dataset(base_name, tracking_file)
        else:
            # Visualize specific dataset
            dataset_idx = int(choice) - 1
            if 0 <= dataset_idx < len(datasets):
                base_name = datasets[dataset_idx]
                tracking_file = files[base_name]
                visualize_dataset(base_name, tracking_file)
            else:
                print("Invalid selection!")
                
    except (ValueError, KeyboardInterrupt):
        print("Visualization cancelled.")


if __name__ == "__main__":
    interactive_visualizer()
