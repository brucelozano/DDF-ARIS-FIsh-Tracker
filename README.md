# ARIS Python Reader

A Python tool for reading and analyzing ARIS (Acoustic Rapid Imaging Sonar) files with fish detection and tracking capabilities.

## QUICK START

1. Install Python 3.7 or higher
   Download from: https://www.python.org/downloads/

2. Install required packages:
   
   Open terminal/command prompt in this folder and run:
   
   pip install -r requirements.txt
   
   Or install manually:
   pip install numpy>=1.21.0 opencv-python>=4.5.0 scipy>=1.7.0 matplotlib>=3.3.0

3. Run the ARIS Reader:
   
   python aris_reader.py
   
   Recommended to compute a static pattern using option P for better results

   Then select Option 1 to play and analyze ARIS files. When the blank window pops up,
   press space to begin playing

   Switch files by using option 9.
   
   Use option V to visualize tracking results after playback completes.

## CONFIGURATION

- config/ - Sample configuration files
  - Copy and rename for your own ARIS files
  - Format: config_[videoname].json

## NEED HELP?

Email: brlozano@fiu.edu