"""
NeuroNav GUI Application Launcher
Run this script to launch the Tkinter GUI interface for GNSS satellite orbit & clock error forecasting.
Usage:
    python app.py
"""
import os
import sys
from pathlib import Path

# Add root project directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from gui_app import main

if __name__ == "__main__":
    main()
