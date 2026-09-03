"""
NeuroNav GUI interface entrypoint
The GUI application has been organized into the 'gui/' directory:
    gui/gui_app.py
The ML engine has been organized into the 'ml_engine/' directory:
    ml_engine/ml_engine.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.gui_app import NeuroNavApp, main

if __name__ == '__main__':
    main()
