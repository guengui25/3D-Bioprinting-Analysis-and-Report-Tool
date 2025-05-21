"""
Launcher script for the 3D Bioprinting Analysis and Report Tool
"""

import os
import sys
from app.app_gui import ProcessingApp

if __name__ == "__main__":
    # Add the current directory to PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    # Start the application
    app = ProcessingApp()
    app.mainloop()
