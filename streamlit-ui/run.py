#!/usr/bin/env python3
"""
Run script for the Streamlit-based Research Agent UI
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add parent directory to Python path for imports
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
    
    print("🚀 Starting Autonomous Research Agent Web Interface")
    print("=" * 60)
    print("📍 Access the interface at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
