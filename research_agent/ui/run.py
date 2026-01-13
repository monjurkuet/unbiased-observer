#!/usr/bin/env python3
"""
Run script for the Streamlit-based Research Agent UI
"""

import subprocess
import sys
import os


def main():
    """Run Streamlit application"""

    # Get root directory of project (go up three levels from ui/)
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(ui_dir))

    # Change to UI directory
    os.chdir(ui_dir)

    # Add project root to Python path for imports
    sys.path.insert(0, project_root)

    # Run streamlit using uv
    cmd = [
        "uv",
        "run",
        "--",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
    ]

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
