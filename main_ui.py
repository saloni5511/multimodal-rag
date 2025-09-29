#!/usr/bin/env python3
"""
Simple launcher for the Streamlit RAG app.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "frontend" / "streamlit_app.py"
    
    print("🚀 Starting PDF RAG Streamlit App...")
    print(f"📂 App location: {app_path}")
    print("🌐 The app will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()