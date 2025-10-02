#!/usr/bin/env python3
"""
ABOUTME: Main entry point for Badminton Stroke Classifier
ABOUTME: Launches the Gradio web interface for stroke classification
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the Gradio application"""
    try:
        from ui.gradio_app import GradioApp

        print("🏸 Starting Badminton Stroke Classifier...")
        print("🔧 Initializing models and interface...")

        # Create and launch the app
        app = GradioApp()
        interface = app.build().queue()

        print("🚀 Launching web interface...")
        print("📱 Access the app at: http://127.0.0.1:7860")
        print("🌐 Public access enabled via share link")
        print()
        print("📝 Instructions:")
        print("   1. Upload a badminton video (0.5-30 seconds)")
        print("   2. Adjust confidence threshold if needed")
        print("   3. Click 'Analyze Stroke' and wait for results")
        print("   4. View prediction, confidence, and technical details")
        print()
        print("❌ Press Ctrl+C to stop the server")

        # Launch the interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=True,  # Enable public sharing
            debug=True,
            show_error=True,
            quiet=False
        )

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you've installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"❌ File Not Found: {e}")
        print("💡 Make sure you've downloaded the model weights:")
        print("   - Check weights/ folder for bst_model.pt and tracknet_model.pt")
        print("   - See README.md for download instructions")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("💡 Please check the error above and ensure:")
        print("   - All dependencies are installed")
        print("   - Model weights are in weights/ folder")
        print("   - You have sufficient system resources")
        sys.exit(1)

if __name__ == "__main__":
    main()