#!/usr/bin/env python3
"""
Start the AI Meeting Assistant with file upload features on port 8002
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting AI Meeting Assistant with File Upload Features...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("demo.py"):
        print("❌ demo.py not found! Please run this from the project directory.")
        return False
    
    # Install aiofiles if not already installed
    print("📦 Installing required dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "aiofiles"], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Start the server
    print("🚀 Starting server on port 8002...")
    print("📱 Open your browser to: http://localhost:8002")
    print("🔧 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
