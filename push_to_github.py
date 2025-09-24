#!/usr/bin/env python3
"""
Script to push changes to GitHub when terminal is broken
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/Users/neha/AI-Meeting-Assistant-Agentic-RAG-V1-")
        if result.returncode == 0:
            print(f"✅ {description} - Success!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed!")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    print("🚀 Pushing AI Meeting Assistant changes to GitHub...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("/Users/neha/AI-Meeting-Assistant-Agentic-RAG-V1-/demo.py"):
        print("❌ Wrong directory! Please run this from the project root.")
        return False
    
    # Step 1: Check git status
    if not run_command("git status", "Checking git status"):
        return False
    
    # Step 2: Add all changes
    if not run_command("git add .", "Adding all changes"):
        return False
    
    # Step 3: Commit changes
    commit_message = """feat: Integrate co-pilot-nexus UI with AI Meeting Assistant

- Complete UI overhaul with modern dark theme
- Professional purple/blue gradient design system  
- Enhanced components: header, meeting controls, AI suggestions
- Responsive design with smooth animations
- Updated demo.py with co-pilot-nexus integration
- Changed server port to 8001 to avoid conflicts
- Fixed Pydantic deprecation warnings
- Added proper WebSocket handling for new UI
- Integrated co-pilot-nexus-main folder with all components"""
    
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        return False
    
    # Step 4: Push to GitHub
    if not run_command("git push origin main", "Pushing to GitHub"):
        return False
    
    print("=" * 60)
    print("🎉 SUCCESS! All changes have been pushed to GitHub!")
    print("🌐 Check your repository: https://github.com/your-username/AI-Meeting-Assistant-Agentic-RAG-V1-")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
