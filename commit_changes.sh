#!/bin/bash

# Script to commit co-pilot-nexus UI integration changes
echo "🚀 Committing co-pilot-nexus UI integration changes..."

# Navigate to the project directory
cd /Users/neha/AI-Meeting-Assistant-Agentic-RAG-V1-

# Check git status
echo "📋 Checking git status..."
git status

# Add all changes
echo "➕ Adding all changes..."
git add .

# Commit with descriptive message
echo "💾 Committing changes..."
git commit -m "feat: Integrate co-pilot-nexus UI with AI Meeting Assistant

- Complete UI overhaul with modern dark theme
- Professional purple/blue gradient design system
- Enhanced components: header, meeting controls, AI suggestions
- Responsive design with smooth animations
- Updated demo.py with co-pilot-nexus integration
- Changed server port to 8001 to avoid conflicts
- Fixed Pydantic deprecation warnings
- Added proper WebSocket handling for new UI
- Integrated co-pilot-nexus-main folder with all components"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Changes committed and pushed to GitHub successfully!"
echo "🌐 Check your repository at: https://github.com/your-username/AI-Meeting-Assistant-Agentic-RAG-V1-"
